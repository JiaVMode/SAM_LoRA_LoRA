#!/usr/bin/env python3
"""
AutoSAM 2D Training for Thymoma Segmentation
基于 AutoSAM 的自动分割训练，无需手动 prompt
"""

import os
import sys
import argparse
import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F
from torch.utils.data import DataLoader
from tqdm import tqdm
from pathlib import Path

# 添加项目路径
sys.path.insert(0, str(Path(__file__).parent))

from models.model_single import ModelEmb, MaskRefinement2D
from dataset.thymoma_dataset import get_thymoma_dataset_2d
from segment_anything import sam_model_registry
from segment_anything.utils.transforms import ResizeLongestSide


device = torch.device("cuda" if torch.cuda.is_available() else "cpu")


def norm_batch(x):
    """Normalize batch to [0, 1]"""
    bs = x.shape[0]
    Isize = x.shape[-1]
    min_value = x.view(bs, -1).min(dim=1)[0].view(bs, 1, 1, 1)
    max_value = x.view(bs, -1).max(dim=1)[0].view(bs, 1, 1, 1)
    x = (x - min_value) / (max_value - min_value + 1e-6)
    return x


def dice_loss(y_pred, y_true, smooth=1e-6):
    """Dice Loss (y_pred is logits)"""
    y_pred = y_pred.sigmoid().clamp(0, 1)
    y_true = y_true.clamp(0, 1)
    
    intersection = (y_pred * y_true).sum(dim=(2, 3))
    union = y_pred.sum(dim=(2, 3)) + y_true.sum(dim=(2, 3))
    
    dice = (2 * intersection + smooth) / (union + smooth)
    return 1 - dice.mean()


def calculate_dice(pred, target):
    """Calculate Dice coefficient"""
    pred = (pred > 0.5).float()
    target = (target > 0.5).float()
    
    intersection = (pred * target).sum()
    union = pred.sum() + target.sum()
    
    if union == 0:
        return 1.0
    return (2 * intersection / union).item()


def sam_forward(sam, image, dense_embeddings):
    """SAM forward pass with custom dense embeddings"""
    with torch.no_grad():
        # 获取图像 embedding
        image_embeddings = sam.image_encoder(image)
        
        # 获取空的 sparse embeddings
        sparse_embeddings, _ = sam.prompt_encoder(
            points=None, boxes=None, masks=None
        )
    
    # 使用自定义的 dense embeddings (来自 ModelEmb)
    low_res_masks, iou_predictions = sam.mask_decoder(
        image_embeddings=image_embeddings,
        image_pe=sam.prompt_encoder.get_dense_pe(),
        sparse_prompt_embeddings=sparse_embeddings,
        dense_prompt_embeddings=dense_embeddings,
        multimask_output=False,
    )
    
    return low_res_masks, iou_predictions


class AutoSAMTrainer:
    """AutoSAM 训练器"""
    
    def __init__(self, args):
        self.args = args
        self.device = device
        
        # 创建输出目录
        self.output_dir = Path(args['output_dir'])
        self.output_dir.mkdir(parents=True, exist_ok=True)
        
        # 初始化 SAM
        print(f"加载 SAM: {args['sam_checkpoint']}")
        self.sam = sam_model_registry[args['model_type']](
            checkpoint=args['sam_checkpoint']
        )
        self.sam.to(device)
        self.sam.eval()  # SAM 保持冻结
        
        # 冻结 SAM 参数
        for param in self.sam.parameters():
            param.requires_grad = False
        
        # 初始化 ModelEmb (可训练)
        print("初始化 ModelEmb...")
        self.model = ModelEmb(args).to(device)
        
        # 可选的 mask 细化层
        self.mask_refine = MaskRefinement2D(1, 1).to(device)
        
        # LoRA 模式
        self.use_lora = args.get('use_lora', False)
        if self.use_lora:
            from models.lora import apply_lora_to_model, get_lora_parameters
            lora_rank = int(args.get('lora_rank', 8))
            lora_alpha = float(args.get('lora_alpha', 16.0))
            
            print(f"[LoRA] 启用 LoRA 微调模式 (rank={lora_rank}, alpha={lora_alpha})")
            
            # 先冻结所有参数
            for param in self.model.parameters():
                param.requires_grad = False
            for param in self.mask_refine.parameters():
                param.requires_grad = False
            # SAM 已经在上面被冻结了
            
            # 应用 LoRA
            # 1. ModelEmb & MaskRefine (Conv2d 密集)
            self.model = apply_lora_to_model(self.model, rank=lora_rank, alpha=lora_alpha, apply_to_conv=True)
            self.mask_refine = apply_lora_to_model(self.mask_refine, rank=lora_rank, alpha=lora_alpha, apply_to_conv=True)
            
            # 2. SAM (Linear 密集 - Attention)
            # 我们只对 SAM 的 Encoder 和 Mask Decoder 应用 LoRA，Prompt Encoder 保持冻结
            print("[LoRA] 对 SAM 应用 LoRA...")
            self.sam.image_encoder = apply_lora_to_model(self.sam.image_encoder, rank=lora_rank, alpha=lora_alpha, apply_to_conv=False)
            self.sam.mask_decoder = apply_lora_to_model(self.sam.mask_decoder, rank=lora_rank, alpha=lora_alpha, apply_to_conv=False)
            
            # 收集所有 LoRA 参数
            trainable_params = (
                get_lora_parameters(self.model) + 
                get_lora_parameters(self.mask_refine) + 
                get_lora_parameters(self.sam.image_encoder) + 
                get_lora_parameters(self.sam.mask_decoder)
            )
            print(f"[LoRA] 可训练参数总数: {sum(p.numel() for p in trainable_params):,}")
        else:
            # 全量微调模式
            trainable_params = list(self.model.parameters()) + list(self.mask_refine.parameters())
        
        # 优化器
        self.optimizer = optim.AdamW(
            trainable_params,
            lr=float(args['learning_rate']),
            weight_decay=float(args['weight_decay']),
        )
        
        # 学习率调度器
        self.scheduler = optim.lr_scheduler.CosineAnnealingLR(
            self.optimizer, T_max=int(args['epochs'])
        )
        
        # 损失函数 - BCEWithLogitsLoss (AMP safe, 不使用 pos_weight 以避免不稳定)
        self.bce_loss = nn.BCEWithLogitsLoss()
        

        
        # AMP 混合精度
        self.use_amp = args.get('use_amp', True)
        self.scaler = torch.cuda.amp.GradScaler(enabled=self.use_amp)
        if self.use_amp:
            print("启用 AMP 混合精度训练")
        
        # SAM 变换
        self.sam_transform = ResizeLongestSide(self.sam.image_encoder.img_size)
        
        # 加载数据
        print("加载数据...")
        self.train_dataset, self.val_dataset = get_thymoma_dataset_2d(args, self.sam_transform)
        
        self.train_loader = DataLoader(
            self.train_dataset,
            batch_size=int(args['batch_size']),
            shuffle=True,
            num_workers=int(args['num_workers']),
            drop_last=True,
            pin_memory=True,
        )
        
        self.val_loader = DataLoader(
            self.val_dataset,
            batch_size=int(args.get('val_batch_size', args['batch_size'])),  # 使用配置的 batch size
            shuffle=False,
            num_workers=int(args['num_workers']),
        )
        
        self.best_dice = 0
        
        # 统计参数量
        total_params = sum(p.numel() for p in trainable_params if p.requires_grad)
        print(f"可训练参数: {total_params:,}")
    
    def train_epoch(self, epoch):
        """训练一个 epoch"""
        self.model.train()
        self.mask_refine.train()
        
        losses = []
        dices = []
        
        pbar = tqdm(self.train_loader, desc=f"Epoch {epoch}")
        
        for batch_idx, (images, masks, _, _) in enumerate(pbar):
            images = images.to(self.device)
            masks = masks.to(self.device).unsqueeze(1)  # (B, 1, H, W)
            
            # ModelEmb 需要较小的输入尺寸
            Idim = int(self.args.get('Idim', 256))
            images_small = F.interpolate(images, (Idim, Idim), mode='bilinear', align_corners=True)
            
            # AMP 混合精度
            with torch.cuda.amp.autocast(enabled=self.use_amp):
                # 获取 dense embeddings
                dense_embeddings = self.model(images_small)
                
                # SAM 前向传播
                low_res_masks, _ = sam_forward(self.sam, images, dense_embeddings)
                
                # 可选的 mask 细化
                refined_masks = self.mask_refine(low_res_masks)
                
                # 调整 GT 尺寸匹配预测
                masks_resized = F.interpolate(masks, refined_masks.shape[-2:], mode='nearest')
                
                # 计算损失
                loss_bce = self.bce_loss(refined_masks, masks_resized)
                loss_dice = dice_loss(refined_masks, masks_resized)
                # 提高 Dice Loss 权重，降低 BCE 权重 (背景太多)
                loss = 0.5 * loss_bce + loss_dice
            
            # 反向传播 (使用 GradScaler)
            self.optimizer.zero_grad()
            self.scaler.scale(loss).backward()
            self.scaler.unscale_(self.optimizer)
            torch.nn.utils.clip_grad_norm_(self.model.parameters(), 1.0)
            self.scaler.step(self.optimizer)
            self.scaler.update()
            
            # 计算 Dice
            with torch.no_grad():
                pred_binary = (refined_masks.sigmoid() > 0.5).float()  # 应用 sigmoid 后阈值化
                dice = calculate_dice(pred_binary, masks_resized)
                
                # 调试输出 (每 100 个 batch)
                if batch_idx % 10 == 0:
                    pred_sum = pred_binary.sum().item()
                    gt_sum = masks_resized.sum().item()
                    pred_max = refined_masks.max().item()
                    pred_min = refined_masks.min().item()
                    tqdm.write(f"  [Debug] Pred sum: {pred_sum:.0f}, GT sum: {gt_sum:.0f}, Pred prob: [{pred_min:.2f}, {pred_max:.2f}]")
            
            losses.append(loss.item())
            dices.append(dice)
            
            pbar.set_postfix({
                'loss': f'{np.mean(losses[-50:]):.4f}',
                'dice': f'{np.mean(dices[-50:]):.4f}',
            })
        
        return np.mean(losses), np.mean(dices)
    
    @torch.no_grad()
    def validate(self, epoch):
        """验证"""
        self.model.eval()
        self.mask_refine.eval()
        
        dices = []
        
        pbar = tqdm(self.val_loader, desc="Validation")
        
        for images, masks, _, _ in pbar:
            images = images.to(self.device)
            masks = masks.to(self.device).unsqueeze(1)
            
            Idim = int(self.args.get('Idim', 256))
            images_small = F.interpolate(images, (Idim, Idim), mode='bilinear', align_corners=True)
            
            dense_embeddings = self.model(images_small)
            low_res_masks, _ = sam_forward(self.sam, images, dense_embeddings)
            refined_masks = self.mask_refine(low_res_masks)
            
            masks_resized = F.interpolate(masks, refined_masks.shape[-2:], mode='nearest')
            
            pred_binary = (refined_masks.sigmoid() > 0.5).float()
            dice = calculate_dice(pred_binary, masks_resized)
            dices.append(dice)
            
            pbar.set_postfix({'dice': f'{np.mean(dices):.4f}'})
        
        return np.mean(dices)
    
    def save_checkpoint(self, epoch, dice, is_best=False):
        """保存检查点"""
        checkpoint = {
            'epoch': epoch,
            'epoch': epoch,
            'model_state_dict': self.model.state_dict(),
            'mask_refine_state_dict': self.mask_refine.state_dict(),
            # 如果用了 LoRA，也保存 SAM 的 LoRA 权重
            'sam_lora_state_dict': self.sam.state_dict() if self.use_lora else None,
            'optimizer_state_dict': self.optimizer.state_dict(),
            'dice': dice,
            'args': self.args,
            'use_lora': self.use_lora,  # 记录是否使用 LoRA
        }
        
        # 保存最新
        torch.save(checkpoint, self.output_dir / 'latest.pt')
        
        # 保存最佳
        if is_best:
            torch.save(checkpoint, self.output_dir / 'best.pt')
            print(f"💾 保存最佳模型: Dice = {dice:.4f}")
    
    def train(self):
        """完整训练流程"""
        epochs = int(self.args['epochs'])
        patience = int(self.args.get('patience', 5))  # Early Stopping 耐心值
        no_improve_count = 0
        
        print(f"\n{'='*50}")
        print(f"开始训练 AutoSAM ({epochs} epochs)")
        print(f"Early Stopping: patience={patience}")
        print(f"{'='*50}\n")
        
        for epoch in range(1, epochs + 1):
            # 训练
            train_loss, train_dice = self.train_epoch(epoch)
            
            # 验证
            val_dice = self.validate(epoch)
            
            # 学习率调度
            self.scheduler.step()
            
            # 保存检查点
            is_best = val_dice > self.best_dice
            if is_best:
                self.best_dice = val_dice
                no_improve_count = 0  # 重置计数
            else:
                no_improve_count += 1
            
            self.save_checkpoint(epoch, val_dice, is_best)
            
            print(f"\nEpoch {epoch}/{epochs}:")
            print(f"  Train Loss: {train_loss:.4f}, Train Dice: {train_dice:.4f}")
            print(f"  Val Dice: {val_dice:.4f} {'(Best!)' if is_best else ''}")
            print(f"  LR: {self.scheduler.get_last_lr()[0]:.6f}")
            
            # Early Stopping 检查
            if no_improve_count >= patience:
                print(f"\n⚠️ Early Stopping: Val Dice 连续 {patience} 个 epoch 没有提升")
                print(f"最佳模型已保存 (Dice: {self.best_dice:.4f})")
                break
        
        print(f"\n训练完成! 最佳 Dice: {self.best_dice:.4f}")


def main():
    parser = argparse.ArgumentParser(description='AutoSAM 2D Training')
    
    # 配置文件
    parser.add_argument('--config', '-c', type=str, default=None,
                        help='YAML 配置文件路径')
    
    # 路径参数
    parser.add_argument('--dataset_path', type=str, 
                        default='./data/png_output',
                        help='数据集路径')
    parser.add_argument('--output_dir', type=str, default='./output',
                        help='输出目录')
    parser.add_argument('--sam_checkpoint', type=str,
                        default='./checkpoints/sam_vit_h.pth',
                        help='SAM 检查点路径')
    
    # 模型参数
    parser.add_argument('--model_type', type=str, default='vit_h',
                        choices=['vit_h', 'vit_l', 'vit_b'],
                        help='SAM 模型类型')
    parser.add_argument('--Idim', type=int, default=256,
                        help='ModelEmb 输入尺寸')
    parser.add_argument('--order', type=int, default=85,
                        help='HarDNet order')
    parser.add_argument('--depth_wise', type=bool, default=False,
                        help='使用 depth-wise 卷积')
    
    # 训练参数
    parser.add_argument('--epochs', type=int, default=50,
                        help='训练轮数')
    parser.add_argument('--batch_size', type=int, default=4,
                        help='批次大小')
    parser.add_argument('--learning_rate', type=float, default=1e-3,
                        help='学习率')
    parser.add_argument('--weight_decay', type=float, default=1e-4,
                        help='权重衰减')
    parser.add_argument('--num_workers', type=int, default=4,
                        help='数据加载线程数')
    
    args = parser.parse_args()
    
    # 从 YAML 加载配置
    if args.config:
        import yaml
        with open(args.config, 'r') as f:
            config = yaml.safe_load(f)
        
        # 用 YAML 配置覆盖默认值
        for key, value in config.items():
            if hasattr(args, key):
                setattr(args, key, value)
    
    args = vars(args)
    
    # 打印配置
    print("\n配置:")
    for k, v in args.items():
        if k != 'config':
            print(f"  {k}: {v}")
    print()
    
    # 创建训练器并开始训练
    trainer = AutoSAMTrainer(args)
    trainer.train()


if __name__ == '__main__':
    main()

