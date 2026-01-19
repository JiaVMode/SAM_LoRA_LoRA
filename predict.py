                                   #!/usr/bin/env python3
"""
AutoSAM 推理脚本 - 胸腺瘤自动分割

工作流程:
1. 读取 NIfTI 文件
2. 将每个切片保存为 PNG (原始)
3. 使用 AutoSAM 模型分割每个切片 (无需 Prompt)
4. 保存分割结果为 PNG
5. 合并所有分割结果为 NIfTI 文件

使用方法:
    python predict.py -i /path/to/ct.nii.gz --output_dir ./output/patient001
    
    # 指定模型权重
    python predict.py -i /path/to/ct.nii.gz --weights ./output/best.pt
"""

import argparse
import os
import sys
from pathlib import Path

import numpy as np
import torch
import torch.nn.functional as F
import cv2
import nibabel as nib
from tqdm import tqdm

sys.path.insert(0, str(Path(__file__).parent))

from models.model_single import ModelEmb, MaskRefinement2D
from segment_anything import sam_model_registry


class AutoSAMPredictor:
    """AutoSAM 推理器 - 自动分割无需 Prompt"""
    
    def __init__(
        self,
        sam_checkpoint: str = "./checkpoints/sam_vit_b.pth",
        weights: str = "./output/best.pt",
        model_type: str = "vit_b",
        device: str = "cuda",
        Idim: int = 256,
    ):
        self.device = torch.device(device)
        self.Idim = Idim
        
        # CT 窗宽窗位 (纵隔窗)
        self.window_center = 45.0
        self.window_width = 400.0
        
        # 加载 SAM
        print(f"加载 SAM: {sam_checkpoint}")
        self.sam = sam_model_registry[model_type](checkpoint=sam_checkpoint)
        self.sam.to(self.device)
        self.sam.eval()
        
        # 冻结 SAM
        for param in self.sam.parameters():
            param.requires_grad = False
        
        # 加载 ModelEmb + MaskRefine
        print(f"加载权重: {weights}")
        
        args = {
            'Idim': Idim,
            'order': 85,
            'depth_wise': False,
            'pretrained': True,  # 需要 True 来初始化 full_features，然后加载 checkpoint 覆盖
        }
        
        self.model = ModelEmb(args).to(self.device)
        self.mask_refine = MaskRefinement2D(1, 1).to(self.device)
        
        # 加载权重
        if os.path.exists(weights):
            checkpoint = torch.load(weights, map_location=self.device)
            
            # 检查是否是 LoRA 模型
            use_lora = checkpoint.get('use_lora', False)
            if use_lora:
                from models.lora import apply_lora_to_model
                saved_args = checkpoint.get('args', {})
                lora_rank = int(saved_args.get('lora_rank', 8))
                lora_alpha = float(saved_args.get('lora_alpha', 16.0))
                
                print(f"[LoRA] 检测到 LoRA 模型 (rank={lora_rank}, alpha={lora_alpha})")
                # 1. ModelEmb & MaskRefine (Conv2d 密集)
                self.model = apply_lora_to_model(self.model, rank=lora_rank, alpha=lora_alpha, apply_to_conv=True)
                self.mask_refine = apply_lora_to_model(self.mask_refine, rank=lora_rank, alpha=lora_alpha, apply_to_conv=True)
                
                # 2. SAM (如果 checkpoint 里有 sam_lora_state_dict)
                if 'sam_lora_state_dict' in checkpoint and checkpoint['sam_lora_state_dict'] is not None:
                    print("[LoRA] 检测到 SAM LoRA 权重，正在应用...")
                    self.sam.image_encoder = apply_lora_to_model(self.sam.image_encoder, rank=lora_rank, alpha=lora_alpha, apply_to_conv=False)
                    self.sam.mask_decoder = apply_lora_to_model(self.sam.mask_decoder, rank=lora_rank, alpha=lora_alpha, apply_to_conv=False)
                    
                    # 加载 SAM LoRA 权重
                    # 注意: 这里我们需要过滤出只属于 self.sam 的权重，或者直接加载 state_dict (如果只有 LoRA 参数)
                    # 由于 checkpoint['sam_lora_state_dict'] 是 self.sam.state_dict()，它包含所有参数
                    # 我们使用 strict=False 来只加载匹配的 LoRA 参数 (因为 LoRA 参数名是匹配的)
                    self.sam.load_state_dict(checkpoint['sam_lora_state_dict'], strict=False)
                    print("[LoRA] SAM LoRA 权重已加载")
            
            self.model.load_state_dict(checkpoint['model_state_dict'])
            self.mask_refine.load_state_dict(checkpoint['mask_refine_state_dict'])
            dice = checkpoint.get('dice', 'N/A')
            print(f"模型加载完成, Best Dice: {dice:.4f}" if isinstance(dice, float) else f"模型加载完成")
        else:
            print(f"警告: 权重文件不存在 {weights}, 使用随机初始化")
        
        self.model.eval()
        self.mask_refine.eval()
    
    def _normalize_slice(self, data: np.ndarray) -> np.ndarray:
        """归一化切片 (与 nii2png.py 第 38-52 行完全一致)"""
        data_min = np.min(data)
        data_max = np.max(data)
        
        if data_max <= 1.0 and data_min >= 0:
            # Binary mask
            normalized = data * 255.0
        elif data_max > 255:
            # CT or similar, simple normalization
            if data_max != data_min:
                normalized = (data - data_min) / (data_max - data_min) * 255.0
            else:
                normalized = data * 0  # All same value
        else:
            normalized = data
        
        return normalized.astype(np.uint8)
    
    def _preprocess(self, ct_slice: np.ndarray, size: int = 1024) -> torch.Tensor:
        """预处理 CT 切片 (与训练数据一致)"""
        # 使用与 nii2png.py 相同的归一化
        normalized = self._normalize_slice(ct_slice).astype(np.float32) / 255.0
        resized = cv2.resize(normalized, (size, size), interpolation=cv2.INTER_LINEAR)
        rgb = np.stack([resized, resized, resized], axis=0)
        return torch.from_numpy(rgb).float().unsqueeze(0)
    
    def _preprocess_normalized(self, ct_slice_uint8: np.ndarray, size: int = 1024) -> torch.Tensor:
        """预处理已归一化的 uint8 切片"""
        # 输入已经是 uint8 [0, 255]，转为 float [0, 1]
        normalized = ct_slice_uint8.astype(np.float32) / 255.0
        resized = cv2.resize(normalized, (size, size), interpolation=cv2.INTER_LINEAR)
        rgb = np.stack([resized, resized, resized], axis=0)
        return torch.from_numpy(rgb).float().unsqueeze(0)
    
    @torch.no_grad()
    def _segment_slice(self, image: torch.Tensor) -> np.ndarray:
        """分割单个切片 (AutoSAM - 无需 Prompt)"""
        image = image.to(self.device)
        
        # ModelEmb 需要小尺寸输入
        images_small = F.interpolate(image, (self.Idim, self.Idim), 
                                     mode='bilinear', align_corners=True)
        
        # 获取 dense embeddings (来自 ModelEmb)
        dense_embeddings = self.model(images_small)
        
        # SAM 前向传播
        image_embeddings = self.sam.image_encoder(image)
        sparse_embeddings, _ = self.sam.prompt_encoder(
            points=None, boxes=None, masks=None
        )
        
        low_res_masks, _ = self.sam.mask_decoder(
            image_embeddings=image_embeddings,
            image_pe=self.sam.prompt_encoder.get_dense_pe(),
            sparse_prompt_embeddings=sparse_embeddings,
            dense_prompt_embeddings=dense_embeddings,
            multimask_output=False,
        )
        
        # Mask 细化
        refined_masks = self.mask_refine(low_res_masks)
        
        # 上采样到原始尺寸
        masks = F.interpolate(refined_masks, size=(image.shape[2], image.shape[3]), 
                              mode='bilinear', align_corners=False)
        
        return (masks[0, 0].sigmoid().cpu().numpy() > 0.5).astype(np.uint8)
    
    def segment(
        self,
        input_path: str,
        output_dir: str,
        min_area: int = 100,
        save_png: bool = True,
    ):
        """
        分割 NIfTI 文件
        
        Args:
            input_path: 输入 NIfTI 文件路径
            output_dir: 输出目录
            min_area: 最小有效区域面积
            save_png: 是否保存 PNG 切片
        
        Returns:
            output_mask: 分割结果 numpy 数组
        """
        output_dir = Path(output_dir)
        output_dir.mkdir(parents=True, exist_ok=True)
        
        # 创建子目录
        if save_png:
            slices_dir = output_dir / "slices_original"
            masks_dir = output_dir / "slices_mask"
            slices_dir.mkdir(exist_ok=True)
            masks_dir.mkdir(exist_ok=True)
        
        # 读取 NIfTI
        print(f"读取: {input_path}")
        nii = nib.load(input_path)
        ct_data = nii.get_fdata()
        
        # 整个体积级别归一化 (与 nii2png.py 第 38-52 行完全一致)
        data_min = np.min(ct_data)
        data_max = np.max(ct_data)
        
        if data_max <= 1.0 and data_min >= 0:
            ct_normalized = ct_data * 255.0
        elif data_max > 255:
            if data_max != data_min:
                ct_normalized = (ct_data - data_min) / (data_max - data_min) * 255.0
            else:
                ct_normalized = ct_data * 0
        else:
            ct_normalized = ct_data
        
        ct_normalized = ct_normalized.astype(np.uint8)
        
        output_mask = np.zeros_like(ct_data, dtype=np.uint8)
        
        print(f"分割中... (共 {ct_data.shape[2]} 个切片)")
        
        for i in tqdm(range(ct_data.shape[2]), desc="处理切片"):
            # 使用已归一化的数据
            ct_slice = ct_normalized[:, :, i]
            
            # NIfTI -> PNG 方向匹配 (与 nii2png.py 第57行完全一致)
            # 使用逆时针旋转 90 度来匹配训练数据
            ct_slice_oriented = np.rot90(ct_slice)  # k=1 逆时针 90 度
            
            # 保存原始切片 PNG (已归一化，直接保存)
            if save_png:
                cv2.imwrite(str(slices_dir / f"slice_{i:04d}.png"), ct_slice_oriented)
            
            # 预处理和分割 (转为 [0,1] float)
            image = self._preprocess_normalized(ct_slice_oriented)
            pred = self._segment_slice(image)
            
            # Resize 回原始尺寸
            pred_resized = cv2.resize(
                pred.astype(np.float32),
                (ct_slice_oriented.shape[1], ct_slice_oriented.shape[0]),
                interpolation=cv2.INTER_NEAREST
            ).astype(np.uint8)
            
            # 逆旋转回 NIfTI 坐标系 (顺时针 90 度)
            pred_final = np.rot90(pred_resized, k=-1)
            
            # 保存分割结果 PNG
            if save_png:
                mask_png = pred_resized * 255
                cv2.imwrite(str(masks_dir / f"slice_{i:04d}.png"), mask_png)
            
            # 形态学后处理 - 消除噪点
            # 1. 开运算 (先腐蚀后膨胀) 去除小噪点
            kernel = np.ones((3, 3), np.uint8)
            pred_cleaned = cv2.morphologyEx(pred_final, cv2.MORPH_OPEN, kernel, iterations=2)
            
            # 2. 连通域分析 - 只保留最大连通域
            if np.sum(pred_cleaned) >= min_area:
                num_labels, labels, stats, centroids = cv2.connectedComponentsWithStats(pred_cleaned, connectivity=8)
                
                if num_labels > 1:
                    # 找最大连通域 (排除背景 index 0)
                    areas = stats[1:, cv2.CC_STAT_AREA]
                    largest_idx = np.argmax(areas) + 1
                    largest_area = areas[largest_idx - 1]
                    
                    # 只保留最大连通域 (如果足够大)
                    if largest_area >= min_area:
                        pred_cleaned = (labels == largest_idx).astype(np.uint8)
                    else:
                        pred_cleaned = np.zeros_like(pred_final)
                        
            output_mask[:, :, i] = pred_cleaned
        
        # 3D 连通域分析 - 只保留最大的 3D 连通域
        from scipy import ndimage
        
        print("执行 3D 连通域分析去噪...")
        labeled_array, num_features = ndimage.label(output_mask)
        
        if num_features > 1:
            # 找到最大的 3D 连通域
            component_sizes = ndimage.sum(output_mask, labeled_array, range(1, num_features + 1))
            largest_component = np.argmax(component_sizes) + 1
            
            # 只保留最大连通域
            output_mask = (labeled_array == largest_component).astype(np.uint8)
            print(f"  移除了 {num_features - 1} 个噪点区域")
        elif num_features == 1:
            print("  只有一个连通域，无需过滤")
        else:
            print("  警告：没有检测到任何区域")
        
        # 保存分割结果 NIfTI
        output_nii_path = output_dir / "segmentation.nii.gz"
        output_nii = nib.Nifti1Image(output_mask, nii.affine, nii.header)
        nib.save(output_nii, str(output_nii_path))
        
        # 统计信息
        roi_volume = np.sum(output_mask)
        roi_slices = np.sum(np.any(output_mask, axis=(0, 1)))
        
        print(f"\n=== 分割完成 ===")
        print(f"ROI 体积: {roi_volume:,} 体素")
        print(f"ROI 切片数: {roi_slices}")
        print(f"输出目录: {output_dir}")
        if save_png:
            print(f"  ├── slices_original/  ({ct_data.shape[2]} 张原始切片 PNG)")
            print(f"  ├── slices_mask/      ({ct_data.shape[2]} 张分割结果 PNG)")
        print(f"  └── segmentation.nii.gz  (分割结果 NIfTI)")
        
        return output_mask


def main():
    parser = argparse.ArgumentParser(description='AutoSAM NIfTI Inference')
    
    parser.add_argument('--input', '-i', type=str, required=True,
                        help='输入 NIfTI 文件路径')
    parser.add_argument('--output_dir', '-o', type=str, default=None,
                        help='输出目录 (默认: 自动生成)')
    parser.add_argument('--weights', '-w', type=str, default='./output/best.pt',
                        help='模型权重路径')
    parser.add_argument('--sam_checkpoint', type=str, default='./checkpoints/sam_vit_b.pth',
                        help='SAM 检查点路径')
    parser.add_argument('--model_type', type=str, default='vit_b',
                        choices=['vit_h', 'vit_l', 'vit_b'],
                        help='SAM 模型类型')
    parser.add_argument('--Idim', type=int, default=256,
                        help='ModelEmb 输入尺寸')
    parser.add_argument('--min_area', type=int, default=100,
                        help='最小有效区域面积')
    parser.add_argument('--no_png', action='store_true',
                        help='不保存 PNG 切片')
    
    args = parser.parse_args()
    
    # 自动生成输出目录
    if args.output_dir is None:
        input_name = Path(args.input).stem.replace('.nii', '')
        args.output_dir = f'./predictions/{input_name}'
    
    # 创建预测器
    predictor = AutoSAMPredictor(
        sam_checkpoint=args.sam_checkpoint,
        weights=args.weights,
        model_type=args.model_type,
        Idim=args.Idim,
    )
    
    # 执行分割
    predictor.segment(
        input_path=args.input,
        output_dir=args.output_dir,
        min_area=args.min_area,
        save_png=not args.no_png,
    )


if __name__ == '__main__':
    main()
