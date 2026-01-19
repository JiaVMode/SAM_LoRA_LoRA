"""
Thymoma Dataset for AutoSAM Training
适配现有的 PNG 切片数据格式

数据结构:
png_output/
  ├── 003-65/           # 患者组
  │   ├── 003000/       # Mask 目录 (小文件)
  │   │   └── slice_*.png
  │   ├── 003000-origin/ # CT 图像目录 (大文件)
  │   │   └── slice_*.png
  │   └── ...
  └── ...
"""

import os
import cv2
import numpy as np
import torch
from torch.utils.data import Dataset, DataLoader
from pathlib import Path
from typing import Tuple, Optional, List
import random


def augment_image_mask(img: np.ndarray, mask: np.ndarray) -> Tuple[np.ndarray, np.ndarray]:
    """
    数据增强函数 (同时对图像和 mask 应用相同变换)
    
    Args:
        img: 图像 (H, W, 3) float32 [0, 1]
        mask: 掩码 (H, W) float32 [0, 1]
    
    Returns:
        增强后的 (img, mask)
    """
    # 随机水平翻转 (50%)
    if random.random() > 0.5:
        img = np.fliplr(img).copy()
        mask = np.fliplr(mask).copy()
    
    # 随机垂直翻转 (50%)
    if random.random() > 0.5:
        img = np.flipud(img).copy()
        mask = np.flipud(mask).copy()
    
    # 随机 90 度旋转 (25% 每种)
    k = random.randint(0, 3)
    if k > 0:
        img = np.rot90(img, k=k).copy()
        mask = np.rot90(mask, k=k).copy()
    
    return img, mask


class ThymomaDataset2D(Dataset):
    """2D 切片数据集，用于 AutoSAM 训练"""
    
    def __init__(
        self,
        data_root: str,
        transform=None,
        sam_transform=None,
        train: bool = True,
        train_ratio: float = 0.8,
        target_size: int = 1024,
    ):
        """
        Args:
            data_root: PNG 数据根目录
            transform: 数据增强
            sam_transform: SAM 的 ResizeLongestSide 变换
            train: 是否为训练集
            train_ratio: 训练集比例
            target_size: 目标图像尺寸
        """
        self.data_root = Path(data_root)
        self.transform = transform
        self.sam_transform = sam_transform
        self.train = train
        self.target_size = target_size
        
        # 收集所有切片
        self.samples = []
        self._collect_samples(train_ratio)
        
        print(f"{'训练' if train else '验证'}集: {len(self.samples)} 个切片")
    
    def _collect_samples(self, train_ratio: float):
        """收集所有有效的切片对"""
        # 遍历所有患者组目录
        patient_groups = sorted([d for d in self.data_root.iterdir() if d.is_dir()])
        
        # 按患者组划分训练/验证
        n_train = int(len(patient_groups) * train_ratio)
        if self.train:
            groups = patient_groups[:n_train]
        else:
            groups = patient_groups[n_train:]
        
        for group_dir in groups:
            # 找到所有 case 目录 (不带 -origin 后缀的是 mask)
            all_dirs = [d.name for d in group_dir.iterdir() if d.is_dir()]
            
            # 识别 mask 目录和对应的 origin 目录
            for dir_name in all_dirs:
                if dir_name.endswith('-origin'):
                    continue  # 跳过 origin 目录
                
                mask_dir = group_dir / dir_name
                origin_dir = group_dir / f"{dir_name}-origin"
                
                if not origin_dir.exists():
                    continue
                
                # 收集所有切片
                mask_files = sorted(mask_dir.glob("slice_*.png"))
                
                for mask_file in mask_files:
                    origin_file = origin_dir / mask_file.name
                    
                    if origin_file.exists():
                        # 检查 mask 是否非空 (至少有 50 个非零像素)
                        mask_check = cv2.imread(str(mask_file), cv2.IMREAD_GRAYSCALE)
                        has_tumor = mask_check is not None and np.sum(mask_check > 127) >= 50
                        
                        # 策略: 保留所有正样本，随机保留 10% 负样本
                        if has_tumor or (np.random.rand() < 0.1):
                            # origin_file = CT 图像，mask_file = mask
                            self.samples.append((str(origin_file), str(mask_file)))
    
    def _load_image(self, path: str) -> np.ndarray:
        """加载并预处理图像"""
        img = cv2.imread(path, cv2.IMREAD_GRAYSCALE)
        if img is None:
            raise ValueError(f"无法加载图像: {path}")
        
        # 归一化到 [0, 1]
        img = img.astype(np.float32) / 255.0
        
        # 调整尺寸
        if img.shape[0] != self.target_size or img.shape[1] != self.target_size:
            img = cv2.resize(img, (self.target_size, self.target_size), 
                           interpolation=cv2.INTER_LINEAR)
        
        # 转为 3 通道 (SAM 需要 RGB)
        img = np.stack([img, img, img], axis=-1)  # (H, W, 3)
        
        return img
    
    def _load_mask(self, path: str) -> np.ndarray:
        """加载 mask"""
        mask = cv2.imread(path, cv2.IMREAD_GRAYSCALE)
        if mask is None:
            raise ValueError(f"无法加载 mask: {path}")
        
        # 二值化
        mask = (mask > 127).astype(np.float32)
        
        # 调整尺寸
        if mask.shape[0] != self.target_size or mask.shape[1] != self.target_size:
            mask = cv2.resize(mask, (self.target_size, self.target_size),
                            interpolation=cv2.INTER_NEAREST)
        
        return mask
    
    def __len__(self):
        return len(self.samples)
    
    def __getitem__(self, idx):
        img_path, mask_path = self.samples[idx]
        
        # 加载数据
        img = self._load_image(img_path)  # (H, W, 3)
        mask = self._load_mask(mask_path)  # (H, W)
        
        # 数据增强 (训练时自动应用)
        if self.train:
            if self.transform:
                img, mask = self.transform(img, mask)
            else:
                # 默认使用内置增强
                img, mask = augment_image_mask(img, mask)
        
        # 转为 tensor
        img = torch.from_numpy(img).permute(2, 0, 1).float()  # (3, H, W)
        mask = torch.from_numpy(mask).float()  # (H, W)
        
        original_size = torch.tensor([self.target_size, self.target_size])
        image_size = torch.tensor([self.target_size, self.target_size])
        
        return img, mask, original_size, image_size


class ThymomaDataset3D(Dataset):
    """3D 体积数据集，用于 AutoSAM 3D 训练"""
    
    def __init__(
        self,
        data_root: str,
        num_slices: int = 32,
        train: bool = True,
        train_ratio: float = 0.8,
        target_size: int = 256,
    ):
        self.data_root = Path(data_root)
        self.num_slices = num_slices
        self.train = train
        self.target_size = target_size
        
        self.volumes = []
        self._collect_volumes(train_ratio)
        
        print(f"{'训练' if train else '验证'}集: {len(self.volumes)} 个体积")
    
    def _collect_volumes(self, train_ratio: float):
        """收集所有体积"""
        patient_groups = sorted([d for d in self.data_root.iterdir() if d.is_dir()])
        
        n_train = int(len(patient_groups) * train_ratio)
        if self.train:
            groups = patient_groups[:n_train]
        else:
            groups = patient_groups[n_train:]
        
        for group_dir in groups:
            all_dirs = [d.name for d in group_dir.iterdir() if d.is_dir()]
            
            for dir_name in all_dirs:
                if dir_name.endswith('-origin'):
                    continue
                
                mask_dir = group_dir / dir_name
                origin_dir = group_dir / f"{dir_name}-origin"
                
                if origin_dir.exists():
                    self.volumes.append((str(origin_dir), str(mask_dir)))
    
    def _load_volume(self, origin_dir: str, mask_dir: str) -> Tuple[np.ndarray, np.ndarray]:
        """加载体积"""
        origin_dir = Path(origin_dir)
        mask_dir = Path(mask_dir)
        
        slice_files = sorted(origin_dir.glob("slice_*.png"))
        
        images = []
        masks = []
        
        for sf in slice_files:
            img = cv2.imread(str(sf), cv2.IMREAD_GRAYSCALE)
            img = cv2.resize(img, (self.target_size, self.target_size))
            img = img.astype(np.float32) / 255.0
            images.append(img)
            
            mf = mask_dir / sf.name
            if mf.exists():
                mask = cv2.imread(str(mf), cv2.IMREAD_GRAYSCALE)
                mask = cv2.resize(mask, (self.target_size, self.target_size),
                                interpolation=cv2.INTER_NEAREST)
                mask = (mask > 127).astype(np.float32)
            else:
                mask = np.zeros((self.target_size, self.target_size), dtype=np.float32)
            masks.append(mask)
        
        volume = np.stack(images, axis=-1)
        mask_volume = np.stack(masks, axis=-1)
        
        return volume, mask_volume
    
    def __len__(self):
        return len(self.volumes)
    
    def __getitem__(self, idx):
        origin_dir, mask_dir = self.volumes[idx]
        
        volume, mask_volume = self._load_volume(origin_dir, mask_dir)
        
        D = volume.shape[-1]
        if D > self.num_slices:
            if self.train:
                start = random.randint(0, D - self.num_slices)
            else:
                start = (D - self.num_slices) // 2
            volume = volume[:, :, start:start + self.num_slices]
            mask_volume = mask_volume[:, :, start:start + self.num_slices]
        elif D < self.num_slices:
            pad_size = self.num_slices - D
            volume = np.pad(volume, ((0, 0), (0, 0), (0, pad_size)), mode='constant')
            mask_volume = np.pad(mask_volume, ((0, 0), (0, 0), (0, pad_size)), mode='constant')
        
        volume = torch.from_numpy(volume).float()
        mask_volume = torch.from_numpy(mask_volume).float()
        
        original_size = torch.tensor(volume.shape)
        
        return volume, mask_volume, original_size, original_size


def get_thymoma_dataset_2d(args, sam_trans=None):
    """获取 2D 训练/验证数据集"""
    data_root = args.get('dataset_path', '/map-vepfs/haozhe/lj/report/sam2/data/png_output')
    
    # SAM 需要 1024x1024 输入，Idim 只用于 ModelEmb
    sam_input_size = args.get('sam_input_size', 1024)
    
    train_dataset = ThymomaDataset2D(
        data_root=data_root,
        sam_transform=sam_trans,
        train=True,
        target_size=int(sam_input_size),
    )
    
    test_dataset = ThymomaDataset2D(
        data_root=data_root,
        sam_transform=sam_trans,
        train=False,
        target_size=int(sam_input_size),
    )
    
    return train_dataset, test_dataset


def get_thymoma_dataset_3d(args, sam_trans=None):
    """获取 3D 训练/验证数据集"""
    data_root = args.get('dataset_path', '/map-vepfs/haozhe/lj/report/sam2/data/png_output')
    
    train_dataset = ThymomaDataset3D(
        data_root=data_root,
        num_slices=int(args.get('NumSliceDim', 32)),
        train=True,
        target_size=int(args.get('Idim', 256)),
    )
    
    test_dataset = ThymomaDataset3D(
        data_root=data_root,
        num_slices=int(args.get('NumSliceDim', 32)),
        train=False,
        target_size=int(args.get('Idim', 256)),
    )
    
    return train_dataset, test_dataset


if __name__ == "__main__":
    # 测试数据集
    args = {
        'dataset_path': './data/png_output',
        'Idim': 1024,
    }
    
    train_ds, test_ds = get_thymoma_dataset_2d(args)
    print(f"训练集: {len(train_ds)} 样本")
    print(f"测试集: {len(test_ds)} 样本")
    
    if len(train_ds) > 0:
        img, mask, orig_sz, img_sz = train_ds[0]
        print(f"图像形状: {img.shape}")
        print(f"Mask形状: {mask.shape}")
        print(f"Mask 非零像素: {mask.sum().item()}")
