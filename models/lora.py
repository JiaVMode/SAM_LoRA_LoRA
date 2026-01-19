"""
LoRA (Low-Rank Adaptation) 模块
用于高效微调大模型，只训练低秩分解的适配器权重
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
import math
from typing import Optional, List


class LoRALinear(nn.Module):
    """
    带 LoRA 适配器的线性层
    
    原始线性层: y = Wx + b
    LoRA 适配后: y = Wx + b + (BAx) * scaling
    其中 A: (in_features, rank), B: (rank, out_features)
    """
    
    def __init__(
        self,
        original_layer: nn.Linear,
        rank: int = 8,
        alpha: float = 16.0,
        dropout: float = 0.0,
    ):
        super().__init__()
        
        self.original_layer = original_layer
        self.rank = rank
        self.alpha = alpha
        self.scaling = alpha / rank
        
        in_features = original_layer.in_features
        out_features = original_layer.out_features
        
        # LoRA 低秩分解矩阵
        self.lora_A = nn.Parameter(torch.zeros(in_features, rank))
        self.lora_B = nn.Parameter(torch.zeros(rank, out_features))
        
        # Dropout
        self.dropout = nn.Dropout(p=dropout) if dropout > 0 else nn.Identity()
        
        # 初始化
        nn.init.kaiming_uniform_(self.lora_A, a=math.sqrt(5))
        nn.init.zeros_(self.lora_B)
        
        # 冻结原始权重
        for param in self.original_layer.parameters():
            param.requires_grad = False
    
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # 原始输出
        original_output = self.original_layer(x)
        
        # LoRA 输出
        lora_output = self.dropout(x) @ self.lora_A @ self.lora_B * self.scaling
        
        return original_output + lora_output
    
    def merge_weights(self):
        """将 LoRA 权重合并到原始层 (用于推理加速)"""
        with torch.no_grad():
            self.original_layer.weight.data += (
                self.lora_B.T @ self.lora_A.T * self.scaling
            )
    
    def unmerge_weights(self):
        """取消合并 (用于继续训练)"""
        with torch.no_grad():
            self.original_layer.weight.data -= (
                self.lora_B.T @ self.lora_A.T * self.scaling
            )


class LoRAConv2d(nn.Module):
    """
    带 LoRA 适配器的 Conv2d 层
    """
    
    def __init__(
        self,
        original_layer: nn.Conv2d,
        rank: int = 8,
        alpha: float = 16.0,
        dropout: float = 0.0,
    ):
        super().__init__()
        
        self.original_layer = original_layer
        self.rank = rank
        self.alpha = alpha
        self.scaling = alpha / rank
        
        in_channels = original_layer.in_channels
        out_channels = original_layer.out_channels
        kernel_size = original_layer.kernel_size
        stride = original_layer.stride
        padding = original_layer.padding
        
        # LoRA: 使用 1x1 卷积进行低秩分解
        self.lora_down = nn.Conv2d(in_channels, rank, kernel_size=1, bias=False)
        self.lora_up = nn.Conv2d(rank, out_channels, kernel_size=1, bias=False)
        
        # Dropout
        self.dropout = nn.Dropout2d(p=dropout) if dropout > 0 else nn.Identity()
        
        # 初始化
        nn.init.kaiming_uniform_(self.lora_down.weight, a=math.sqrt(5))
        nn.init.zeros_(self.lora_up.weight)
        
        # 冻结原始权重
        for param in self.original_layer.parameters():
            param.requires_grad = False
    
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # 原始输出
        original_output = self.original_layer(x)
        
        # LoRA 输出 (只用 1x1 卷积)
        lora_output = self.lora_up(self.lora_down(self.dropout(x))) * self.scaling
        
        # 需要匹配空间尺寸
        if lora_output.shape[-2:] != original_output.shape[-2:]:
            lora_output = F.interpolate(
                lora_output, size=original_output.shape[-2:], mode='bilinear', align_corners=False
            )
        
        return original_output + lora_output


def apply_lora_to_model(
    model: nn.Module,
    rank: int = 8,
    alpha: float = 16.0,
    target_modules: Optional[List[str]] = None,
    apply_to_conv: bool = False,
) -> nn.Module:
    """
    将 LoRA 应用到模型的指定层
    
    Args:
        model: 要修改的模型
        rank: LoRA 秩
        alpha: LoRA 缩放因子
        target_modules: 要应用 LoRA 的模块名称列表 (默认: 所有 Linear)
        apply_to_conv: 是否应用到 Conv2d 层
    
    Returns:
        修改后的模型
    """
    lora_count = 0
    
    for name, module in list(model.named_modules()):
        # 检查是否是目标模块
        if target_modules and not any(t in name for t in target_modules):
            continue
        
        # 获取父模块和属性名
        parts = name.split('.')
        parent = model
        for part in parts[:-1]:
            parent = getattr(parent, part)
        attr = parts[-1] if parts else None
        
        if attr is None:
            continue
        
        # 替换 Linear 层
        if isinstance(module, nn.Linear):
            lora_layer = LoRALinear(module, rank=rank, alpha=alpha)
            setattr(parent, attr, lora_layer)
            lora_count += 1
        
        # 替换 Conv2d 层 (可选)
        elif apply_to_conv and isinstance(module, nn.Conv2d):
            if module.kernel_size[0] > 1:  # 跳过 1x1 卷积
                lora_layer = LoRAConv2d(module, rank=rank, alpha=alpha)
                setattr(parent, attr, lora_layer)
                lora_count += 1
    
    print(f"[LoRA] 应用了 {lora_count} 个 LoRA 适配器 (rank={rank}, alpha={alpha})")
    return model


def get_lora_parameters(model: nn.Module) -> List[nn.Parameter]:
    """获取模型中所有 LoRA 参数"""
    lora_params = []
    for name, param in model.named_parameters():
        if 'lora_' in name:
            lora_params.append(param)
    return lora_params


def save_lora_weights(model: nn.Module, path: str):
    """只保存 LoRA 权重"""
    lora_state_dict = {}
    for name, param in model.named_parameters():
        if 'lora_' in name:
            lora_state_dict[name] = param.data
    torch.save(lora_state_dict, path)
    print(f"[LoRA] 保存了 {len(lora_state_dict)} 个 LoRA 权重到 {path}")


def load_lora_weights(model: nn.Module, path: str):
    """加载 LoRA 权重"""
    lora_state_dict = torch.load(path, map_location='cpu')
    model_state = model.state_dict()
    
    loaded = 0
    for name, param in lora_state_dict.items():
        if name in model_state:
            model_state[name] = param
            loaded += 1
    
    model.load_state_dict(model_state)
    print(f"[LoRA] 加载了 {loaded} 个 LoRA 权重从 {path}")
