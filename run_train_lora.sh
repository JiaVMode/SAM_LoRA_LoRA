#!/bin/bash
# AutoSAM LoRA 微调训练脚本
# 使用 LoRA 模式，显存占用更少，训练更快

# 切换到脚本所在目录
cd "$(dirname "$0")"

# 提示用户激活环境
if [[ -z "$VIRTUAL_ENV" && -z "$CONDA_DEFAULT_ENV" ]]; then
    echo "警告: 未检测到激活的虚拟环境。请确保已安装依赖。"
fi

# 设置 GPU (默认 0)
export CUDA_VISIBLE_DEVICES=${CUDA_VISIBLE_DEVICES:-0}

# 开始 LoRA 训练
python train.py --config configs/train_lora.yaml
