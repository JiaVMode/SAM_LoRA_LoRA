#!/bin/bash
# AutoSAM 全量微调训练脚本

# 切换到脚本所在目录
cd "$(dirname "$0")"

# 提示用户激活环境
if [[ -z "$VIRTUAL_ENV" && -z "$CONDA_DEFAULT_ENV" ]]; then
    echo "警告: 未检测到激活的虚拟环境。请确保已安装依赖。"
fi

# 设置 GPU
export CUDA_VISIBLE_DEVICES=${CUDA_VISIBLE_DEVICES:-0}

# 开始训练
# nohup python train2.py --config configs/train.yaml > logs/output.log 2>&1 &
python train.py --config configs/train.yaml
