# SAM-LoRA-Auto: Efficient Medical Image Segmentation / SAM-LoRA-Auto: 高效医学图像分割

[![Python](https://img.shields.io/badge/Python-3.10%2B-blue)](https://www.python.org/)
[![PyTorch](https://img.shields.io/badge/PyTorch-2.0%2B-red)](https://pytorch.org/)
[![License](https://img.shields.io/badge/license-MIT-green)](LICENSE)

本项目基于 **AutoSAM** (Automated Segment Anything Model) 结合 **LoRA** (Low-Rank Adaptation) 微调技术，实现了高效且精准的医学图像分割。该方案专为胸腺肿瘤 CT 扫描分割优化，通过自动生成提示（Dense Embeddings）消除了对手动提示的依赖。

This repository implements **AutoSAM** with **LoRA** fine-tuning for efficient and accurate medical image segmentation, specifically optimized for thymic tumor segmentation from CT scans.

## ✨ Features / 主要特性

- **AutoSAM Architecture**: 结合 SAM (Segment Anything Model) 与轻量级编码器 (**HarDNet**)，自动生成分割提示（Dense Embeddings），无需人工干预。
- **LoRA Fine-Tuning**: 高效微调整个流程（包括 SAM 的图像编码器、掩码解码器及 HarDNet 主干），参数量减少 90% 以上，同时保持高性能。
- **3D Post-Processing**: 包含基于形态学操作和 3D 连通域分析的后处理流程，有效去噪并从 2D 切片重建 3D 结果。
- **Dual Training Modes**: 支持 **LoRA 微调**（推荐）和 **全量微调**。

## 📁 Project Structure / 项目结构

```
SAM_LoRA_Auto/
├── checkpoints/            # 预训练模型权重 (SAM, HarDNet)
├── configs/                # 训练配置文件 (YAML)
├── dataset/                # 数据加载与处理
├── models/                 # 模型定义 (SAM, LoRA, AutoSAM)
├── segment_anything/       # SAM 核心源码
├── train.py                # 训练主程序
├── predict.py              # 推理/预测脚本
├── run_train_lora.sh       # LoRA 微调启动脚本 (推荐)
├── run_train_2d.sh         # 全量微调启动脚本
├── run_predict.sh          # 批量推理脚本
└── requirements.txt        # 项目依赖
```

## 🚀 Quick Start / 快速开始

### 1. Installation / 环境安装

```bash
# Clone the repository
git clone https://github.com/yourusername/SAM_LoRA_Auto.git
cd SAM_LoRA_Auto

# Install dependencies
pip install -r requirements.txt
```

### 2. Data Preparation / 数据准备

训练脚本默认接受 PNG 格式的切片数据。请按下述结构组织您的数据，或修改配置文件中的路径。

```
data/
  ├── png_output/
      ├── case_001/
          ├── image/ (CT slices .png)
          ├── mask/  (Ground truth .png)
```

### 3. Model Preparation / 模型准备

请下载预训练权重并放置在 `checkpoints/` 目录下：

- **SAM 权重**: 下载 `sam_vit_b.pth` (或 `vit_h`, `vit_l`)。
- **HarDNet 权重**: 程序会自动下载，或手动放置。

### 4. Training / 训练

**方式一：LoRA 微调 (推荐)**
显存占用低，训练速度快。

```bash
bash run_train_lora.sh
# 或者
python train.py --config configs/train_lora.yaml
```

**配置 (`configs/train_lora.yaml`):**

```yaml
use_lora: true
lora_rank: 8
batch_size: 16
learning_rate: 1.0e-3
```

**方式二：全量微调**
微调所有参数，需要更多显存。

```bash
bash run_train_2d.sh
# 或者
python train.py --config configs/train.yaml
```

### 5. Inference / 推理

推理脚本会自动检测模型是否通过 LoRA 训练。

**单文件预测:**

```bash
python predict.py -i /path/to/image.nii.gz -o /path/to/output_dir -w ./output/best.pt
```

**批量预测:**

处理整个文件夹中的 NIfTI 文件：

```bash
bash run_predict.sh /input/dir /output/dir ./output/best.pt
```

## 🙏 Acknowledgements / 致谢

本项目参考并感谢以下开源项目的工作：

- **SAM**: [Segment Anything](https://github.com/facebookresearch/segment-anything)
- **AutoSAM**: [AutoSAM: Adapting SAM to Medical Images by Overcoming the Prompt Barrier](https://github.com/talshaharabany/AutoSAM)
