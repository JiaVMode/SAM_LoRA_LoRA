#!/bin/bash
# AutoSAM 批量预测脚本
# 用法: bash run_predict.sh <输入目录> <输出目录> [权重路径]

set -e

# 参数
INPUT_DIR="${1:-./data/test_images}"
OUTPUT_DIR="${2:-./output/predictions}"
WEIGHTS="${3:-./output/best.pt}"

# 切换到脚本所在目录
cd "$(dirname "$0")"

# 提示用户激活环境
if [[ -z "$VIRTUAL_ENV" && -z "$CONDA_DEFAULT_ENV" ]]; then
    echo "警告: 未检测到激活的虚拟环境。请确保已安装依赖。"
fi

echo "=========================================="
echo "AutoSAM 批量预测"
echo "=========================================="
echo "输入目录: $INPUT_DIR"
echo "输出目录: $OUTPUT_DIR"
echo "模型权重: $WEIGHTS"
echo ""

# 创建输出目录
mkdir -p "$OUTPUT_DIR"

# 统计
total=0
success=0
failed=0

# 遍历所有 NIfTI 文件
for nii_file in $(find "$INPUT_DIR" -name "*-origin.nii.gz" -type f | sort); do
    total=$((total + 1))
    
    # 获取相对路径作为输出子目录
    rel_path=$(dirname "${nii_file#$INPUT_DIR/}")
    case_name=$(basename "$nii_file" .nii.gz)
    out_subdir="$OUTPUT_DIR/$rel_path/$case_name"
    
    echo "[$total] 处理: $nii_file"
    echo "    输出: $out_subdir"
    
    # 运行预测
    if python predict.py -i "$nii_file" -o "$out_subdir" -w "$WEIGHTS" --no_png 2>&1; then
        success=$((success + 1))
        echo "    ✅ 成功"
    else
        failed=$((failed + 1))
        echo "    ❌ 失败"
    fi
    echo ""
done
