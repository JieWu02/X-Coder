#!/bin/bash

# 批量转换脚本 - 支持并行处理和详细日志
# 使用方法: ./batch_convert_checkpoints_advanced.sh <checkpoint_dir> <hf_model_path> <output_base_dir> [max_parallel_jobs]

# chmod +x fsdp2hf_folder_ckpt.sh

# bash fsdp2hf_folder_ckpt.sh /mnt/haoling/epicoder2/ckpt/qwen25_7b_sft_200k_ckpt8600_clt_bsz_256_5e7_r8_4nh200_32k_0804/ Qwen/Qwen2.5-Coder-7B-Instruct /mnt/haoling/epicoder2/ckpt/hf/qwen25_7b_sft_200k_ckpt8600_clt_bsz_256_5e7_r8_4nh200_32k_0804/

# 检查参数
if [ $# -ne 3 ]; then
    echo "使用方法: $0 <checkpoint_dir> <hf_model_path> <output_base_dir>"
    echo "示例: $0 /mnt/haoling/epicoder2/ckpt/qwen25_7b_sft_200k_ckpt8600_clt_bsz_256_5e7_r8_8nh200_32k_0730 Qwen/Qwen2.5-7B-Base /mnt/haoling/epicoder2/ckpt/converted"
    exit 1
fi

CHECKPOINT_DIR="$1"
HF_MODEL_PATH="$2"
OUTPUT_BASE_DIR="$3"

# 检查输入目录是否存在
if [ ! -d "$CHECKPOINT_DIR" ]; then
    echo "错误: checkpoint目录不存在: $CHECKPOINT_DIR"
    exit 1
fi

# 创建输出目录
mkdir -p "$OUTPUT_BASE_DIR"

echo "开始批量转换checkpoint..."
echo "输入目录: $CHECKPOINT_DIR"
echo "HF模型路径: $HF_MODEL_PATH"
echo "输出目录: $OUTPUT_BASE_DIR"
echo "----------------------------------------"

# 使用数组存储checkpoint路径，避免子shell问题
checkpoint_paths=()

# 首先尝试查找global_step_*目录
echo "查找global_step_*目录..."
while IFS= read -r -d '' path; do
    # 跳过输出目录
    if [[ "$path" == "$OUTPUT_BASE_DIR"* ]]; then
        continue
    fi
    echo "找到checkpoint目录: $path"
    checkpoint_paths+=("$path")
done < <(find "$CHECKPOINT_DIR" -type d -name "global_step_*" -print0)

# 如果没有找到global_step_*目录，尝试其他模式
if [ ${#checkpoint_paths[@]} -eq 0 ]; then
    echo "未找到global_step_*目录，尝试其他模式..."
    while IFS= read -r -d '' path; do
        # 跳过输出目录
        if [[ "$path" == "$OUTPUT_BASE_DIR"* ]]; then
            continue
        fi
        echo "找到checkpoint目录: $path"
        checkpoint_paths+=("$path")
    done < <(find "$CHECKPOINT_DIR" -type d \( -name "*checkpoint*" -o -name "*ckpt*" \) -print0)
fi

# 如果仍然没有找到，尝试直接查找包含模型文件的目录
if [ ${#checkpoint_paths[@]} -eq 0 ]; then
    echo "尝试查找包含模型文件的目录..."
    while IFS= read -r -d '' path; do
        # 跳过输出目录
        if [[ "$path" == "$OUTPUT_BASE_DIR"* ]]; then
            continue
        fi
        # 检查目录是否包含模型文件
        if [ -f "$path/consolidated.00.pth" ] || [ -f "$path/consolidated.00-of-*.pth" ] || [ -d "$path/consolidated" ]; then
            echo "找到包含模型文件的目录: $path"
            checkpoint_paths+=("$path")
        fi
    done < <(find "$CHECKPOINT_DIR" -type d -print0)
fi

echo "总共找到 ${#checkpoint_paths[@]} 个checkpoint目录"

if [ ${#checkpoint_paths[@]} -eq 0 ]; then
    echo "错误: 没有找到任何checkpoint目录"
    echo "请检查目录结构，或者手动指定checkpoint目录"
    exit 1
fi

# 计数器
total_processed=0
total_success=0
total_failed=0

# 处理每个checkpoint
for checkpoint_path in "${checkpoint_paths[@]}"; do
    # 提取checkpoint名称
    checkpoint_name=$(basename "$checkpoint_path")
    parent_dir=$(basename "$(dirname "$checkpoint_path")")
    
    # 创建输出目录
    output_dir="$OUTPUT_BASE_DIR/${parent_dir}/${checkpoint_name}"
    mkdir -p "$output_dir"
    
    echo "处理checkpoint: $checkpoint_path"
    echo "输出到: $output_dir"
    
    # 检查是否已经处理过
    if [ -f "$output_dir/SUCCESS" ]; then
        echo "跳过已处理的checkpoint: $checkpoint_name"
        continue
    fi
    
# python ./scripts/model_merger.py \
#     --backend fsdp \
#     --hf_model_path Qwen/Qwen2.5-Coder-7B-Instruct \
#     --local_dir /mnt/haoling/epicoder2/ckpt/qwen25_7b_sft_200k_ckpt8600_clt_bsz_256_5e7_r8_4nh200_32k_0804/global_step_10/actor \
#     --target_dir  /mnt/haoling/epicoder2/ckpt/hf/qwen25_7b_sft_200k_ckpt8600_clt_bsz_256_5e7_r8_4nh200_32k_0804/global_step_10

    # 执行转换
    if python ./scripts/model_merger.py \
        --backend fsdp \
        --hf_model_path "$HF_MODEL_PATH" \
        --local_dir "$checkpoint_path/actor" \
        --target_dir "$output_dir"; then
        
        echo "✓ 成功转换: $checkpoint_name"
        echo "$(date)" > "$output_dir/SUCCESS"
        ((total_success++))
    else
        echo "✗ 转换失败: $checkpoint_name"
        echo "$(date)" > "$output_dir/FAILED"
        ((total_failed++))
    fi
    
    ((total_processed++))
    echo "----------------------------------------"
done

echo "批量转换完成!"
echo "总计处理: $total_processed"
echo "成功: $total_success"
echo "失败: $total_failed"

# 显示失败的checkpoint
echo ""
echo "失败的checkpoint:"
find "$OUTPUT_BASE_DIR" -name "FAILED" -exec dirname {} \; | while read -r failed_dir; do
    echo "  - $(basename "$failed_dir")"
done

# 显示成功的checkpoint
echo ""
echo "成功的checkpoint:"
find "$OUTPUT_BASE_DIR" -name "SUCCESS" -exec dirname {} \; | while read -r success_dir; do
    echo "  - $(basename "$success_dir")"
done