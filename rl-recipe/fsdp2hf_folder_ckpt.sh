#!/bin/bash

# =============================================================================
# 批量转换 FSDP checkpoint 到 HuggingFace 格式
# =============================================================================
#
# 使用方法:
#   ./fsdp2hf_folder_ckpt.sh <checkpoint_dir> <hf_model_path> <output_base_dir>
#
# 示例:
#   bash fsdp2hf_folder_ckpt.sh ./models Qwen/Qwen2.5-Coder-7B-Instruct ./hf_models
#
# 参数说明:
#   checkpoint_dir  - 包含 global_step_* 目录的 checkpoint 根目录
#   hf_model_path   - 原始 HuggingFace 模型路径 (用于复制 config, tokenizer, generation_config 等)
#   output_base_dir - 输出目录，转换后的 HF 模型将保存在此
#
# 输出结构:
#   hf_models/
#   ├── checkpoint_100/
#   │   ├── config.json
#   │   ├── generation_config.json
#   │   ├── tokenizer.json
#   │   ├── tokenizer_config.json
#   │   ├── model.safetensors
#   │   └── SUCCESS
#   ├── checkpoint_200/
#   │   └── ...
#   └── ...
#
# =============================================================================

# chmod +x fsdp2hf_folder_ckpt.sh

# 检查参数
if [ $# -ne 3 ]; then
    echo "使用方法: $0 <checkpoint_dir> <hf_model_path> <output_base_dir>"
    echo "示例: $0 ./models Qwen/Qwen2.5-Coder-7B-Instruct ./hf_models"
    echo ""
    echo "参数说明:"
    echo "  checkpoint_dir  - 包含 global_step_* 目录的 checkpoint 根目录"
    echo "  hf_model_path   - 原始 HuggingFace 模型路径 (用于复制 config, tokenizer, generation_config 等)"
    echo "  output_base_dir - 输出目录，转换后的 HF 模型将保存在此"
    echo ""
    echo "输出结构:"
    echo "  hf_models/"
    echo "  ├── checkpoint_100/"
    echo "  │   ├── config.json"
    echo "  │   ├── generation_config.json"
    echo "  │   ├── tokenizer.json"
    echo "  │   ├── model.safetensors"
    echo "  │   └── SUCCESS"
    echo "  ├── checkpoint_200/"
    echo "  │   └── ..."
    echo "  └── ..."
    exit 1
fi

CHECKPOINT_DIR="$1"
HF_MODEL_PATH="$2"
OUTPUT_BASE_DIR="$3"

# 获取脚本所在目录
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"

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
    # 将 global_step_XXX 转换为 checkpoint_XXX
    step_num=$(echo "$checkpoint_name" | sed 's/global_step_//')
    output_name="checkpoint_${step_num}"
    output_dir="$OUTPUT_BASE_DIR/${output_name}"

    echo ""
    echo "----------------------------------------"
    echo "处理 checkpoint: $checkpoint_path"
    echo "输出到: $output_dir"

    # 检查是否已经处理过
    if [ -f "$output_dir/SUCCESS" ]; then
        echo "跳过已处理的 checkpoint: $checkpoint_name"
        continue
    fi

    # 创建输出目录
    mkdir -p "$output_dir"

    # 检查 actor 子目录是否存在
    actor_dir="$checkpoint_path/actor"
    if [ ! -d "$actor_dir" ]; then
        echo "警告: actor 目录不存在: $actor_dir"
        echo "尝试直接使用 checkpoint 目录..."
        actor_dir="$checkpoint_path"
    fi

    # 执行转换
    echo "开始转换模型权重..."
    if python "$SCRIPT_DIR/scripts/model_merger.py" \
        --backend fsdp \
        --hf_model_path "$HF_MODEL_PATH" \
        --local_dir "$actor_dir" \
        --target_dir "$output_dir"; then

        echo "模型权重转换成功"

        # 复制 config, tokenizer, generation_config 等文件
        echo "复制 tokenizer 和配置文件..."

        # 如果 HF_MODEL_PATH 是本地路径
        if [ -d "$HF_MODEL_PATH" ]; then
            # 复制所有非模型权重文件
            for file in "$HF_MODEL_PATH"/*; do
                filename=$(basename "$file")
                # 跳过模型权重文件
                if [[ "$filename" == *.safetensors ]] || \
                   [[ "$filename" == *.bin ]] || \
                   [[ "$filename" == "pytorch_model"* ]] || \
                   [[ "$filename" == "model"*.safetensors ]]; then
                    continue
                fi
                # 如果目标文件不存在，则复制
                if [ ! -e "$output_dir/$filename" ]; then
                    cp -r "$file" "$output_dir/"
                    echo "  复制: $filename"
                fi
            done
        else
            # 如果是 HuggingFace Hub 路径，使用 Python 下载
            echo "从 HuggingFace Hub 下载配置文件..."
            python3 << EOF
from transformers import AutoTokenizer, AutoConfig, GenerationConfig
import os

output_dir = "$output_dir"
hf_model_path = "$HF_MODEL_PATH"

# 加载并保存 tokenizer
print("  下载 tokenizer...")
tokenizer = AutoTokenizer.from_pretrained(hf_model_path, trust_remote_code=True)
tokenizer.save_pretrained(output_dir)

# 加载并保存 config (如果不存在)
config_path = os.path.join(output_dir, "config.json")
if not os.path.exists(config_path):
    print("  下载 config...")
    config = AutoConfig.from_pretrained(hf_model_path, trust_remote_code=True)
    config.save_pretrained(output_dir)

# 加载并保存 generation_config
gen_config_path = os.path.join(output_dir, "generation_config.json")
if not os.path.exists(gen_config_path):
    print("  下载 generation_config...")
    try:
        gen_config = GenerationConfig.from_pretrained(hf_model_path)
        gen_config.save_pretrained(output_dir)
    except Exception as e:
        print(f"  警告: 无法下载 generation_config: {e}")

print("  配置文件复制完成")
EOF
        fi

        echo "✓ 成功转换: $checkpoint_name -> $output_name"
        echo "$(date)" > "$output_dir/SUCCESS"
        ((total_success++))
    else
        echo "✗ 转换失败: $checkpoint_name"
        echo "$(date)" > "$output_dir/FAILED"
        ((total_failed++))
    fi

    ((total_processed++))
done

echo ""
echo "=========================================="
echo "批量转换完成!"
echo "总计处理: $total_processed"
echo "成功: $total_success"
echo "失败: $total_failed"
echo "=========================================="

# 显示结果
if [ $total_failed -gt 0 ]; then
    echo ""
    echo "失败的 checkpoint:"
    find "$OUTPUT_BASE_DIR" -name "FAILED" -exec dirname {} \; | while read -r failed_dir; do
        echo "  - $(basename "$failed_dir")"
    done
fi

echo ""
echo "成功的 checkpoint:"
find "$OUTPUT_BASE_DIR" -name "SUCCESS" -exec dirname {} \; | while read -r success_dir; do
    echo "  - $(basename "$success_dir")"
done
