#!/bin/bash
# X-Coder SFT Training Script
# Requires: pip install ms-swift -U

export TORCH_NCCL_ASYNC_ERROR_HANDLING=1
export NCCL_TIMEOUT=3600
export NCCL_IB_TIMEOUT=60
export NCCL_IB_RETRY_CNT=7

# Multi-node training configuration
# Set these environment variables before running:
# - NODE_RANK: Current node rank (0, 1, 2, ...)
# - MASTER_ADDR: Master node address
# - MASTER_PORT: Master node port

CUDA_VISIBLE_DEVICES=0,1,2,3,4,5,6,7 \
NNODES=8 \
NODE_RANK=${NODE_RANK:-0} \
MASTER_ADDR=${MASTER_ADDR:-localhost} \
MASTER_PORT=${MASTER_PORT:-29500} \
NPROC_PER_NODE=8 \
swift sft \
    --dataset hybrid_376k.jsonl \
    --num_train_epochs 8 \
    --model Qwen/Qwen2.5-Coder-7B-Instruct \
    --train_type full \
    --output_dir X-Coder-7B-SFT \
    --torch_dtype bfloat16 \
    --max_length 32768 \
    --deepspeed zero3_offload \
    --gradient_accumulation_steps 2 \
    --max_grad_norm 1.0 \
    --per_device_train_batch_size 1 \
    --lazy_tokenize true \
    --dataloader_num_workers 4 \
    --weight_decay 0.1 \
    --learning_rate 5e-5 \
    --warmup_ratio 0.03 \
    --save_steps 400 \
    --save_total_limit 40 \
    --logging_steps 20 \
    --lr_scheduler_type cosine \
    --ddp_backend nccl \
    --attn_impl flash_attn \
    --gradient_checkpointing true \
    --save_only_model true
