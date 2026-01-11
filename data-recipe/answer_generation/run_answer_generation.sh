#!/bin/bash
set -euo pipefail

# Adjust the variables below to match your hardware and paths.

MODEL_PATH="deepseek-ai/DeepSeek-R1-0528"
HOST="0.0.0.0"
PORT=30001
TP=8
MEM_FRACTION=0.90

INPUT_FILE="../output/questions_0_1000.jsonl"
OUTPUT_BATCH="../output/answers_batch.jsonl"
OUTPUT_CONCURRENT="../output/answers_concurrent.jsonl"

# Start SGLang server in background
python -m sglang.launch_server \
  --model-path "$MODEL_PATH" \
  --host "$HOST" \
  --port "$PORT" \
  --tp "$TP" \
  --mem-fraction-static "$MEM_FRACTION" \
  > /tmp/sglang_answer_server.log 2>&1 &

SERVER_PID=$!
trap 'kill ${SERVER_PID} 2>/dev/null || true' EXIT

# Wait for server to be ready
sleep 10

# Batch API version
python scripts/gen_answer_batched.py \
  --input "$INPUT_FILE" \
  --output "$OUTPUT_BATCH" \
  --model-name "$MODEL_PATH" \
  --use-batch-api \
  --worker-ip "${HOST}" \
  --worker-port "${PORT}"

# Concurrent version
python scripts/gen_answer_concurrent.py \
  --input-file "$INPUT_FILE" \
  --output-file "$OUTPUT_CONCURRENT" \
  --model-name "$MODEL_PATH" \
  --worker-ips "${HOST}" \
  --worker-port "${PORT}" \
  --concurrency 16
