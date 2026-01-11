#!/bin/bash
set -euo pipefail

# Adjust the variables below to match your API and paths.

API_BASE="https://api.openai.com/v1"
API_KEY="your_api_key"
MODEL_NAME="deepseek-ai/DeepSeek-R1-0528"

START_IDX=0
END_IDX=1000
BATCH_SIZE=64
TEMPLATE_STYLE=""  # codeforces | leetcode | atcoder
RESUME=false

OUTPUT_FILE="../output/questions_${START_IDX}_${END_IDX}.jsonl"
FEATURES_FILE="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)/features_trees_data/feature_all.jsonl"
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")/scripts" && pwd)"

if [[ ! -f "$FEATURES_FILE" ]]; then
  echo "Missing features file: $FEATURES_FILE"
  exit 1
fi

export OPENAI_BASE_URL="$API_BASE"
export OPENAI_API_KEY="$API_KEY"
export OPENAI_MODEL="$MODEL_NAME"

ARGS=(
  --features-file "$FEATURES_FILE"
  --start "$START_IDX"
  --end "$END_IDX"
  --output "$OUTPUT_FILE"
  --batch-size "$BATCH_SIZE"
)

if [[ -n "$TEMPLATE_STYLE" ]]; then
  ARGS+=(--template-style "$TEMPLATE_STYLE")
fi

if [[ "$RESUME" == "true" ]]; then
  ARGS+=(--resume)
fi

python "$SCRIPT_DIR/generate_questions.py" "${ARGS[@]}"
