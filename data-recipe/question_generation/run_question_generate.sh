#!/bin/bash
set -euo pipefail

# Adjust the variables below to match your API and paths.

API_PROVIDER="openai"  # openai | aoai

API_BASE="https://api.openai.com/v1"
API_KEY="your_api_key"
MODEL_NAME="deepseek-ai/DeepSeek-R1-0528"

# For AOAI (TRAPI-style). Used only when API_PROVIDER="aoai".
TRAPI_INSTANCE="gcr/shared"
TRAPI_API_VERSION="2024-12-01-preview"
TRAPI_TOKEN_SCOPE="api://trapi/.default"

START_IDX=0
END_IDX=1000
BATCH_SIZE=64
TEMPLATE_STYLE=""  # codeforces | leetcode | atcoder
RESUME=false
MIN_LEAF_COUNT=50  # config is skipped if leaf_count <= this value
MAX_TOKENS=32768

OUTPUT_FILE="../output/questions_${START_IDX}_${END_IDX}.jsonl"
FEATURES_FILE="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)/features_trees_data/feature_config_sampled.jsonl"
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")/scripts" && pwd)"

if [[ ! -f "$FEATURES_FILE" ]]; then
  echo "Missing features file: $FEATURES_FILE"
  exit 1
fi

export OPENAI_BASE_URL="$API_BASE"
export OPENAI_API_KEY="$API_KEY"
export OPENAI_MODEL="$MODEL_NAME"
export XCODER_API_PROVIDER="$API_PROVIDER"

if [[ "$API_PROVIDER" == "aoai" ]]; then
  export TRAPI_INSTANCE="$TRAPI_INSTANCE"
  export TRAPI_API_VERSION="$TRAPI_API_VERSION"
  export AZURE_OPENAI_TOKEN_SCOPE="$TRAPI_TOKEN_SCOPE"
fi

ARGS=(
  --features-file "$FEATURES_FILE"
  --start "$START_IDX"
  --end "$END_IDX"
  --output "$OUTPUT_FILE"
  --batch-size "$BATCH_SIZE"
  --min-leaf-count "$MIN_LEAF_COUNT"
  --api-provider "$API_PROVIDER"
  --max-tokens "$MAX_TOKENS"
)

if [[ -n "$TEMPLATE_STYLE" ]]; then
  ARGS+=(--template-style "$TEMPLATE_STYLE")
fi

if [[ "$RESUME" == "true" ]]; then
  ARGS+=(--resume)
fi

python "$SCRIPT_DIR/generate_questions.py" "${ARGS[@]}"
