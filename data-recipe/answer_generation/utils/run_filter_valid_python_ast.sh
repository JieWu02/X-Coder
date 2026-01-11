#!/bin/bash
set -euo pipefail

# Adjust the variables below.

INPUT_FILE="../output/answers.jsonl"
OUTPUT_FILE="../output/answers_valid.jsonl"
MIN_TOKENS=0
MAX_TOKENS=0
FIELD_NAME=""

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"

ARGS=(
  "$INPUT_FILE"
  --output "$OUTPUT_FILE"
)

if [[ -n "$FIELD_NAME" ]]; then
  ARGS+=(--field "$FIELD_NAME")
fi

if [[ "$MIN_TOKENS" -gt 0 ]]; then
  ARGS+=(--min-tokens "$MIN_TOKENS")
fi

if [[ "$MAX_TOKENS" -gt 0 ]]; then
  ARGS+=(--max-tokens "$MAX_TOKENS")
fi

python "$SCRIPT_DIR/filter_valid_python_ast.py" "${ARGS[@]}"
