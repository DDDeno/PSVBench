#!/usr/bin/env bash
# PSVBench evaluation example.
#
# Usage:
#   bash run_example.sh <model_config> [extra args...]
#
# Examples:
#   bash run_example.sh eval/configs/models/qwen2.5_vl_7b.yaml
#   bash run_example.sh eval/configs/models/gpt4o.yaml --num-frames 16
#   bash run_example.sh eval/configs/models/random.yaml --limit 50
#   bash run_example.sh eval/configs/models/gpt4o.yaml --use-transcript

set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "$0")" && pwd)"
QA_FILE="${SCRIPT_DIR}/qa/eval.json"
DATA_ROOT="${DATA_ROOT:-${SCRIPT_DIR}}"

MODEL_CONFIG="${1:?Usage: bash run_example.sh <model_config.yaml> [extra args...]}"
shift

python -m eval.run_eval \
    --qa-file "$QA_FILE" \
    --data-root "$DATA_ROOT" \
    --model-config "$MODEL_CONFIG" \
    --seed 0 \
    --resume \
    "$@"
