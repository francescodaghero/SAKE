#!/usr/bin/env bash
set -euo pipefail

DATASET_PATH="data/sake-dataset.json"
N_SHOTS=0
OUTPUT_DIR="outputs"

while [[ $# -gt 0 ]]; do
  case "$1" in
    --dataset_path) DATASET_PATH="$2"; shift 2 ;;
    --n_shots)      N_SHOTS="$2";      shift 2 ;;
    --output_dir)   OUTPUT_DIR="$2";   shift 2 ;;
    *) shift ;;
  esac
done

if [[ -f ".env" ]]; then
  set -a
  source .env
  set +a
fi

python eval/eval_from_openai.py \
  --dataset_path "${DATASET_PATH}" \
  --max_new_tokens 16 \
  --n_shots "${N_SHOTS}" \
  --model openai/gpt-5.4-nano \
  --output_dir "${OUTPUT_DIR}"
