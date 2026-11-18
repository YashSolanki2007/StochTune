#!/usr/bin/env bash
# Submit a sequential experiment, using at most one H200 at a time.
set -euo pipefail
if [[ $# != 1 ]]; then
  echo 'Usage: bash scripts/submit_tillicum.sh YOUR_SLURM_ACCOUNT' >&2
  exit 2
fi
cd "$(dirname "${BASH_SOURCE[0]}")/.."
mkdir -p logs
test -f data/numeric-v1/manifest.json || { echo 'Generate the data first.' >&2; exit 1; }
for directory in outputs/qwen25-3b-sft outputs/eval-base outputs/eval-sft; do
  if [[ -e "$directory" ]]; then
    echo "Output already exists: $directory. Choose fresh output paths before resubmitting." >&2
    exit 1
  fi
done
ACCOUNT_NAME="$1"
export HF_HOME="${HF_HOME:-$PWD/.cache/huggingface}"
.venv-h200/bin/python -m stochastic_llm tracking-check --config configs/sft-h200.yaml
MODEL_REVISION="$(.venv-h200/bin/python -c \
  'from huggingface_hub import snapshot_download; from pathlib import Path; from stochastic_llm.io import read_config; c=read_config("configs/sft-h200.yaml"); print(Path(snapshot_download(c["model"], revision=c["revision"], local_files_only=True)).name)')"
export MODEL_REVISION
TRAIN_JOB="$(sbatch --parsable --account="$ACCOUNT_NAME" --export=ALL,CONFIG=configs/sft-h200.yaml slurm/train.sbatch)"
TRAIN_JOB="${TRAIN_JOB%%;*}"
BASE_JOB="$(sbatch --parsable --account="$ACCOUNT_NAME" --dependency="afterok:$TRAIN_JOB" \
  --export=ALL,CONFIG=configs/evaluate.yaml,ADAPTER=,OUTPUT_DIR=outputs/eval-base slurm/evaluate.sbatch)"
BASE_JOB="${BASE_JOB%%;*}"
ADAPTER_JOB="$(sbatch --parsable --account="$ACCOUNT_NAME" --dependency="afterok:$BASE_JOB" \
  --export=ALL,CONFIG=configs/evaluate.yaml,ADAPTER=outputs/qwen25-3b-sft/adapter,OUTPUT_DIR=outputs/eval-sft \
  slurm/evaluate.sbatch)"
ADAPTER_JOB="${ADAPTER_JOB%%;*}"
echo "Submitted train=$TRAIN_JOB, baseline=$BASE_JOB, adapter evaluation=$ADAPTER_JOB"
echo 'After completion: .venv-h200/bin/python -m stochastic_llm compare outputs/eval-base outputs/eval-sft'
