#!/usr/bin/env bash
# Sourced from batch scripts; submit those scripts from the project root.
set -euo pipefail
PROJECT_DIR="${PROJECT_DIR:-${SLURM_SUBMIT_DIR:-$PWD}}"
cd "$PROJECT_DIR"
test -f pyproject.toml || { echo 'Run sbatch from the stochastic-llm project root.' >&2; exit 1; }
VENV_DIR="${VENV_DIR:-$PROJECT_DIR/.venv-h200}"
test -x "$VENV_DIR/bin/python" || { echo 'Run scripts/install_tillicum.sh first.' >&2; exit 1; }
export HF_HOME="${HF_HOME:-$PROJECT_DIR/.cache/huggingface}"
export TOKENIZERS_PARALLELISM=false
export OMP_NUM_THREADS="${SLURM_CPUS_PER_TASK:-8}"
export PYTHONUNBUFFERED=1
export WANDB_BASE_URL="${WANDB_BASE_URL:-https://api.forge.coreweave.com}"
mkdir -p "$HF_HOME" outputs logs
PYTHON="$VENV_DIR/bin/python"
"$PYTHON" -c 'import torch; assert torch.cuda.is_available(), "CUDA unavailable"; print(torch.cuda.get_device_name(0))'
