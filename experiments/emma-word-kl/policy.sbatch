#!/usr/bin/env bash
#SBATCH --job-name=emma-quality-kl
#SBATCH --partition=gpu-h200
#SBATCH --qos=normal
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --gres=gpu:2
#SBATCH --cpus-per-task=16
#SBATCH --mem=128G
#SBATCH --time=01:50:00
#SBATCH --signal=B:USR1@1200
#SBATCH --no-requeue
#SBATCH --output=outputs/budget-%j.out
set -euo pipefail
cd "$SLURM_SUBMIT_DIR"
if [[ -n "${STOCH_PROJECT_DIR:-}" ]]; then
    PROJECT_DIR="$STOCH_PROJECT_DIR"
    if [[ ! -x "$PROJECT_DIR/.venv-h200/bin/python" ]]; then
        echo "STOCH_PROJECT_DIR must contain .venv-h200/bin/python" >&2
        exit 1
    fi
else
    PROJECT_DIR="$SLURM_SUBMIT_DIR"
    while [[ ! -x "$PROJECT_DIR/.venv-h200/bin/python" ]]; do
        if [[ "$PROJECT_DIR" == / ]]; then
            echo "Cannot find shared .venv-h200; set STOCH_PROJECT_DIR" >&2
            exit 1
        fi
        PROJECT_DIR="$(dirname "$PROJECT_DIR")"
    done
fi
export HF_HOME="$PROJECT_DIR/.cache/huggingface"
export HF_HUB_OFFLINE=1 TRANSFORMERS_OFFLINE=1
export TOKENIZERS_PARALLELISM=false PYTHONUNBUFFERED=1 OMP_NUM_THREADS=4 WANDB_MODE=online
export TORCH_NCCL_ASYNC_ERROR_HANDLING=1
exec "$PROJECT_DIR/.venv-h200/bin/python" run.py
