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
PROJECT_DIR="$(cd "$SLURM_SUBMIT_DIR/../.." && pwd)"
export HF_HOME="$PROJECT_DIR/.cache/huggingface"
export HF_HUB_OFFLINE=1 TRANSFORMERS_OFFLINE=1
export TOKENIZERS_PARALLELISM=false PYTHONUNBUFFERED=1 OMP_NUM_THREADS=4 WANDB_MODE=online
export TORCH_NCCL_ASYNC_ERROR_HANDLING=1
exec "$PROJECT_DIR/.venv-h200/bin/python" run.py
