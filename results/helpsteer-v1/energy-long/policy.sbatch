#!/usr/bin/env bash
#SBATCH --job-name=stoch-energy-long
#SBATCH --partition=gpu-h200
#SBATCH --qos=normal
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --gres=gpu:2
#SBATCH --cpus-per-task=16
#SBATCH --mem=128G
#SBATCH --time=04:00:00
#SBATCH --signal=B:USR1@1200
#SBATCH --no-requeue
#SBATCH --output=outputs/policy-%j.out
set -euo pipefail
cd "$SLURM_SUBMIT_DIR"
PROJECT_DIR="$(cd ../.. && pwd)"
export HF_HOME="$PROJECT_DIR/.cache/huggingface"
export HF_HUB_OFFLINE=1 TRANSFORMERS_OFFLINE=1
export WANDB_MODE=online WANDB_BASE_URL=https://api.forge.coreweave.com
export TOKENIZERS_PARALLELISM=false PYTHONUNBUFFERED=1 OMP_NUM_THREADS=4
export TORCH_NCCL_ASYNC_ERROR_HANDLING=1
exec "$PROJECT_DIR/.venv-h200/bin/python" run_policy.py
