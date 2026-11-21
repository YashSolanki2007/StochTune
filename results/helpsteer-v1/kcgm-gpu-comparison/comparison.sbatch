#!/usr/bin/env bash
#SBATCH --job-name=stoch-kcgm-rbf-energy
#SBATCH --partition=gpu-h200
#SBATCH --qos=normal
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --gres=gpu:2
#SBATCH --cpus-per-task=16
#SBATCH --mem=128G
#SBATCH --time=02:00:00
#SBATCH --no-requeue
#SBATCH --output=outputs/comparison-%j.out
set -euo pipefail
PROJECT_DIR=/gpfs/projects/mathai/permuformer2/yash_data/stoch
export HF_HOME="$PROJECT_DIR/.cache/huggingface"
export HF_HUB_OFFLINE=1 TRANSFORMERS_OFFLINE=1
export WANDB_MODE=online WANDB_BASE_URL=https://api.forge.coreweave.com
export TOKENIZERS_PARALLELISM=false PYTHONUNBUFFERED=1 OMP_NUM_THREADS=4
export TORCH_NCCL_ASYNC_ERROR_HANDLING=1
for variant in rbf energy; do
  cd "$PROJECT_DIR/experiments/helpsteer-kcgm-$variant-v2"
  echo "Starting $variant at $(date -Is)"
  "$PROJECT_DIR/.venv-h200/bin/python" run_policy.py
  echo "Completed $variant at $(date -Is)"
done
