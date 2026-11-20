#!/usr/bin/env bash
#SBATCH --job-name=helpsteer-bt
#SBATCH --partition=gpu-h200
#SBATCH --qos=normal
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=8
#SBATCH --mem=96G
#SBATCH --time=01:00:00
#SBATCH --signal=USR1@300
#SBATCH --no-requeue
#SBATCH --output=outputs/reward-%j.out
set -euo pipefail
PROJECT_DIR="$(cd "$SLURM_SUBMIT_DIR/../.." && pwd)"
export HF_HOME="$PROJECT_DIR/.cache/huggingface"
export TOKENIZERS_PARALLELISM=false PYTHONUNBUFFERED=1 OMP_NUM_THREADS=8 WANDB_MODE=online
"$PROJECT_DIR/.venv-h200/bin/python" train_reward.py --output "outputs/reward-$SLURM_JOB_ID"
