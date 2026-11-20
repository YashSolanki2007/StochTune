#!/usr/bin/env bash
#SBATCH --job-name=helpsteer-v2-audit
#SBATCH --partition=gpu-h200
#SBATCH --qos=normal
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=8
#SBATCH --mem=96G
#SBATCH --time=00:10:00
#SBATCH --no-requeue
#SBATCH --output=outputs/audit-%j.out
set -euo pipefail
REWARD_OUTPUT="${1:?Specify the existing reward-model output directory}"
PROJECT_DIR="$(cd "$SLURM_SUBMIT_DIR/../.." && pwd)"
export HF_HOME="$PROJECT_DIR/.cache/huggingface"
export TOKENIZERS_PARALLELISM=false PYTHONUNBUFFERED=1 OMP_NUM_THREADS=8 WANDB_MODE=online
"$PROJECT_DIR/.venv-h200/bin/python" audit_reward.py \
  --reward-output "$REWARD_OUTPUT" --output "outputs/reward-audit-$SLURM_JOB_ID"
