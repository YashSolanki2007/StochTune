#!/usr/bin/env bash
#SBATCH --job-name=helpsteer-subset
#SBATCH --partition=gpu-h200
#SBATCH --qos=normal
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --gres=gpu:2
#SBATCH --cpus-per-task=16
#SBATCH --mem=128G
#SBATCH --time=01:45:00
#SBATCH --signal=USR1@600
#SBATCH --no-requeue
#SBATCH --output=outputs/subset-%j.out
set -euo pipefail
PROJECT_DIR="$(cd "$SLURM_SUBMIT_DIR/../.." && pwd)"
export HF_HOME="$PROJECT_DIR/.cache/huggingface"
export TOKENIZERS_PARALLELISM=false PYTHONUNBUFFERED=1 OMP_NUM_THREADS=4 WANDB_MODE=online
export TORCH_NCCL_ASYNC_ERROR_HANDLING=1
cp "$PROJECT_DIR/experiments/helpsteer-rl-v1/outputs/reward-ready.json" outputs/reward-ready.json
for ARM in quality-only quality-unigram; do
  "$PROJECT_DIR/.venv-h200/bin/python" -m torch.distributed.run --standalone \
    --nnodes=1 --nproc-per-node=2 --max-restarts=0 train_policy.py \
    --arm "$ARM" --output "outputs/$ARM-$SLURM_JOB_ID"
done
