#!/usr/bin/env bash
#SBATCH --job-name=shakespeare-unigram
#SBATCH --partition=gpu-h200
#SBATCH --qos=normal
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --gres=gpu:2
#SBATCH --cpus-per-task=16
#SBATCH --mem=128G
#SBATCH --time=02:00:00
#SBATCH --signal=USR1@600
#SBATCH --no-requeue
#SBATCH --output=outputs/unigram-%j.out
set -euo pipefail
PILOT_DIR="$SLURM_SUBMIT_DIR"
PROJECT_DIR="$(cd "$PILOT_DIR/../.." && pwd)"
export HF_HOME="$PROJECT_DIR/.cache/huggingface"
export TOKENIZERS_PARALLELISM=false PYTHONUNBUFFERED=1 OMP_NUM_THREADS=4
export TORCH_NCCL_ASYNC_ERROR_HANDLING=1 WANDB_MODE=online
"$PROJECT_DIR/.venv-h200/bin/python" -m torch.distributed.run --standalone \
  --nnodes=1 --nproc-per-node=2 --max-restarts=0 ddp_passage_rl.py \
  --config unigram-config.json --output "outputs/unigram-$SLURM_JOB_ID"
