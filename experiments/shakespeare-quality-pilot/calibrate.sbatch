#!/usr/bin/env bash
#SBATCH --job-name=shakespeare-calibrate
#SBATCH --partition=gpu-h200
#SBATCH --qos=normal
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=8
#SBATCH --mem=64G
#SBATCH --time=00:15:00
#SBATCH --output=outputs/calibrate-%j.out
set -euo pipefail
PILOT_DIR="$SLURM_SUBMIT_DIR"
PROJECT_DIR="$(cd "$PILOT_DIR/../.." && pwd)"
export HF_HOME="$PROJECT_DIR/.cache/huggingface"
export TOKENIZERS_PARALLELISM=false PYTHONUNBUFFERED=1 OMP_NUM_THREADS=8
"$PROJECT_DIR/.venv-h200/bin/python" quality.py \
  --input outputs/audit-498097/candidates.jsonl --output "outputs/calibration-$SLURM_JOB_ID"
