#!/usr/bin/env bash
#SBATCH --job-name=shakespeare-judge
#SBATCH --partition=gpu-h200
#SBATCH --qos=normal
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=8
#SBATCH --mem=64G
#SBATCH --time=00:30:00
#SBATCH --output=outputs/slurm-%j.out

set -euo pipefail
PILOT_DIR="$SLURM_SUBMIT_DIR"
PROJECT_DIR="$(cd "$PILOT_DIR/../.." && pwd)"
export HF_HOME="${HF_HOME:-$PROJECT_DIR/.cache/huggingface}"
export TOKENIZERS_PARALLELISM=false
export OMP_NUM_THREADS=8
export PYTHONUNBUFFERED=1
cd "$PILOT_DIR"
"$PROJECT_DIR/.venv-h200/bin/python" judge.py --output "outputs/audit-$SLURM_JOB_ID"
