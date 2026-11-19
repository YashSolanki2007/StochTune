#!/usr/bin/env bash
#SBATCH --job-name=shakespeare-rl
#SBATCH --partition=gpu-h200
#SBATCH --qos=normal
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=8
#SBATCH --mem=64G
#SBATCH --time=02:00:00
#SBATCH --signal=USR1@120
#SBATCH --output=outputs/passage-%j.out
set -euo pipefail
PILOT_DIR="$SLURM_SUBMIT_DIR"
PROJECT_DIR="$(cd "$PILOT_DIR/../.." && pwd)"
export HF_HOME="$PROJECT_DIR/.cache/huggingface"
export TOKENIZERS_PARALLELISM=false PYTHONUNBUFFERED=1 OMP_NUM_THREADS=8
ARGS=(--output "outputs/${STAGE:-rl}-$SLURM_JOB_ID" --calibration "$CALIBRATION_DIR")
if [[ "${STAGE:-rl}" == benchmark ]]; then
  ARGS+=(--benchmark)
else
  ARGS+=(--benchmark-timing "$BENCHMARK_TIMING")
fi
"$PROJECT_DIR/.venv-h200/bin/python" passage_rl.py "${ARGS[@]}"
