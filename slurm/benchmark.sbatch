#!/usr/bin/env bash
#SBATCH --job-name=stochastic-bench
#SBATCH --partition=gpu-h200
#SBATCH --qos=debug
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=8
#SBATCH --mem=64G
#SBATCH --time=00:30:00
#SBATCH --output=logs/%x-%j.out

set -euo pipefail
source "${PROJECT_DIR:-$SLURM_SUBMIT_DIR}/scripts/tillicum_env.sh"
ARGS=(--config "${CONFIG:-configs/sft-h200.yaml}" --steps 100 \
  --output-dir "${OUTPUT_DIR:-outputs/benchmark-${SLURM_JOB_ID}}")
if [[ -n "${MODEL_REVISION:-}" ]]; then ARGS+=(--revision "$MODEL_REVISION"); fi
"$PYTHON" -m stochastic_llm train "${ARGS[@]}"
