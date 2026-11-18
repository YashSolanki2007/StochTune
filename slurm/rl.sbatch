#!/usr/bin/env bash
#SBATCH --job-name=stochastic-kl-rl
#SBATCH --partition=gpu-h200
#SBATCH --qos=normal
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=8
#SBATCH --mem=64G
#SBATCH --time=02:00:00
#SBATCH --output=logs/%x-%j.out
#SBATCH --signal=B:USR1@120

set -euo pipefail
source "${PROJECT_DIR:-$SLURM_SUBMIT_DIR}/scripts/tillicum_env.sh"
ARGS=(--config "${CONFIG:-configs/batch-kl-rl-h200.yaml}")
if [[ -n "${OUTPUT_DIR:-}" ]]; then ARGS+=(--output-dir "$OUTPUT_DIR"); fi
if [[ -n "${RL_STEPS:-}" ]]; then ARGS+=(--steps "$RL_STEPS"); fi
if [[ -n "${RESUME_FROM:-}" ]]; then ARGS+=(--resume-from-checkpoint "$RESUME_FROM"); fi
"$PYTHON" -m stochastic_llm train-rl "${ARGS[@]}" &
TRAIN_PID=$!
trap 'kill -USR1 "$TRAIN_PID" 2>/dev/null || true' USR1
while true; do
  if wait "$TRAIN_PID"; then
    exit 0
  else
    STATUS=$?
    if kill -0 "$TRAIN_PID" 2>/dev/null; then continue; fi
    exit "$STATUS"
  fi
done
