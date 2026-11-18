#!/usr/bin/env bash
#SBATCH --job-name=stochastic-train
#SBATCH --partition=gpu-h200
#SBATCH --qos=normal
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=8
#SBATCH --mem=64G
#SBATCH --time=04:00:00
#SBATCH --output=logs/%x-%j.out
#SBATCH --signal=B:USR1@120

set -euo pipefail
source "${PROJECT_DIR:-$SLURM_SUBMIT_DIR}/scripts/tillicum_env.sh"
ARGS=(--config "${CONFIG:-configs/sft-h200.yaml}")
if [[ -n "${OUTPUT_DIR:-}" ]]; then ARGS+=(--output-dir "$OUTPUT_DIR"); fi
if [[ -n "${RESUME_FROM:-}" ]]; then ARGS+=(--resume-from-checkpoint "$RESUME_FROM"); fi
if [[ -n "${MODEL_REVISION:-}" ]]; then ARGS+=(--revision "$MODEL_REVISION"); fi
# Trainer saves regular checkpoints; a time-limit signal requests a clean stop
# after the current update and a final checkpoint (handled in Python).
"$PYTHON" -m stochastic_llm train "${ARGS[@]}" &
TRAIN_PID=$!
trap 'kill -USR1 "$TRAIN_PID" 2>/dev/null || true' USR1
while true; do
  if wait "$TRAIN_PID"; then
    exit 0
  else
    STATUS=$?
    # A trapped signal interrupts Bash's wait while Python finishes its update.
    if kill -0 "$TRAIN_PID" 2>/dev/null; then continue; fi
    exit "$STATUS"
  fi
done
