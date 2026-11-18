#!/usr/bin/env bash
#SBATCH --job-name=stochastic-eval
#SBATCH --partition=gpu-h200
#SBATCH --qos=normal
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=8
#SBATCH --mem=64G
#SBATCH --time=04:00:00
#SBATCH --output=logs/%x-%j.out

set -euo pipefail
source "${PROJECT_DIR:-$SLURM_SUBMIT_DIR}/scripts/tillicum_env.sh"
ARGS=(--config "${CONFIG:-configs/evaluate.yaml}")
if [[ -n "${ADAPTER:-}" ]]; then ARGS+=(--adapter "$ADAPTER"); fi
if [[ -n "${OUTPUT_DIR:-}" ]]; then ARGS+=(--output-dir "$OUTPUT_DIR"); fi
if [[ -n "${SAMPLES_PER_PROMPT:-}" ]]; then ARGS+=(--samples-per-prompt "$SAMPLES_PER_PROMPT"); fi
if [[ -n "${MODEL_REVISION:-}" ]]; then ARGS+=(--revision "$MODEL_REVISION"); fi
if [[ "${RESUME:-0}" == 1 ]]; then ARGS+=(--resume); fi
"$PYTHON" -m stochastic_llm evaluate "${ARGS[@]}"
