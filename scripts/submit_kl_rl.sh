#!/usr/bin/env bash
# Run only after the user gives final approval for this new GPU experiment.
set -euo pipefail
if [[ $# != 1 ]]; then
  echo 'Usage: bash scripts/submit_kl_rl.sh YOUR_SLURM_ACCOUNT' >&2
  exit 2
fi
cd "$(dirname "${BASH_SOURCE[0]}")/.."
export HF_HOME="${HF_HOME:-$PWD/.cache/huggingface}"
ACCOUNT_NAME="$1"
.venv-h200/bin/python scripts/preflight_kl_rl.py
for directory in outputs/kl-rl-benchmark outputs/qwen25-3b-batch-kl-rl outputs/eval-batch-kl-rl; do
  if [[ -e "$directory" ]]; then
    echo "Output already exists: $directory. Inspect it before submitting another run." >&2
    exit 1
  fi
done
JOURNAL=logs/kl-rl-submitted-jobs.tsv
mkdir -p logs
if [[ -e "$JOURNAL" ]]; then
  echo "Submission journal already exists: $JOURNAL. Inspect those jobs before resubmitting." >&2
  exit 1
fi
printf 'stage\tjob_id\tdepends_on\n' > "$JOURNAL"
BENCH_JOB="$(sbatch --parsable --account="$ACCOUNT_NAME" --qos=debug --time=00:30:00 \
  --job-name=stochastic-kl-bench \
  --export=ALL,CONFIG=configs/batch-kl-rl-h200.yaml,RL_STEPS=20,OUTPUT_DIR=outputs/kl-rl-benchmark \
  slurm/rl.sbatch)"
BENCH_JOB="${BENCH_JOB%%;*}"
printf 'benchmark\t%s\t\n' "$BENCH_JOB" >> "$JOURNAL"
RL_JOB="$(sbatch --parsable --account="$ACCOUNT_NAME" --dependency="afterok:$BENCH_JOB" \
  --export=ALL,CONFIG=configs/batch-kl-rl-h200.yaml,RL_STEPS=,OUTPUT_DIR= slurm/rl.sbatch)"
RL_JOB="${RL_JOB%%;*}"
printf 'batch_kl_rl\t%s\t%s\n' "$RL_JOB" "$BENCH_JOB" >> "$JOURNAL"
EVAL_JOB="$(sbatch --parsable --account="$ACCOUNT_NAME" --dependency="afterok:$RL_JOB" \
  --time=01:00:00 \
  --export=ALL,CONFIG=configs/evaluate.yaml,ADAPTER=outputs/qwen25-3b-batch-kl-rl/adapter,OUTPUT_DIR=outputs/eval-batch-kl-rl \
  slurm/evaluate.sbatch)"
EVAL_JOB="${EVAL_JOB%%;*}"
printf 'evaluation\t%s\t%s\n' "$EVAL_JOB" "$RL_JOB" >> "$JOURNAL"
echo "Submitted benchmark=$BENCH_JOB, KL-reward RL=$RL_JOB, evaluation=$EVAL_JOB"
echo 'Compare after completion: .venv-h200/bin/python -m stochastic_llm compare outputs/eval-base outputs/eval-sft outputs/eval-batch-kl-rl'
