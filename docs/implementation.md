# Controlled stochastic sampling with Qwen2.5-3B

Train `Qwen/Qwen2.5-3B` to produce one numeric draw from a distribution specified
in its prompt. This project includes a sampled-target supervised fine-tuning
(SFT) baseline, a direct soft-target KL experiment, numeric evaluation, and
single-H200 Slurm jobs for UW Tillicum. The upload destination is
`/gpfs/projects/mathai/permuformer2/yash_data/stoch` on `tillicum.hyak.uw.edu`.
New GPU experiments require the user's final approval. The SFT baseline and the 250-update batch-level KL-reward RL pilot have both
completed. The optional direct soft-target KL trainer is implemented but has not
been run. See the repository README and recorded results for the measurements.

## Batch-level KL-reward RL

This is a separate training stage that implements the original proposed reward:
generate independent answers from the model, estimate their distribution, and
reward the batch for matching the requested target. It is **not** the earlier
sampled-target SFT baseline or the soft-target supervised KL loss.

The default pilot starts from the completed SFT adapter to retain its number-only
formatting. It keeps a frozen copy of that same policy as a stability reference.
Each update selects two prompts, generates 256 independent one-number requests
per prompt, computes a histogram, and applies one on-policy REINFORCE update.
The 250-update pilot uses 128,000 newly generated training responses. No SciPy
sampled target answers or supervised completion labels are used for optimization.
SciPy supplies the mathematical target CDF and bin probabilities only.

For each prompt, the target law defines up to 32 bins, using approximately equal
target-probability intervals. Finite discrete support is kept exact when it fits
in the bin budget. The final category is an invalid answer: malformed text,
truncation, a noninteger draw for an integer law, or a value outside its support.
Invalid draws stay in the model histogram denominator and have zero target mass.
With empirical counts `n_b`, group size `N`, number of categories `B`, and
per-category pseudocount `alpha=0.1`, the batch reward is:

```text
q_hat[b] = (n_b + alpha) / (N + alpha * B)
reward = -sum_b p[b] * log(p[b] / q_hat[b])
```

Zero-target-mass terms are omitted using the usual `0 log 0 = 0` convention.
The pseudocount prevents infinite estimated KL for unseen bins. This is a finite
histogram approximation with finite-sample bias, not exact KL between continuous
densities. Within-bin density and open-ended tail values are not fully determined
by this reward; KS, Wasserstein, validity, and support checks remain necessary.

Because the reward depends on the whole group, credit assignment uses
`R(group) - R(group_without_sample_i)`. The baseline depends only on other
samples, preserving the group score-function gradient. Advantages are multiplied
by `N` to compensate for averaging the sequence losses. There is no within-group
mean/std normalization: assigning the same reward to every draw and normalizing
it like ordinary GRPO would erase the signal. Unit tests enumerate all outcomes
of a toy group to verify the gradient exactly, and test convergence toward a
known categorical target.

The full objective also includes a separate, small stability regularizer:
`beta=0.02` times reverse sequence KL to the frozen initial SFT policy. It is
estimated using model-generated sequences and differentiated with its
score-function gradient. It is logged as `rl/reference_sequence_kl_estimate`;
the requested target-matching KL is logged separately as
`rl/target_histogram_kl`, and the batch reward as `rl/group_reward`.

Only the current LoRA adapter is trainable. Dropout is disabled for both rollout
generation and gradient recomputation to keep the action probabilities consistent.
Completion log-probabilities include EOS and exclude prompt tokens and padding.
Truncated completions include every generated token and are penalized as invalid.
Each rollout batch is used for exactly one optimizer update. The reference copy
is frozen and shares the base model weights, avoiding a second full 3B model.

The pilot uses learning rate `1e-5`, cosine decay, 3% warmup, microbatch 16,
rollout batch 128, and gradient clipping at norm 1. Validation uses 12 disjoint
parameter configurations with 256 samples each, every 25 updates. Fixed validation
seeds are isolated from training RNG streams. Test cases and held-out families
are used only after training. A best validation adapter is retained, but the
planned test evaluation uses the **final** adapter rather than selecting it by
test performance. This is one training seed and not a compute-matched comparison.

```bash
# No GPU allocation: run the numerical and gradient tests locally.
.venv/bin/python -m pytest -q tests/test_kl_reward.py tests/test_rl.py

# After the user's final approval, from the Tillicum project root:
bash scripts/submit_kl_rl.sh mathai
```

The submission helper chains a 20-update RL benchmark, a fresh 250-update RL run
from the original SFT checkpoint, then 100,000 test generations. It uses one H200
at a time, requires W&B online logging, and journals each submitted job ID.
The benchmark is not reused as a production checkpoint. Actual runtime must be
measured: on-policy generation and likelihood recomputation make this stage more
expensive than SFT. Walltime caps are 30 minutes for the benchmark, two hours for
RL, and one hour for evaluation; jobs terminate when work finishes.

Final outputs go to `outputs/qwen25-3b-batch-kl-rl/adapter` and
`outputs/eval-batch-kl-rl`. Raw generated actions, batch rewards, and per-action
advantages are saved as `rollouts-*.jsonl` and included in W&B result artifacts.
Checkpoints include actor weights, optimizer/scheduler, Torch/CUDA RNG states,
the prompt-selection RNG, and a completion marker. Resume with the exact config
and a completed RL checkpoint using `train-rl --resume-from-checkpoint PATH` or
the batch script's `RESUME_FROM` environment variable. SIGUSR1 requests a
checkpoint after the current update and prevents downstream evaluation of an
interrupted run.

Related work: [REINFORCE-style optimization](https://arxiv.org/abs/2402.14740)
and [group-aware rewards](https://aclanthology.org/2025.emnlp-main.1649/).

## Weights & Biases tracking

Production training and evaluation require online W&B logging to
[`yash2007-university-of-washington/stochastic-llm`](https://forge.coreweave.com/wandb/yash2007-university-of-washington/stochastic-llm).
The API endpoint is `https://api.forge.coreweave.com`. The required W&B SDK version
is pinned in the environment. A missing login or conflicting `WANDB_MODE` /
`WANDB_DISABLED` setting causes a failure before model loading; production jobs
cannot silently fall back to offline logging.

Logging covers training loss, validation loss, learning rate, gradient norm,
Trainer throughput/timing, system/GPU metrics collected by W&B, experiment
configuration, dataset hashes, model revision, source/config artifacts, final
adapter, per-prompt sampling metrics, example-output tables, and full raw numeric
sample files as result artifacts. Intermediate optimizer checkpoints remain on
GPFS for resuming; the final adapter and result files are uploaded to W&B.
Local tests use explicitly offline tracking and do not create online training runs.

Credentials are not included in source, YAML configs, datasets, or logs. Verify
authentication and project access without starting a run:

```bash
.venv-h200/bin/python -m stochastic_llm tracking-check
```

The submission helper performs this check before submitting any jobs. It does
not replace the requirement to obtain the user's final approval before the
official run. Each training/evaluation output has a `wandb_run.json` with its run
URL; W&B run identities are retained for checkpoint/evaluation resume.

## Experiment

The default SFT run uses 2,000 distinct distribution configurations and 32
independent target draws per configuration: 64,000 examples. At effective batch
size 32, 2,000 updates process this dataset once. All outputs are canonical
numbers followed by EOS; loss applies to the completion and EOS, not the prompt.
No example is silently truncated. Continuous targets have five decimal places;
integer-valued laws have integer outputs. Negative zero is canonicalized.

Training families: normal, uniform, exponential, beta, gamma, and binomial.
Validation has 160 disjoint configurations from those families. The 100 test
configurations alternate between parameters outside the training ranges,
held-out Weibull/Poisson families, and held-out prompt wording. All splits have
disjoint family/parameter combinations. Test cases are never used in training
or validation. Dataset files have SHA-256 checksums in `manifest.json`.

The optional soft-KL run uses three independently sampled target-trie paths per
training configuration (6,000 examples, about 188 updates). At each visited
prefix, it matches the target distribution over next tokens with forward KL.
The model's normalizer includes the full vocabulary, so invalid output mass is
penalized. Loss sums over prefixes and averages over paths, following the KL
chain rule. This is a distribution-matching objective, not an online RL reward.

**The two runs have different data and compute budgets.** Their defaults are
initial baselines, not a compute-controlled comparison. Increase soft-KL paths
or match update budgets deliberately before drawing conclusions about methods.
Soft KL uses at most 1,001 canonical output points spanning target quantiles
0.001–0.999, with tail mass absorbed into edge bins. SFT samples the full rounded
law. Both are evaluated against the full rounded law; quantization and tail
approximation can therefore limit soft-KL fidelity. No exact-continuous or
cryptographic randomness guarantee is claimed.

## Local verification

Requires Python 3.10 or newer. Python 3.12 is recommended. The dependency versions
are deliberately fixed at the direct-dependency level in `pyproject.toml`.

```bash
python3.12 -m venv .venv
source .venv/bin/activate
python -m pip install -e '.[dev]'
python -m pytest -q
python -m stochastic_llm smoke --output-dir outputs/smoke
```

The smoke command creates a tiny random Qwen2 model and a character tokenizer
locally. It trains both objectives on CPU, restores a checkpoint, reloads the
adapter, generates base/adapter samples, resumes evaluation, and exports a
comparison. It downloads no model weights. Successful execution verifies the
pipeline, not sampling quality. Test artifacts from pytest use temporary paths.

This local path is for testing. The production configuration requests CUDA and
fails explicitly if it is unavailable. A Mac training implementation with
MLX/QLoRA is not included. The CLI also allows `--device mps --precision fp32`
for experimental PyTorch/MPS use, but that path has not been validated.

## Prepare Tillicum

Once access is available, copy or extract this project into your Tillicum project
storage. Use a full H200 in `gpu-h200`, not a MIG slice. The job scripts request
one GPU, eight CPU cores, 64 GB host RAM, and four hours under normal QoS. The
benchmark uses debug QoS with a 30-minute limit. Pass your actual Slurm account
to `sbatch --account=...`; none is embedded in the repository.

On the login node, from the project root:

```bash
# Choose an available Python >=3.10; adjust this path to the cluster environment.
PYTHON_BIN=python3.12 bash scripts/install_tillicum.sh

export HF_HOME="$PWD/.cache/huggingface"
.venv-h200/bin/python -m stochastic_llm generate --config configs/data.yaml
.venv-h200/bin/python -m stochastic_llm prepare-model
mkdir -p logs
```

Environment installation uses the PyTorch 2.8.0 CUDA 12.6 wheel. If the cluster's
driver/software policy calls for another supported build, set `TORCH_INDEX_URL`
when installing. The environment and model cache should live in project storage;
they are excluded from the source bundle. Keep `HF_HOME` the same during model
preparation and scheduled jobs. No GPU computation belongs on the login node.

The prepared model cache contains approximately 6.2 GB of BF16 parameters plus
tokenizer/config files; the Python environment and checkpoints need additional
space. There is no need to quantize the base model on a full H200.

## Benchmark, then run

```bash
sbatch --account=YOUR_ACCOUNT slurm/benchmark.sbatch
```

The benchmark uses the full model and 100 SFT updates. It writes its adapter,
metrics, and `timing.json` under `outputs/benchmark-JOB_ID`. The timing estimate
excludes the first 20 updates and projects the measured update time to 2,000
updates. Loading/setup time and final evaluation are separate. Do not continue
the benchmark as the production run: its learning-rate schedule is shorter.

To submit training and both numeric evaluations in sequence, using at most one
GPU at a time:

```bash
bash scripts/submit_tillicum.sh YOUR_ACCOUNT
```

This resolves a cached model commit and uses it for all three jobs. The sequence
is SFT → base evaluation → adapter evaluation. A failed job blocks its successors.
Inspect jobs with `squeue -u "$USER"` and logs under `logs/`. Submission itself
allocates paid compute when jobs start; the script is provided for later use.

Individual submissions, including the optional direct-KL experiment:

```bash
sbatch --account=YOUR_ACCOUNT slurm/train.sbatch
sbatch --account=YOUR_ACCOUNT --export=ALL,CONFIG=configs/soft-kl-h200.yaml slurm/train.sbatch
sbatch --account=YOUR_ACCOUNT slurm/evaluate.sbatch
sbatch --account=YOUR_ACCOUNT \
  --export=ALL,ADAPTER=outputs/qwen25-3b-sft/adapter,OUTPUT_DIR=outputs/eval-sft \
  slurm/evaluate.sbatch
```

Individual submissions do not create dependencies: wait for an adapter to finish
training before evaluating it. For strict reproducibility, set `MODEL_REVISION`
to the same cached commit for every job. The adapter records its resolved base
revision, and evaluation loads that revision automatically. Comparison rejects
different base revisions or evaluation cases/settings. Fixed seeds do not imply
bit-for-bit reproducibility across GPU kernels or software/hardware changes.

## Checkpoints and interruptions

Production SFT checkpoints are saved every 250 updates, retaining the latest
three. They contain adapter weights, optimizer, scheduler, and RNG state. The
final inference adapter is written to `outputs/qwen25-3b-sft/adapter`.
The training Slurm script forwards its time-limit warning as SIGUSR1. Python
requests a checkpoint and stops after the current update, returning a failed
status so downstream jobs cannot mistake an interrupted run for a complete one.
Very long updates/checkpoints can still exceed the scheduler's warning window.

Resume with the exact same training config and an existing checkpoint:

```bash
sbatch --account=YOUR_ACCOUNT \
  --export=ALL,RESUME_FROM=outputs/qwen25-3b-sft/checkpoint-1250 \
  slurm/train.sbatch
```

Existing output directories are protected from accidental replacement. To make
a fresh run, override `OUTPUT_DIR` or edit a copy of the config. Do not change
the dataset, training settings, or model revision while resuming a run.

## Evaluation and outputs

Each request produces exactly one number with a fresh sampling stream. Requests
share no generated context and are batched only for throughput. Decoding uses
temperature 1, top-p 1, top-k disabled, no repetition penalty, and at most 32 new
tokens. No grammar constraints or rejection sampling repair the answers. A new
`GenerationConfig` prevents checkpoint defaults from silently changing sampling.

A valid response must be one finite number and must terminate with EOS. Extra
text, nonfinite values, and token-limit truncation are invalid. Numeric responses
outside the target support are retained in distribution metrics and separately
flagged; silently dropping them would make calibration look better than it is.

Per-case sample files retain text, parsed value, validity, termination status,
and sample index. `results.json` / `results.csv` report:

- Numeric validity, canonical-format rate, and truncation rate.
- Support violations and unique-value rate.
- A midpoint-quantile approximation to Wasserstein-1 distance.
- Wasserstein normalized by the target's 95th–5th percentile width.
- KS distance using the discrete or rounded target CDF, without invalid continuous-law p-values.
- Sample mean and standard deviation.

`summary.json` groups results by family and test category, reporting validity
alongside median fidelity and the count of prompts with no valid samples.
Fidelity is conditional on valid responses. Null fidelity indicates no valid
samples, degenerate normalization, or flagged metric overflow; it never means
perfect calibration. These are marginal-distribution
checks, not a comprehensive test of independence or general language retention.

Evaluation atomically commits each completed prompt file. Resume skips completed
prompts and reruns only incomplete prompts, without duplicating observations:

```bash
sbatch --account=YOUR_ACCOUNT \
  --export=ALL,ADAPTER=outputs/qwen25-3b-sft/adapter,OUTPUT_DIR=outputs/eval-sft,RESUME=1 \
  slurm/evaluate.sbatch
```

For final 10,000-sample evaluations, choose fresh output directories and set
`SAMPLES_PER_PROMPT=10000` for both base and adapter. Allow more job time if needed.
Evaluate candidate checkpoints on validation cases for model selection; reserve
the test cases for final comparisons. The CLI supports `--cases` to select the
validation configurations instead of the test file.

```bash
.venv-h200/bin/python -m stochastic_llm compare outputs/eval-base outputs/eval-sft \
  --output outputs/comparison.json
```

Training outputs include `run_config.json`, dataset checksums, package versions,
GPU information, the resolved model revision, metrics, checkpoints, and timing.
The soft-KL trainer builds target tries lazily and caches 64; CPU trie construction
is part of practical throughput and should be included when interpreting timing.

## Source bundle

```bash
python scripts/package.py
```

This creates `dist/stochastic-llm-source.tar.gz`, containing source, configs,
Slurm scripts, documentation, and tests. It excludes environments, data, model
weights, generated outputs, and credentials. On the cluster, generate data again
from its fixed seed before running the jobs. No SSH connection or transfer is
attempted by this command.

## References

- [Qwen2.5-3B model card](https://huggingface.co/Qwen/Qwen2.5-3B)
- [Calibration Fine-Tuning paper](https://arxiv.org/abs/2605.11845)
- [Transformers 4.57.1 Trainer](https://huggingface.co/docs/transformers/v4.57.1/en/main_classes/trainer)
- [Tillicum scheduling documentation](https://hyak.uw.edu/docs/systems/tillicum/scheduling-jobs/)
