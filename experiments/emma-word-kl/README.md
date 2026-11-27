# Emma: corpus screening and quality-plus-word-KL RL

This directory implements corpus screening and the subsequently authorized Emma
GPU experiment. The screening statistics below are **source-text diagnostics,
not trained-model results**. The runtime uses one two-H200 allocation capped at
110 minutes, including judge checks, policy training, checkpointing and evaluation.
Together with cancelled LitBench job 519547's eight minutes, this remains below
the original four-GPU-hour budget. No automatic retries or additional allocations
are permitted by this controller.

## Provenance and preparation

The text is Jane Austen's [Emma](https://www.gutenberg.org/ebooks/158), obtained
from the [NLTK Gutenberg corpus](https://github.com/nltk/nltk_data/blob/5f09e470fe339e1a119dd174ca284129a3c7a5bb/packages/corpora/gutenberg.zip)
at immutable revision `5f09e470fe339e1a119dd174ca284129a3c7a5bb`.
Project Gutenberg's text endpoints returned HTTP 503; the pinned NLTK
distribution supplies the text and its source README. Original archive/text
hashes are recorded in `data/manifest.json`. The extracted text has 55 chapters
and 160,874 words under our existing word tokenizer, excluding chapter headings.

The chapter identifiers include the volume, because chapter numbers restart.
Seed 42 assigns 39 whole chapters to training, eight to validation and eight to
test. Complete paragraphs form non-overlapping continuation examples: an
80–180-word prefix followed by a genuine adjacent 250–450-word continuation.
No response is cropped to fit these constraints; oversized examples are skipped.
Prepared examples total **168 train / 35 validation / 29 test**. All prompts fit
the pinned Qwen2.5-3B-Instruct tokenizer's 512-token budget.

The proposed training PMF uses only the **168 source continuations corresponding
to the exact training-prompt population**, excluding all prefixes and held-out
chapters. It retains words occurring at least five times, up to 10,000 words,
and pools the remaining probability into OTHER. For this preparation there are
**1,053 retained words plus OTHER**, which holds **10.47%** of source word mass.
This small support partly reflects our restricted paragraph/length selection;
it is not a vocabulary-size claim about the complete novel.

## What the CPU screen measured

`screening-results.json` contains three distinct diagnostics, all using
target-to-empirical KL with total smoothing mass 1.0 and 1,000 resampled batches.
The reported 5th–95th percentiles describe the sampled batches, not confidence
intervals for a population or independent experiments.

1. **Actual chapter split:** each corpus passage has exactly 400 words; 16
   passages make a 6,400-word batch. The reference is the training chapters.
   Held-out chapters still differ: median random-batch KL is 0.654 for validation
   and 0.673 for test. This target differs from the continuation-only training PMF.
2. **Comparable Emma vs Tiny Shakespeare screen:** both corpora are divided
   into consecutive 4,000-word sections; seed 42 assigns 70% of sections to
   references and the rest to observations. Samples never cross a section.
   All batches again contain sixteen 400-word passages. The same 1,000-word
   vocabulary is chosen from the pooled *training sections only*, with OTHER.
3. **Training-objective human baseline:** the actual continuation-only target
   is compared with batches of sixteen real held-out continuations. Median KL
   is 0.528 on validation and 0.549 on test, at median lengths of 4,677 and
   4,870 words. These variable-length results must not be compared directly with
   the equal-6,400-word section diagnostic.

The comparable section results are:

| Source | Shared-vocabulary median KL: random 16 | Shared-vocabulary median KL: four sections × four passages | Own capped vocabulary: random 16 |
|---|---:|---:|---:|
| Emma | 0.266 | 0.275 | 0.638 |
| Tiny Shakespeare | 0.479 | 0.544 | 1.002 |

Thus Emma has lower source-to-reference discrepancy **under this particular
screening protocol**. The result supports selecting it for a controlled trial;
it does not prove a universal author-level property or predict successful RL.
Shared-support scoring hides differences inside OTHER; the separate full capped
support diagnostic exposes more of the tail. Function-word similarity, text
format, selection effects and finite sample size all contribute.

## GPU experiment

The objective remains quality reward minus **0.1 × batch word-PMF KL**, with
**0.002 × base-policy sequence KL**. The fresh pinned Qwen2.5-3B-Instruct actor
uses LoRA rank 16 / alpha 32, AdamW learning rate 5e-6 and up to 1,024 new tokens.
Global batches have four distinct prompts and four samples per prompt, pooled
across two DDP ranks. REINFORCE uses leave-one-out quality credit and the existing
coupled word-PMF credit. Base-policy KL uses same-prompt leave-one-out centering.
The hard update cap is 80, with at most 80 minutes for policy setup/updates and
a global update cutoff reserving 20 minutes for export/evaluation. It is a
time-capped trial; finishing 80 updates is not guaranteed. GRPO is not substituted.

Emma has no preference rankings. **No Bradley–Terry preference model is trained.**
The reward is a frozen Qwen2.5-7B-Instruct model, pinned in `run-config.json`,
scoring coherence, readability and continuation consistency in three separate
forward judgments. Each dimension uses the expected value conditional on rating
tokens 1–5, rejecting low rating-token probability mass. Quality is the mean
rating minus one, divided by four; an unterminated output receives a separate
0.1 penalty. These are model judgments, not calibrated human-preference probabilities.

At the beginning of the same allocation, eight training-only source continuations
from distinct chapters are compared with word-shuffled and repetitive corruptions.
The judge must prefer source coherence in at least 7/8 comparisons for each
corruption, source readability in at least 7/8 shuffled comparisons, and achieve
mean normalized quality margin >=0.1 over repetition. Failure stops the allocation
before RL. These easy synthetic checks do not establish nuanced literary quality
or robust prompt consistency; the rubric remains a limitation.

Model/revision, rubric code, calibration data and normalization are bound by a
hashed certificate. Both stages require online W&B in the configured existing
`stochastic-llm` project. W&B artifacts include exact source/data manifests,
calibration scores, individual quality/distribution/base-KL credit scales,
rollouts, checkpoints and the final adapter. There is no quality-only fallback.

Baseline and final evaluation use the same 16 held-out prompts (covering all
seven test chapters with eligible examples), two samples per prompt, decoding
settings and seeds. Validation uses eight prompts covering its eight chapters.
**Quality evaluation uses the training judge, not an independent scorer.**
There is no additional same-model judge described as independent evidence.
Copying is measured by generated 12-word-window overlap with the entire novel.
Classic text may already be memorized by the base model: chapter holdout does
not eliminate pretraining contamination.
These held-out source statistics have been inspected during dataset selection;
they are not a pristine external benchmark.

The separate LitBench logging bug was fixed by converting NumPy comparison
results to native booleans before JSON/W&B logging. Its regression test now
serializes the actual validation report. The cancelled LitBench job is not
restarted by any script here.

## Length-controlled REINFORCE experiment

`reinforce-length-4gpu-config.json` prepares the same fresh 500-step, four-rank
experiment as job 524624 with one reward change: subtract a deterministic word
range penalty **before** computing the prompt-LOO quality advantage:

```math
\widetilde r_i^Q=r_i^Q-0.2\,[\max(0,250-L_i)+\max(0,L_i-350)]/100.
```

The word tokenizer is identical to histogram evaluation. There is no penalty
within 250--350 words; a 200- or 400-word answer loses 0.1 reward, and a 450-word
answer loses 0.2. Capped answers also retain their separate 0.1 EOS penalty.
Rewards may become negative and are not clipped. Text/counts are never cropped
or discarded. The pooled reference PMF, distribution weight 0.1, reference-policy
KL weight 0.002, model, seed, 500-step schedule and sampling remain unchanged.
Reference responses are not truncated to the new band; 23/168 source references
exceed 350 words, while every policy prompt explicitly requests 250--350 words.
This experiment enforces the prompt band without rebuilding the word target.

`length-penalty-calibration.json` selects the fixed weight from saved **training**
updates 1--50 of job 524624: 800 answers, 200 same-question groups. Among weights
0.025/0.05/0.1/0.2/0.4, choose the largest whose prompt-LOO length-credit RMS
does not exceed original quality-credit RMS. This selects 0.2: length credit RMS
0.09259 versus quality 0.12692 (about 73%). This is a scale heuristic, not an
optimal coefficient or a measurement of gradient variance. Those answers come
from changing early policies, not a fresh fixed initial-policy sample.

A subsequent replay of 32 saved initial-validation answers confirms the reward
arithmetic (mean penalty 0.01581), without changing the selected weight or
generating new answers. No test generation enters coefficient fitting. The
choice to investigate length was motivated by earlier test inspection, so
continued evaluation on the existing test is exploratory, not a pristine new
benchmark. Neither replay establishes that a trained policy will improve.

Raw judge scores, EOS penalties, word-range penalties, compliance rates and
adjusted quality rewards are stored separately for each answer and in W&B.
`judge_quality_mean` remains the raw normalized judge metric;
`quality_reward_mean` is the actual reward after both penalties. The scorer
certificate additionally binds the length/EOS transform's parameters and code,
preventing an old certificate from silently authorizing a new transform.

```bash
.venv/bin/python experiments/emma-word-kl/calibrate_length.py \
  --training-replay results/emma-v1/length-penalty-preparation/training-replay.json \
  --validation-replay results/emma-v1/length-penalty-preparation/validation-0-samples.jsonl \
  --output experiments/emma-word-kl/length-penalty-calibration.json
.venv/bin/python experiments/emma-word-kl/prepare_length.py \
  --output outputs/emma-reinforce-length-4gpu-v1 --stage
```

The new run is prepared with GPU authorization disabled. The proposed ceiling
is four H200s for three hours, including calibration/evaluation (12 GPU-hours),
with the last 30 minutes reserved for final work. Final user approval precedes
submission; no old allocation or automatic retry is reused. Evaluation will
report raw quality, compliance, mean length, count-matched KL and JSD. Improved
length compliance alone will not establish improved distribution matching.

## Prepared longer REINFORCE experiment

### Four-H200 variant selected by the user

The selected variant is `reinforce-long-4gpu-config.json`, generated by
`prepare_long.py --gpus 4`. It targets the same 500 fresh REINFORCE batches,
16 total answers per batch and 8,000 generated training answers. Four GPU ranks
each process one prompt with four answers; the histogram, quality baselines and
reference-KL centering still use the gathered global batch. CPU tests compare
the real four-rank update with the two-rank update on identical examples.
Calibration also partitions all 24 source/corruption examples exactly once
across four ranks, rather than retaining a two-rank stride.

The user selected four H200s after a proposed **three-hour cap (12 GPU-hours)**.
The staged Slurm script uses one node, four GPUs, 32 CPUs and 256 GB host RAM.
Training/setup stops by 2h30m, reserving 30 minutes for final evaluation and
saving. A 500-update target is subject to that time budget. Expected runtime
is approximately 2--3 hours after allocation, extrapolated from two-GPU timings;
four-GPU speedup has not been benchmarked. Queue time is additional.
The two-H200/four-hour draft below is superseded for this requested execution
and will not also be submitted. No extra control allocation or retry is included.

```bash
.venv/bin/python experiments/emma-word-kl/prepare_long.py --gpus 4 \
  --output outputs/emma-reinforce-long-4gpu-v1 --stage
```

The generator keeps authorization disabled; the deployed runtime records the
user's actual approval only after CPU checks and before the single submission.

### Original two-H200 proposal

`reinforce-long-config.json` and `prepare_long.py` define a separate, unsubmitted
500-batch REINFORCE experiment: 8,000 new training answers, one optimizer step
per batch, fresh pinned Qwen2.5-3B-Instruct LoRA, 50 warmup updates and a new
500-step cosine schedule. It keeps the GRPO pilot's same-prompt leave-one-out
quality baseline, pooled word target, weights (0.1 distribution / 0.002 initial
policy KL), sampling, and 1,024-token answer cap. It does not resume the old
80-step schedule, whose learning rate has reached zero, or reuse a GRPO adapter.
The prior pilot does not establish that REINFORCE is superior or that additional
updates will improve the outcome.

The proposed allocation is two H200s for at most four hours (eight GPU-hours).
Training/setup stops by the 3h30m boundary, reserving 30 minutes for evaluation
and saving; an earlier timeout can prevent reaching all 500 updates. Validation
is every 50 updates, checkpoints every 25, with a final checkpoint saved before
diagnostics. Validation and final test use the same prompt sets and four draws
per prompt as the GRPO pilot. Comparison across different training lengths is
not an isolated optimizer ablation. Quality remains the training judge's score.
Count-matched distribution diagnostics and uncertainty remain necessary.

The original REINFORCE run averaged 22.94 seconds per update. At that rate,
500 batches alone take 191 minutes; setup, validation and final evaluation
suggest roughly 3.5--4 hours after allocation start. Queue time and hardware
variation are additional uncertainties. The Slurm request and controller use
the same four-hour ceiling and 30-minute reserve, verified by CPU tests.
No automatic retries or extra control jobs are part of this proposal.

```bash
.venv/bin/python experiments/emma-word-kl/prepare_long.py \
  --output outputs/emma-reinforce-long-v1 --stage
.venv/bin/python -m unittest discover \
  -s outputs/emma-reinforce-long-v1 -p 'test_*.py' -v
```

The generated config has `gpu_submission_authorized: false` and zero approved
GPU resources. CPU staging/upload is allowed; submission awaits the user's
final approval of this separate, longer allocation. Ordinary `run.py` refuses
the unapproved configuration before starting any subprocess. Completed 80-step
run records are unaffected.

## GRPO-style updater details

The shared trainer supports `policy_update: "reinforce"` (the default) and
`policy_update: "grpo"`. The latter is a **custom clipped surrogate for coupled
distribution rewards**, not stock GRPO or an unbiased off-policy batch-divergence
estimator. The historic 80-update result remains REINFORCE. The first GRPO GPU pilot subsequently completed as job 520789: raw test KL
improved only 0.52%, count-matched KL worsened slightly and the quality-change
interval includes zero. See [the completed results](../../results/emma-v1/README.md).

Both comparison arms use four prompts and four answers per prompt, fresh pinned
Qwen2.5-3B-Instruct LoRA, the same frozen fiction judge, target word PMF, reward
weights, and prompt-specific leave-one-out quality baselines. For each global
16-answer batch, the detached task advantage is

```math
A_i=r_i^Q-\frac{1}{G-1}\sum_{j\ne i,\;c_j=c_i}r_j^Q
    +0.1N[D(\mathcal B_{-i})-D(\mathcal B)],\qquad G=4,\;N=16.
```

The distribution term uses the whole batch gathered across both GPU ranks,
removing each answer's full word counts and recomputing the denominator. It is
not a common scalar canceled by group centering. Combined advantages are neither
re-centered nor divided by group standard deviation. The historical whole-batch
quality baseline remains available as `quality_baseline: "global-loo"`; both
comparison arms use `"prompt-loo"` so this change does not favor only GRPO.

Before any update, GRPO snapshots the sampling actor's actual completion-token
log probabilities and initial-policy probabilities, including EOS and excluding
prompts/padding. Its frozen sequence reference credit is

```math
K_i=\log\pi_{old}(y_i|c_i)-\log\pi_0(y_i|c_i)
    -\operatorname{mean}_{j\ne i,c_j=c_i}
       [\log\pi_{old}(y_j|c_j)-\log\pi_0(y_j|c_j)].
```

For each of two passes, the ratio and loss are

```math
\rho_{it}=\exp(\log\pi_\theta(y_{it}|h_{it})-\log\pi_{old}(y_{it}|h_{it})),
L=\frac{1}{Ns}\sum_{i,t}
 [-\min(\rho_{it}A_i,\operatorname{clip}(\rho_{it},0.8,1.2)A_i)
  +0.002\rho_{it}K_i],\qquad s=1.
```

The reference term is separate and unclipped. Its frozen sequence-credit form
preserves the original first-pass REINFORCE gradient instead of substituting
standard GRPO's tokenwise reference-KL estimator. Token losses are summed per
answer, without answer-length division. Clipping changes the objective's
incentive; it does not impose a hard constraint on probability movement.

At the sampling policy all ratios are one, and the first-pass gradient matches
the existing estimator with identical credits. Later passes are approximate:
token ratios do not exactly importance-correct a collective batch reward or
frozen sequence-KL credit under a changed policy. Exact batch-importance weighting
is not claimed. Saved texts, probabilities and credits are reused only inside
their rollout batch, then discarded.

`steps` counts fresh rollout batches. The GRPO pilot has 80 batches / 1,280 new
answers / 160 optimizer steps; REINFORCE has 80 / 1,280 / 80. Warmup and cosine
schedule lengths use optimizer-step units (16/160 versus 8/80). Resume validates
these counts and forbids changing algorithms/scientific settings. This matches
generations, not GPU compute; report GPU elapsed time too. Each arm requests its
own maximum 110 minutes on two H200s, requiring fresh approval. Both ceilings
together are 7.33 GPU-hours; historical-job permission is not reused.

Prepare configs and isolated runtimes on CPU:

```bash
.venv/bin/python experiments/emma-word-kl/prepare_comparison.py \
  --output outputs/emma-optimizer-comparison-v1 --stage
.venv/bin/python -m unittest discover \
  -s experiments/helpsteer-rl-v2 -p 'test_*.py' -v
.venv/bin/python -m unittest discover \
  -s outputs/emma-optimizer-comparison-v1/emma-grpo-v1 -p 'test_*.py' -v
```

Generated configs set `gpu_submission_authorized: false` and zero approved
resources. The GPU entrypoint refuses to launch them. CPU preflight can use
`preflight_gpu(require_authorization=False)`. Both arms require online W&B and
save source manifests, old token probabilities, separate credit scales, per-pass
clipping fractions, old-policy movement and optimizer-step counts. Validation/test
use four draws per prompt (32/64 answers), identically across arms. Model quality
is still scored by the training judge, not independent human evaluation. The
Unicode-reader and checkpoint-before-validation fixes apply to both.

Tests cover exact finite-action collective-reward gradients, nonzero-reference
first-update equivalence, clipping directions, detached snapshots, cross-rank
prompt baselines, one-rank versus two-rank two-pass updates, completion/EOS
masking, and optimizer/scheduler continuation. These are CPU correctness checks,
not policy-performance results.

Prior methods: [GRPO / DeepSeekMath](https://arxiv.org/abs/2402.03300),
[Dr. GRPO normalization analysis](https://arxiv.org/abs/2503.20783).
Our collective credit and frozen reference regularizer differ from those
standard recipes and are labeled explicitly above.

## Reproduce the original preparation on CPU

```bash
.venv/bin/python experiments/emma-word-kl/prepare.py
.venv/bin/python experiments/emma-word-kl/screen.py
.venv/bin/python experiments/emma-word-kl/preflight.py
.venv/bin/python experiments/emma-word-kl/test_emma.py
.venv/bin/python experiments/emma-word-kl/prepare_calibration.py
.venv/bin/python experiments/emma-word-kl/preflight_gpu.py
.venv/bin/python experiments/emma-word-kl/test_runtime.py
```

Preparation downloads a pinned 4.3 MB archive; the policy tokenizer must already
be cached. Shakespeare comparison reuses the existing pinned Tiny Shakespeare
file in `experiments/shakespeare-quality-pilot/data/`.
`stage_runtime.py` builds a fresh runtime with shared helper files copied under
an isolated directory; the deployment manifest covers all uploaded bytes.
Run preflight and CPU tests on the cluster login node before submitting
`policy.sbatch` once. `run.py` refuses a stale calibration certificate and stops
failed stages without resubmitting.

Related references: [original controlled stochasticity blog](https://www.lesswrong.com/posts/upCrcE39aF63GoLJh/training-language-models-for-controlled-stochasticity-2),
[TinyStories](https://arxiv.org/abs/2305.07759),
[E2E](https://arxiv.org/abs/1706.09254), and
[LitBench](https://arxiv.org/abs/2507.00769).
# Length-penalty pilot (November 26, 2026)

The user approved one 100-step proof-of-concept allocation on four H200s,
with a 90-minute wall limit (maximum 6 GPU-hours). The separate
`emma-reinforce-length-pilot-4gpu-v1` profile keeps `steps=500` and
`warmup_steps=50` for the cosine schedule, with `stop_after_steps=100`.
At global batch 16 this generates 1,600 training answers. Full optimizer,
scheduler, adapter, prompt RNG and per-rank RNG state are checkpointed.
No continuation is automatically submitted. Final pilot evaluations use
validation prompts; the test split is reserved for a later decision.
The length coefficient remains 0.2 per 100 words outside 250–350 words;
the word PMF, distribution weight and reference-policy penalty are unchanged.
Training and validation log the raw judge score separately from the length
penalty and adjusted reward. CPU checks and W&B connectivity must pass
before submission. The earlier three-hour full-run draft is not submitted.

The user subsequently reduced this pilot to **two H200s**, keeping the 90-minute
cap (maximum **3 GPU-hours**). The pending four-GPU job 529095 was cancelled
before training. Prepare its isolated replacement with:

```bash
python experiments/emma-word-kl/prepare_length_pilot.py --gpus 2 \
  --output outputs/emma-reinforce-length-pilot-2gpu-v1 --stage
```

The global batch stays 16 (four prompts × four answers); two ranks each process
two prompts. The 100-step stop, 500-step schedule, coefficients and validation
protocol are unchanged. Rank-dependent generation seeds differ, so the two
world sizes are not guaranteed to generate identical passages. Training may
reach its time deadline before 100 steps; final checkpoint/evaluation reserves
remain enforced.

## Longer length-control run submitted

After the completed two-GPU pilot, the user approved the longer four-H200
experiment. Job **530088**, isolated at `emma-reinforce-length-500-4gpu-v1`,
targets 500 updates / 8,000 training answers with a three-hour cap. It uses
`length_profile()` with no pilot stop override and a fresh pinned actor,
matching the prior unpenalized four-rank 500-step schedule. The penalty,
target PMF and KL coefficients stay fixed. All 43 CPU checks passed on Tillicum;
online W&B is required. This run does not resume or overwrite the pilot.
The archived unsubmitted draft is retained; only the new isolated runtime
was submitted. Results remain pending. See
[submission and comparison plan](../../results/emma-v1/length-penalty-long-4gpu-v1/).

## Reward-weight grid implementation (not submitted)

See [GRID_SEARCH.md](GRID_SEARCH.md) for the nine-trial grid, fixed selection
criteria, CPU-only prepare/rank/promote commands, full-state continuation and
resource ceilings. All newly prepared configurations have GPU spending approval
disabled. No grid jobs have been submitted; execution waits for user approval.
The existing completed Emma results are unchanged.
