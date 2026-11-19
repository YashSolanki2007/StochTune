# Passage distribution matching with a frozen quality judge

The completed passage experiments extend numeric sampling to full dramatic
monologues. Every action is a complete generated passage. The reward combines
the quality of individual passages with aggregate word-pair frequencies from a
reference corpus. This is on-policy REINFORCE with LoRA, not PPO or GRPO.

The [README](../README.md#passage-rl-mathematics-and-implementation) contains the
implemented equations for the reference PMF, smoothed empirical PMF, frozen
judge score, joint objective, leave-one-out advantages, and DDP averaging.

## Recorded experiments

| Run | Policy | Updates / generated training passages | GPUs / elapsed |
|---|---|---|---|
| Short run, job 498183 | Fresh Qwen2.5-3B-Instruct | 100 / 1,600 | 1 H200 / 17m34s |
| Longer run, job 498361 | Fresh Qwen2.5-3B-Instruct | 1,200 / 19,200 | 2 H200s / 1h41m21s |

The longer run has broader prompt coverage and a longer learning-rate schedule.
It does not continue from the short run's adapter or from the numeric adapter.
This makes the recorded before/after comparison meaningful within each run,
but prevents attributing the difference between runs solely to more compute.

The numeric experiments remain separate and are documented in
[the numeric RL guide](kl-rl-experiment.md) and [SFT guide](sft-experiment.md).

## Reference corpus and tokenization

The source is Tiny Shakespeare's `input.txt` from the char-rnn repository at
commit `370cbcd448eb7daf32f21a6be560b70e0b33c4e3`. The source, selected reference,
held-out text, and checksum manifest are under
[`results/passage-rl-v1/corpus`](../results/passage-rl-v1/corpus/).

Preparation splits the text on blank paragraphs, retains paragraphs with at
least ten normalized words, and takes the first 80% of those paragraphs as the
reference. It is a contiguous split, not a play-disjoint split. The remaining
paragraphs provide source controls for the initial judge audit; the final prose
test consists of new scene prompts, not corpus excerpts.

Words use the regex `[A-Za-z]+(?:['’][A-Za-z]+)*`, are lowercased, and normalize
curly internal apostrophes. Punctuation is discarded. Speaker names remain as
words. Counts can cross sentence and line boundaries inside a paragraph, but
not paragraph boundaries. Each generated passage is counted separately.

The target has 83,454 distinct reference bigrams. An UNKNOWN category receives
unseen generated pairs and has zero target mass. Empty and one-word passages
contribute one UNKNOWN count. Total smoothing mass is 1 across all 83,455
categories. This differs from the numeric experiment's per-category smoothing.

The reward matches this selected word-bigram PMF. It neither reconstructs the
full probability law of Shakespeare's texts nor ensures long-range meaning.

## Quality judge and audit history

The frozen judge is `Qwen/Qwen2.5-7B-Instruct` at revision
`a09a35458c702b33eeacc393d103063234e8bc28`. It evaluates coherence, fluency, and
prompt fulfillment. These are model judgments, not human annotations.

The initial audit used generated JSON scores. It evaluated 48 passages,
including assistant-authored anchors, shuffled words, repetitive fragments,
off-prompt coherent passages, Markov-bigram text, source excerpts, and samples
from the unadapted Qwen2.5-3B base model. Only 26/48 judgments supplied all
applicable scores: 22 left prompt fulfillment null. Those incomplete judgments
were rejected, not imputed. The raw responses and errors are preserved in
[`quality-audit`](../results/passage-rl-v1/quality-audit/).

The corrected scorer asks for each dimension separately and computes the
expected rating conditional on the next token being one of the digits 1–5.
All rating probabilities and their unconditional total mass are retained.
Mass below 0.01 causes failure. A truncated passage receives zero positive
quality reward but its bigrams remain in the batch distribution.

The corrected audit required the coherent anchor to outrank a designated
corruption on at least five of six scenes for each of three perturbations. It
passed all 18 comparisons. This checks obvious degradations; it does not show
that the judge can reliably distinguish mediocre prose from excellent prose.
The original A/B order-reversed comparisons used the same judge and also do
not constitute independent human evaluation.

The full rubric and scoring implementation are in
[`quality.py`](../experiments/shakespeare-quality-pilot/quality.py).

## Longer-run configuration

The actor is `Qwen/Qwen2.5-3B-Instruct` at revision
`aa8e72537993ba99e69dfaafa59ed015b17504d1`. Base weights stay frozen. LoRA uses
rank 16, alpha 32, dropout 0, and attention q/k/v/o projections. Both model bases
use BF16 without quantization. AdamW uses learning rate `5e-6`, weight decay
0.01, 40 warmup updates, cosine decay to update 1,200, and gradient clipping at 1.

The target-bigram coefficient is **lambda = 0.1**. The separate reverse sequence
KL to the starting policy uses **beta = 0.002**. These are heuristic settings,
not optima selected by a sweep. Neither coefficient changes between passage runs.

There are 256 training scenarios, 16 validation scenarios, and 16 test scenarios.
They are fixed synthetic combinations from disjoint role, setting, and dilemma
pools. The short run had 8/2/4 scenarios. The longer prompt file is committed as
[`ddp-prompts.json`](../experiments/shakespeare-quality-pilot/ddp-prompts.json).

At each update, two different training scenarios are chosen. Each has eight
independent passage generations from a fresh context. Sampling uses temperature
1, top-p 1, top-k disabled, no repetition penalty, and at most 256 new tokens.
The prompt asks for 80–120 words; word count is not enforced by constrained
decoding or rejection sampling.

## DDP and credit assignment

Two processes each host an actor and frozen judge on their local H200. Each
process generates and scores eight passages. The 16 passages form one global
histogram. The individual-quality leave-one-out baseline and bigram reward
differences are also global, not computed independently on eight-passage groups.

Quality credit subtracts the other passages' mean quality. Bigram credit is N
times the difference between the group reward and the reward after removing
the entire selected passage. Removing individual overlapping bigrams would
not be the same estimator. Reference-policy credit subtracts beta times the
generated completion's log-probability ratio to the frozen initial policy.

Each rank averages its local eight sequence losses; DDP averages their
gradients, reproducing the global mean over 16 passages. Gradient microbatch
size is four per rank, with `no_sync` around both forward and backward for
intermediate accumulation. Global batch size and learning rate are not doubled.

Only LoRA changes; judge and reference-policy base weights remain fixed.
Dropout is disabled during both generation and log-probability recomputation.
Completion probabilities include EOS when generated, and exclude prompt and
padding tokens. Every fresh rollout batch is used for exactly one optimizer update.

The CPU tests enumerate the variable-length group objective, compare two-rank
and single-process global updates, and restore a checkpoint to verify the next
optimizer update and rank-specific RNG. They do not prove sampling quality.

## Checkpoints, budget, and evaluation

Validation and complete checkpoints are written every 100 updates. Checkpoints
include adapter, optimizer/scheduler, prompt RNG, both ranks' RNG states, source
and configuration hashes, and a completion marker. Explicit resume uses the
same config, source, and output directory; there is no automatic resubmission.

The Slurm allocation requests two H200s for two hours and disables requeue.
Training has a 105-minute cutoff, reserving time for final checkpointing,
evaluation, and artifact upload. The completed run reached 1,200 updates before
the cutoff and used about 3.38 GPU-hours. It had no extra GPU benchmark stage.

Validation uses 16 scenarios with two draws each. Final test evaluation uses
16 separate scenarios with four draws each. Initial and final policies use
matching prompts, per-rank seeds, and decoding. The initial test policy is
evaluated after training with LoRA disabled. Test results do not select a
checkpoint. The final adapter is evaluated, not a validation-selected adapter.

All test generations, validation generations, metrics, and update history are
committed under [`ddp`](../results/passage-rl-v1/ddp/). Full 19,200 training
rollouts, final weights, and original source artifacts remain on W&B and GPFS.
The committed `source-manifest.json` describes files as captured at run time;
later documentation and analysis changes do not rewrite that provenance.

## Read and reproduce results without GPU work

From the repository root:

```bash
python3.12 -m venv .venv
source .venv/bin/activate
python -m pip install -e '.[dev,analysis]'
python experiments/shakespeare-quality-pilot/analyze_recorded.py
python -m unittest discover -s experiments/shakespeare-quality-pilot -p 'test_*.py' -v
```

The analysis recomputes aggregate metrics from the committed samples, checks
them against recorded scores, reproduces the count-matched diagnostic and
prompt bootstrap, and writes a report and figure under `outputs/passage-analysis`.
It never loads a model, calls a judge, or allocates a GPU.

For future training on Tillicum, install the existing production environment
with `scripts/install_tillicum.sh`, prepare the pinned model caches, and restore
the recorded reference data and judge check:

```bash
mkdir -p experiments/shakespeare-quality-pilot/data
cp results/passage-rl-v1/corpus/*.txt experiments/shakespeare-quality-pilot/data/
cp results/passage-rl-v1/corpus/manifest.json experiments/shakespeare-quality-pilot/data/
mkdir -p experiments/shakespeare-quality-pilot/outputs/calibration-498176
cp results/passage-rl-v1/quality-calibration/checks.json \
  experiments/shakespeare-quality-pilot/outputs/calibration-498176/
cd experiments/shakespeare-quality-pilot
sbatch --account=YOUR_ACCOUNT train-ddp.sbatch
```

The last command allocates two GPUs; the analysis and tests above do not. Model
caches must contain the exact revisions in `ddp-config.json`. Reusing the archived
calibration reproduces this fixed setup; changing the judge or rubric requires
a new evaluation of that scorer. Scheduler wait time and GPU kernel differences
prevent a guarantee of identical runtime or bitwise-identical samples.

## Results and limitations

The short run did not improve held-out bigram KL or the component quality
scores. Its aggregate quality reward increased only because one formerly
truncated response completed. The longer run reduced raw bigram KL by 1.77%
and increased the frozen judge's aggregate quality from 0.7419 to 0.8029.
Prompt fulfillment improved more than fluency or coherence.

Longer outputs change finite-sample KL. The post hoc diagnostic subsamples
6,136 pooled bigrams without replacement, 200 times; a small improvement
persists. This discards passage structure and does not eliminate estimation
bias. An exploratory paired bootstrap over prompt-mean judge rewards reflects
only prompt variation within this one training seed and judge.

The same judge supplies reward and evaluation. Its corruption tests are not
human validation, and examples can still be awkward despite high scores. Zero
12-word corpus-window overlap does not establish absence of memorization.
No equal-compute continued-SFT or MaD-RL comparison, lambda sweep, independent
human study, or multiple-training-seed experiment has been completed.

## Related work

- [Controlled stochasticity blog](https://www.lesswrong.com/posts/upCrcE39aF63GoLJh/training-language-models-for-controlled-stochasticity-2)
  and [Calibration Fine-Tuning paper](https://arxiv.org/abs/2605.11845): supervised
  distribution matching and stochastic transfer.
- [Forcing Diffuse Distributions](https://arxiv.org/abs/2404.10859): sequence-level
  target-weighted likelihood and transfer across output categories.
- [Generative Distributional Control](https://arxiv.org/abs/2012.11635): constraints
  on individual outputs and their aggregate distribution.
- [GAPO](https://aclanthology.org/2025.emnlp-main.1649/): group-aware frequency rewards.
- [MaD-RL](https://arxiv.org/abs/2609.31644): distribution-matching rewards combined
  with task rewards, including several divergences.

These are relevant precedents. This repository does not claim that combining
quality and distribution rewards, REINFORCE, or a leave-one-out baseline is novel.
