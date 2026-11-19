# Coherent passages with aggregate bigram constraints: evaluator pilot

The outputs are complete passages. Bigram frequencies are measured over a
collection of passages; they are not independently generated word pairs.
The evaluator-audit commands perform **no training**. Separate passage-RL and
DDP entry points, documented below, train LoRA adapters using that scorer.
The audit first asks whether a frozen quality judge can detect obvious
degradation before using it as a reward.

The completed [results archive](../../results/passage-rl-v1/) includes the short
and longer runs. See the [method guide](../../docs/passage-rl-experiment.md) for
the implemented mathematics, measured results, limitations, and reproduction.

Six fixed scene prompts have assistant-written coherent anchors, word-shuffled
variants, repeated fragments, fluent off-prompt answers, and sampled Markov-bigram
text. Six genuine source excerpts have no task prompt and are scored only for
coherence, fluency, and style. The GPU run adds two fresh passages per prompt from
the unadapted Qwen2.5-3B base model, then scores all 48 passages with a separate
frozen Qwen2.5-7B-Instruct model. Numeric-sampler adapters are not used for prose.

The rubric, model revisions, seeds, raw completions, parse failures, and intended
perturbation comparisons are recorded. A condition's identity is hidden from
the judge. Scores are 1–5 for coherence, fluency, task fulfillment, and separately
Shakespearean style. The initial candidate quality reward averages the first
three dimensions and maps them to 0–1; style never enters it. No score is imputed
when JSON is malformed or generation truncates. Human validation and blind
pairwise evaluation by an independent judge remain necessary before treating
this judge as reliable. This pilot also compares anchors with Markov passages
using the same judge in both A/B orders, exposing position sensitivity.

## Run locally without models

From this directory, using the existing project environment:

```bash
../../.venv/bin/python -m unittest -v test_pilot.py
../../.venv/bin/python pilot.py prepare
```

The preparation command downloads only the approximately 1 MB Tiny Shakespeare
text at a pinned char-rnn commit. Word normalization lowercases letters, retains
internal apostrophes, excludes punctuation and cross-paragraph bigrams, retains
speaker-name words, and never joins separate generated passages. The reference uses the
first 80% of eligible paragraphs; the remaining 20% supply source controls. This
is not a play-disjoint split. Corpus artifacts have checksums in data/manifest.json.

Full-support bigram KL and JSD include an UNKNOWN category. Smoothing has fixed
total mass 1, rather than a fixed pseudocount on every vocabulary entry. These
sparse finite-batch estimates are diagnostics, not reward-validation evidence.
Copying-window overlap and repetition are reported alongside distribution fit.

## Tillicum frozen-model evaluation

Use the existing project environment and model cache. Prepare/download model
files on the login node; inference runs only in a scheduled GPU job. No new
dependencies are required beyond the existing production environment.

```bash
mkdir -p outputs
sbatch --account=mathai evaluate.sbatch
```

One H200 is requested for at most 30 minutes; both models load sequentially.
Every inference run requires online W&B logging to the existing stochastic-llm
project. Config and source/data artifacts are uploaded before model inference;
raw passages, judgments, summary, and report are uploaded on completion.

This is a small evaluator diagnostic, not a new Shakespeare RL result. It has
one judge, six prompts, same-family generator/judge, no human labels, and a public
source likely present in pretraining. The intended corruption comparisons are
hypotheses; failures must be reported rather than used to silently tune the judge.

References: [Tiny Shakespeare source](https://github.com/karpathy/char-rnn/tree/master/data/tinyshakespeare),
[G-Eval](https://arxiv.org/abs/2303.16634),
[LLM judge biases](https://arxiv.org/abs/2306.05685),
[judge model](https://huggingface.co/Qwen/Qwen2.5-7B-Instruct).

## Passage RL pilot

`passage_rl.py` adds an exploratory 100-update LoRA run from pinned
Qwen2.5-3B-Instruct, using the frozen 7B judge. This is a new prose policy, not
the numeric-sampling adapter. Eight training scenes, two validation scenes, and
four held-out test scenes are fixed in `passage-prompts.json` and differ from the
six evaluator-audit scenes. Each update samples 16 complete passages across two
training prompts, at temperature 1 with no top-k or nucleus truncation.

The objective is mean quality minus 0.1 times full-support batch bigram KL, plus
a separate 0.002 sequence-KL penalty to the initial instruct policy. The full
83,454 reference word pairs and an UNKNOWN category are used. Fixed total
pseudocount mass 1 is spread over these categories. Every generated passage,
including truncated or empty output, remains represented in the histogram.
Truncated passages get zero positive quality reward. Copying is measured,
rather than silently removed or repaired.

Credit assignment uses an independent leave-one-out baseline for individual
quality and N-scaled leave-one-passage-out differences for the shared KL reward.
There is no within-group GRPO normalization. An exact two-action enumeration
tests the gradient for passages with different lengths. A real tiny Qwen/LoRA
test verifies that only adapter weights change, that the disabled-adapter
reference stays fixed, and that checkpoint export succeeds.

The original JSON judge left task scores null on 22/48 examples. `quality.py`
instead evaluates each applicable dimension separately using the model's
probabilities for ratings 1 through 5. Expected ratings are conditional on those
five tokens; their total original probability is recorded, and very low mass
causes failure. The corrected audit must show at least five of six coherent
anchors outperforming each predefined corruption on its designated dimension.
This technical check does not substitute for human reward validation.

Submission order is corrected audit -> five-update timing benchmark -> fresh
100-update pilot, all connected by Slurm `afterok` dependencies. The pilot
refuses to start if the benchmark projects more than 90 minutes or source/config
hashes differ. The projection includes a 30% overhead allowance plus 120 seconds
for loading; it is still an estimate. Benchmark weights are never reused.

Baseline and final evaluation use the same four held-out prompts, 16 independent
draws per prompt, seed, decoding, and judge. Validation uses separate prompts.
Checkpoints retain adapter, optimizer, scheduler, and random states every 25
updates; an automatic resume interface is not included. Raw actions, judge
probabilities, advantages, update metrics, final adapter, and comparison are
logged to W&B. Source and corpus hashes are retained.

This is one training seed with the same judge for reward and evaluation. It
does not establish human-perceived quality, independence, originality beyond
the copying diagnostic, or superiority to continued SFT or MaD-RL. Sparse KL
values depend on group size; compare baseline and final at matched sample counts.

## Two-H200 longer run

`ddp_passage_rl.py` runs two Torch DDP processes on one node, each with its own
3B actor and frozen 7B judge on its local GPU. The global group remains 16
passages (8 per rank). Rank zero computes the reward from both ranks' passages;
the shared histogram, quality baseline, and leave-one-passage-out advantages
are global. LoRA gradients are averaged by DDP, with microbatch accumulation
inside `no_sync`. The reward, model revisions, and coefficients are unchanged.

The frozen prompt set has 256 training scenarios, 16 validation scenarios, and
16 test scenarios. Roles, settings, and dilemmas are separated across splits;
these are synthetic template combinations, not 256 human-authored scenes.
`prepare_ddp.py` produces the fixed data without model calls. This run starts
fresh from the instruction model with a 1,200-update cosine schedule and 40
warmup updates, rather than continuing the already-decayed 100-update schedule.

The allocation requests two H200s for at most two wall-clock hours: a maximum
of four GPU-hours, with no automatic requeue or additional benchmark job.
Training stops at 1,200 updates or 105 minutes after program launch, reserving
time for a complete checkpoint, matched initial/final test evaluation, and W&B
upload. The final update count may therefore be lower than 1,200. Validation
and complete optimizer/RNG checkpoints are written every 100 updates.

The initial test policy is evaluated with LoRA disabled after training; base
weights stay frozen throughout. Initial and final evaluations have identical
prompts, per-rank seeds, and sampling settings. Test samples do not select a
checkpoint. Checkpoints retain RNG state for both ranks, the prompt RNG,
optimizer/scheduler, adapter, source/config hashes, and a completion marker.
Explicit resume is supported with `--resume CHECKPOINT` and the same output
directory; it requires a separately authorized allocation and never resubmits
itself. No extra GPU jobs are created by the training program.

The two-process CPU test verifies that global rewards and synchronized adapter
updates match a single-process global-batch update. It also tests rank-specific
RNG restoration and the next optimizer update after checkpoint restore.

```bash
../../.venv/bin/python -m unittest -v test_ddp.py
# On Tillicum, from this directory, after upload:
sbatch --account=mathai train-ddp.sbatch
```

This is a longer exploratory experiment, not evidence that the judge is human
validated or that longer training guarantees better text. Its broader prompt
set prevents a strictly compute-only comparison against the earlier pilot.
