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

## Reproduce on CPU

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
