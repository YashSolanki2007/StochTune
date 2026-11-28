# StochTune

Investigating distributional fidelity in language-model generation: numeric draws from requested probability laws, and coherent passages whose aggregate word or word-bigram statistics follow a reference corpus.

This repository records numeric SFT and batch-KL RL, Shakespeare passage RL, and HelpSteer word/semantic-distribution experiments. The latest **1,000-step Emma run (job 535088, word-KL weight 0.5) completed 16,000 training answers in 4h10m50s on four H200s**. On 64 final test answers, raw word KL improves **0.5112 → 0.4568 (10.65%)**, JSD **0.06643 → 0.06159**, and training-judge quality **0.7826 → 0.8444**. Length compliance improves over base (39.1% → 51.6%) but is below the preceding 500-step length-controlled run (60.9%). A secondary count-matched KL diagnostic also improves (4.20%). These are exploratory single-seed results with a training-model judge and reused test prompts; changing both KL weight and training horizon prevents isolating either effect. No grid jobs have been submitted. The optional direct soft-target KL trainer remains implemented but unrun.

See [References and acknowledgments](#references-and-acknowledgments) for the papers, original blog, datasets, models, and implementations informing this work.

## What has actually been run

| Stage | Starting policy and signal | Completed budget | Outcome |
|---|---|---|---|
| Numeric SFT | Qwen2.5-3B base; cross-entropy on target-distribution samples | 2,000 updates; 64,000 examples; one H200 | Large improvement in numeric formatting and sampling metrics |
| Numeric RL | Numeric SFT adapter; negative empirical histogram KL | 250 updates; 128,000 generated responses; one H200 | Mean held-out KL 2.2% lower; mixed transfer |
| Passage judge audit | Frozen 7B instruction model; controlled text perturbations | 48 passages; corrected dimension-wise scoring | Initial JSON protocol missed 22 task scores; corrected scorer passed 18 predefined perturbation checks |
| Short passage RL | Fresh Qwen2.5-3B-Instruct; quality + bigram matching | 100 updates; 1,600 passages; one H200 | No clear held-out improvement |
| Longer passage RL | Fresh Qwen2.5-3B-Instruct; same reward, broader prompts, DDP | 1,200 updates; 19,200 passages; two H200s | Higher judge scores; small reduction in empirical bigram KL |
| Unigram passage RL | Same starting model, judge, prompts, and coefficients; single-word KL reward | 1,097 updates; 17,552 passages; two H200s | Better measured word-frequency matching; quality improved over base, slightly below bigram run |
| HelpSteer subset RL | Human-preference reward model; quality-only and unigram KL with λ = 0.1 / 0.5 | 200 updates per arm; 32 final responses per policy | No convincing joint quality/distribution gain |
| Corrected broad HelpSteer RL | Audited frozen Skywork 8B reward; 4,948-answer word PMF; λ = 0.5 | 86 updates; 1,376 training responses; two H200s | Quality reward increased; unigram KL worsened |
| Programming unigram RL | Programming-only references; frequent words + OTHER; same frozen reward | 75 updates; 1,200 training responses; two H200s | Target KL improved by only 0.7%; quality point estimates declined slightly |
| Conditional semantic MMD | Four questions × four answers; two human-rated reference prototypes per exact question | Completed job 504124; two H200s, 81m36s | MMD 0.3922 → 0.3833; quality 0.6696 → 0.6631; uncertain changes |
| Calibrated Gaussian / energy | Fresh actors; quality + calibrated discrepancy + centered base-policy KL | Job 508718: 50 / 53 updates; two H200s, 1h43m52s | Energy improved semantic point estimates; Gaussian worsened; uncertain changes |
| Longer energy continuation | Restore energy checkpoint 53 and retain the original schedule | Job 509031: 200 cumulative updates; 147 new updates; two H200s, 2h00m03s | Quality point estimates improved; most semantic-matching gain disappeared |
| Distribution-only fitting diagnostic | Fresh actor; four fixed training questions × 16 answers; energy + base-policy KL, quality weight zero | Job 510187: 17 updates, 1,088 answers; two H200s, 1h18m37s | Seen energy distance 3.8% lower; unseen 1.5% higher; weak fitting evidence |
| LitBench fantasy attempt | New literary Bradley–Terry scorer, followed by quality + word KL if validated | Job 519547 cancelled after 8 minutes on two H200s | Validation JSON serialization failed; no RL updates or policy results; regression fixed |
| Emma source screening | Pinned single novel; equal-word source batches; chapter/section holdouts | CPU only; 1,000 resampled batches per condition | Shared-vocabulary source KL 0.266 vs Tiny Shakespeare 0.479; not model results |
| Emma quality + unigram RL | Fresh Qwen2.5-3B-Instruct; frozen 7B fiction judge; word KL λ = 0.1 | Job 519712: 80 updates / 1,280 answers; two H200s, 34m09s | Validation matching improves modestly; postprocessing failure fixed on CPU; final test unavailable |
| Emma custom GRPO | Fresh Qwen2.5-3B-Instruct; prompt-LOO quality and coupled word-KL credits; clipped two-pass updates | Job 520789: 80 batches / 1,280 answers / 160 optimizer steps; two H200s, 39m39s | Judge quality 0.8011 → 0.8137; raw KL 0.4882 → 0.4857, but count-matched KL and JSD worsen |
| Emma longer REINFORCE | Fresh actor; prompt-LOO quality, word KL λ = 0.1, base-policy KL β = 0.002 | Job 524624: 500 updates / 8,000 answers; four H200s, 2h17m20s | Judge quality improves; answers become longer; equal-word-budget KL worsens |
| Emma length-penalty pilot | Same REINFORCE objective plus fixed 250–350-word penalty | Job 529144: completed 100 steps / 1,600 answers; two H200s, 42m34s | Validation compliance 43.8% → 59.4%; raw judge quality unchanged; equal-word KL 2.55% lower |
| Emma longer length-control RL | Fresh actor; same 500-step four-GPU setup as job 524624, plus the fixed length penalty | Job 530088: completed 500 updates / 8,000 answers; four H200s, 1h59m14s | Better length compliance than the unpenalized run, lower judge quality; equal-word KL remains worse than base |
| Emma stronger word-KL RL | Fresh actor; word-KL weight 0.5, length weight 0.2, 1,000-step schedule | Job 535088: completed 1,000 updates / 16,000 answers; four H200s, 4h10m50s | Raw KL 10.65% lower, JSD lower and judge quality higher than base; length compliance below the preceding length-controlled run |
| Reward-weight grid search | Nine combinations of word-KL and length weights; quality weight fixed at 1 | Implemented and CPU-tested; no GPU jobs submitted | Awaiting explicit approval; staged validation-only selection and checkpoint-bound promotion |
| Direct soft-target KL | Trie-derived next-token targets | Implemented, not run | No measured results |

The passage runs use actual on-policy REINFORCE updates to LoRA weights. The Shakespeare quality scorer is a frozen instruction-model judge, without preference training in this project. The later HelpSteer stages use a trained Bradley–Terry scorer or an audited pretrained reward model. Generated-answer evaluations remain **model-judged, not human-validated outcomes**.

## Emma 500-step baseline and length-control pilot

Job **524624** completed all 500 REINFORCE updates. Its own before/after
comparison uses 16 held-out prompts × four answers (64 answers per policy):

| Metric | Initial | Final |
|---|---:|---:|
| Raw training-judge quality ↑ | 0.7826 | 0.8806 |
| Target word KL ↓, observed word budgets | 0.5112 | 0.5028 |
| Target word KL ↓, matched 17,506-word budget | 0.5112 | 0.5661 |
| Word JSD ↓ | 0.06643 | 0.06943 |
| Mean generated words | 273.5 | 416.2 |
| Answers exceeding 350 words | 10/64 | 49/64 |

The raw KL reduction does not survive the word-count control. Quality is scored
by the same frozen judge used in training; it is not independent human evaluation.
The prompt-bootstrap quality-change interval is [0.0642, 0.1307], conditional on
these saved generations and one training seed. Different budgets and sampling
prevent a controlled optimizer comparison against the 80-batch GRPO experiment.
See [completed metrics and analysis](results/emma-v1/reinforce-long-4gpu-v1/).

The pending four-GPU job 529095 was cancelled before training at the user's
request. Its two-GPU replacement, **job 529144**, adds

```math
r'_i=r_i-0.2\,\frac{\max(0,250-L_i)+\max(0,L_i-350)}{100}.
```

This adjusted quality reward enters the same-prompt leave-one-out baseline;
word-distribution credit and base-policy regularization remain separate.
The coefficient was selected by CPU replay of the preceding run's first 50
training updates (800 answers), excluding test generations. This calibration
is not evidence of a trained-policy improvement.

The pilot requests **two H200s for at most 90 minutes / 3 GPU-hours** and stops
at **100 updates / 1,600 training answers**, retaining the 500-step cosine schedule
and 50-step warmup. Checkpoints preserve adapters, optimizer, scheduler and RNG
state. Evaluation uses validation prompts at 0/50/100; final pilot probes also
use validation, reserving the test split. **All 43 cluster CPU checks passed.**
The pilot **completed on November 26 in 42m34s (1.419 GPU-hours)**, with W&B synced.
On eight validation prompts × four answers, compliance increased from 14/32 to
19/32, raw quality was 0.80725 → 0.80704, and equal-9,449-word-budget KL was
0.74742 → 0.72833. JSD fell from 0.08560 to 0.08270. The mean length penalty
halved, but 13/32 answers still fall outside the requested band. The quality-change
95% prompt-bootstrap interval is [-0.0532, 0.0510]. These are exploratory
validation results using the training judge; a matched unpenalized control is
needed to isolate the penalty effect.
The longer four-H200 run, **job 530088**, completed 500 updates / 8,000 answers
in **1h59m14s**, using 7.949 GPU-hours of the 12-GPU-hour cap. It starts from
the original actor, matching job 524624's model, seed, batch and optimizer
schedule. See the [final results and comparison plan](results/emma-v1/length-penalty-long-4gpu-v1/).

On the same 16 test prompts × four answers, the new run achieves **39/64 in
range**, versus **15/64 without the penalty** and **25/64 for base**. Overlength
answers fall from 49/64 without the penalty to 18/64; mean length falls from
416.2 to 320.2 words. Raw judge quality is 0.83388, above base's 0.78261 but
below the unpenalized run's 0.88062. Equal-17,506-word-budget KL is 0.55886,
versus 0.56611 unpenalized and 0.51122 base. JSD is 0.07253, versus 0.06943
unpenalized and 0.06643 base. The earlier step-250 validation matching gain
did not establish final test improvement. The length penalty produces a clear
length/quality tradeoff; distribution matching remains unresolved.

The raw-quality change's prompt-bootstrap 95% interval is [0.0072, 0.0942]
versus base and [-0.0711, -0.0220] versus the unpenalized run. These condition
on saved answers from one seed. Test prompts have been previously inspected,
and quality uses the training judge; this is exploratory, not independent
human validation.

- [Pilot submission, exact config and CPU checks](results/emma-v1/length-penalty-pilot-2gpu-v1/)
- [Coefficient calibration](experiments/emma-word-kl/length-penalty-calibration.json)
- [Complete Emma experiment record](results/emma-v1/README.md)
- [W&B project](https://forge.coreweave.com/wandb/yash2007-university-of-washington/stochastic-llm)
- [LaTeX report sources](paper/README.md) — report scope is through November 22;
  the newer Emma results above are not yet incorporated into that manuscript.

## Completed stronger-word-KL run

**Job 535088** completed 1,000 updates / 16,000 training answers in
**4h10m50s** on four H200s (16.722 GPU-hours, below the 20-hour ceiling).
It uses quality weight 1, word-KL weight **0.5**, length coefficient 0.2,
reference-policy KL 0.002 and EOS penalty 0.1. The actor and 1,000-step cosine
schedule start fresh, with 50 warmup steps. W&B synced; checkpoint 1,000 is saved.

| Final test metric | Base | 500 steps, word KL 0.1 | 1,000 steps, word KL 0.5 |
|---|---:|---:|---:|
| Raw word KL ↓ | 0.51122 | 0.52828 | **0.45678** |
| Word JSD ↓ | 0.06643 | 0.07253 | **0.06159** |
| Training-judge quality ↑ | 0.78261 | 0.83388 | **0.84438** |
| In 250–350 words ↑ | 39.1% | 60.9% | 51.6% |
| Mean words | 273.5 | 320.2 | 335.0 |

Raw KL improves **10.65% over base / 13.54% over the 500-step control**;
JSD improves 7.28% over base. A secondary equal-word-budget KL check is
0.51122 → 0.48975 (4.20% lower). Raw quality's change interval is
[0.0230, 0.1001] versus base, but [-0.0203, 0.0417] versus the 500-step
control: the latter quality difference is uncertain. Length compliance loses
9.4 points versus that control, and 37.5% of answers remain over 350 words.
These are sixteen prompts × four answers, one seed, evaluated by the training
judge; the test prompts have been previously inspected. The stronger weight,
longer horizon and changed LR schedule cannot be separated by this comparison.
See [saved final outputs and analysis](results/emma-v1/stronger-word-kl05-1000-4gpu-v1/).
The grid remains deferred with zero jobs submitted.

## Reward-weight search prepared, awaiting approval

A CPU-only [grid-search workflow](experiments/emma-word-kl/GRID_SEARCH.md) now
prepares nine combinations: word-KL weights **0.1 / 0.3 / 1.0** and length
weights **0 / 0.2 / 0.4**, with quality weight 1 and reference-policy KL 0.002.
It retains the current setting as a control and progresses **nine trials at
150 steps → three at 400 → two at 1,000**, preserving a 1,000-step schedule,
optimizer and RNG state. Selection uses raw validation KL subject to frozen
quality/compliance criteria, not differently weighted training rewards.
The validation panel expands to sixteen prompts × four answers; test outputs
are excluded. Plans, training-code/data hashes and promotion inputs are checked.

**No search job is submitted or authorized.** The proposed initial-stage ceiling
is **54 GPU-hours** (nine four-H200 allocations capped at 90 minutes each).
Conditional intermediate/final-stage ceilings are 24 and 40 GPU-hours, for
**118 GPU-hours total maximum** before any separate seed-confirmation runs.
These are allocation ceilings; estimated training-only usage is substantially
lower, with setup/evaluation overhead and generation speed uncertain.
See the [reviewable preparation record](results/emma-v1/grid-search-preparation-v1/).

## Emma: source stability and the submitted word-KL experiment

The broad HelpSteer target pooled unrelated answers. A narrow fantasy keyword
filter in LitBench still mixed different settings and vocabulary. Emma instead
provides one author's narrative within a single novel. This is a hypothesis about
a more suitable pooled target, not an assumption that all chapters have one PMF.

### CPU source-text results, not trained-model performance

For a comparable screen, both Emma and the existing Tiny Shakespeare text were
divided into consecutive 4,000-word sections, with 70% assigned to references and
the remainder held out using seed 42. Each sampled batch contains sixteen
400-word passages, totaling exactly 6,400 words. The shared vocabulary consists
of the 1,000 most frequent words in the two corpora's **training sections only**,
plus OTHER. Forward KL uses total generated pseudocount mass 1.0. Each condition
contains 1,000 resampled batches.

| Source-text condition: median KL ↓ | Emma | Tiny Shakespeare |
|---|---:|---:|
| Random sixteen passages; shared 1,000 words + OTHER | **0.266** | 0.479 |
| Four sections × four passages; same shared vocabulary | **0.275** | 0.544 |
| Random sixteen passages; each corpus's own capped vocabulary | **0.638** | 1.002 |

Emma has lower discrepancy under this protocol. This does not establish that RL
will improve, or that author identity alone explains the difference. Shared
support hides distinctions inside OTHER; the own-vocabulary comparison has
different supports. Resampling percentiles describe finite-corpus batches, not
independent experiments or population confidence intervals.

### Actual continuation data and reward

The pinned NLTK Gutenberg edition of Emma contains 55 chapters and 160,874 words
under our existing tokenizer, excluding headings. Whole chapters are assigned
**39 train / 8 validation / 8 test**, independently of the comparable section
screen. Non-overlapping complete paragraphs produce **168 / 35 / 29** continuation
pairs: an 80–180-word prefix followed by a 250–450-word source continuation.
Length constraints leave eligible examples in 38 training chapters and seven
test chapters; no chapter crosses partitions.

The training word PMF counts only the **168 source continuations paired with the
exact training-prompt population**. Prefixes and held-out chapters are excluded.
Words need at least five occurrences, with a maximum vocabulary of 10,000;
this preparation yields **1,053 retained words plus OTHER**, holding 10.47% of
source word mass in OTHER. This is the selected training subset's support, not
the complete novel's vocabulary. Pooling still does not enforce prompt-specific
word frequencies.

Emma has **no preference rankings**. No Bradley–Terry reward model is trained for
this run. A frozen, pinned **Qwen2.5-7B-Instruct** judge scores coherence,
readability, and consistency with the provided narrative prefix. Each dimension
uses the expected rating conditional on tokens 1–5; rating-token mass below 0.01
fails scoring. If those expected ratings are \(s_d(q,y)\), quality is

\[
r_{\mathrm{quality}}(q,y)
=\frac{1}{3}\sum_{d=1}^{3}\frac{s_d(q,y)-1}{4}
-0.1\,\mathbf 1\{\text{generation did not terminate}\}.
\]

These are model judgments, not calibrated human-preference probabilities. At the
start of the same allocation, eight training-only source continuations and their
shuffled/repetitive corruptions form 24 sanity-check examples. Source coherence
must beat each corruption in at least 7/8 comparisons, source readability must
beat shuffling in at least 7/8, and the mean normalized quality margin over
repetition must be at least 0.1. A failed check stops the allocation before RL.
Passing these easy controls does not establish nuanced literary-quality accuracy.

For a generated batch \(\mathcal B=(Y_1,\ldots,Y_N)\), the objective is

\[
J(\theta)=\mathbb E\left[
\frac1N\sum_i r_{\mathrm{quality}}(q_i,Y_i)
-0.1\,D_{\mathrm{KL}}(P_{\mathrm{words}}\|\widehat Q_{\mathcal B})
\right]
-0.002\,\mathbb E_qD_{\mathrm{KL}}(\pi_\theta(\cdot\mid q)\|\pi_0(\cdot\mid q)).
\]

The word-PMF KL measures corpus fidelity; the separate sequence-policy KL
regularizes departure from the initial model. Word statistics are pooled across
complete responses, not rewarded after each individual word. The initial 80-step on-policy
REINFORCE run uses whole-batch leave-one-out quality credit and coupled distribution
credit \(N[D_{-i}-D_{\mathrm{full}}]\), with same-prompt leave-one-out centering
for base-policy KL. Four prompts × four samples form each global DDP batch.
Later runs use same-prompt quality baselines, and the custom GRPO pilot is recorded above. Lambda 0.1 remains a
heuristic starting value, not an optimized coefficient.

### Submitted run and reproducibility

**Job 519712 completed 80 updates and then failed during postprocessing.** The valid
JSONL samples contained a Unicode line separator; physical-line parsing fixes
the failure. Validation samples were recovered without new GPU work; the latest
saved weights are step 70 and no final test comparison exists. See the
[completed-run and CPU recovery record](results/emma-v1/README.md).
The original allocation
uses two H200s for at most **110 minutes**, including scorer checks, training,
checkpointing and evaluation. Together with cancelled LitBench job 519547's
eight minutes, maximum cumulative usage is **3.93 GPU-hours**, below the original
four-GPU-hour approval. There are no automatic retries or extra benchmark jobs.

The fresh Qwen2.5-3B-Instruct actor uses LoRA rank 16 / alpha 32 and AdamW learning
rate 5e-6. Generation uses temperature 1, no top-k/top-p truncation, and at most
1,024 new tokens. Training has an 80-update cap and an 80-minute setup/update
budget, reserving time for final evaluation; completing 80 updates is not
guaranteed. Both stages require online W&B in the existing
[stochastic-llm project](https://forge.coreweave.com/wandb/yash2007-university-of-washington/stochastic-llm).

Baseline and final evaluation use identical held-out prompts, sampling settings
and seeds: sixteen prompts, two responses each. Quality evaluation uses the
**same frozen training judge**, not independent or human evaluation. Generated
12-word-window overlap with the complete novel tracks copying. Chapter holdout
does not rule out base-model pretraining memorization. Held-out source statistics
were inspected during dataset selection, so this is not a pristine external
benchmark.

The preceding LitBench attempt trained neither a usable reward model nor an RL
policy: its initial validation report contained NumPy boolean scalars that could
not be JSON-serialized. The job was cancelled after eight minutes. The source now
converts these scalars to native booleans, and its regression test serializes the
actual report. The original uploaded runtime is not silently restarted.

- [Emma implementation, protocol and CPU reproduction](experiments/emma-word-kl/README.md)
- [Full CPU screening statistics](experiments/emma-word-kl/screening-results.json)
- [Emma submission and preparation metadata](results/emma-v1/README.md)
- [LitBench implementation and cancelled-attempt record](experiments/litbench-rl/README.md)

All twelve Emma CPU checks passed locally and on Tillicum before submission,
including actual two-rank policy updates and scorer-certificate export/reload.
The LitBench validation regression suite contains seven additional checks.

## HelpSteer results and conditional semantic MMD

The completed HelpSteer runs do not establish successful joint control of writing
quality and reference-word frequencies. They motivate changing the target from a
pooled vocabulary across unrelated questions to a distribution conditioned on
each exact question. Full metrics, baseline/final generations, training histories,
scorer audits, configs, and source hashes are recorded in
[results/helpsteer-v1](results/helpsteer-v1/README.md).

### Completed broad-domain and programming experiments

Each column pair below compares its own initial/final policy on 32 held-out
questions × four responses. **The two experiments use different questions,
reference corpora, and vocabularies; their absolute KL values are not a common
cross-experiment benchmark.** Both use λ = 0.5 and initial-policy KL coefficient
β = 0.002, with temperature 1, top-p 1, and top-k disabled.

| Metric | Broad: initial | Broad: final | Programming: initial | Programming: final |
|---|---:|---:|---:|---:|
| Optimized word-PMF KL ↓ | 1.9163 | 1.9923 | 0.5920 | 0.5880 |
| Full-vocabulary KL ↓ | 1.9163 | 1.9923 | 1.0729 | 1.0674 |
| Frozen quality reward ↑ | 0.5432 | 0.5674 | 0.6641 | 0.6585 |
| Independent coherence /5 ↑ | 4.023 | 4.080 | 4.510 | 4.464 |
| Independent readability /5 ↑ | 4.045 | 4.081 | 4.319 | 4.283 |
| Independent fulfillment /5 ↑ | 4.055 | 4.112 | 4.614 | 4.541 |
| Mean words per answer | 227.3 | 206.7 | 434.4 | 432.6 |

The broad run uses 4,948 preferred training references with 30,078 word types.
It completed 86 updates; its pooled unigram KL increased despite an improvement
in the reward model's quality score. The programming run uses 228 reference
answers, retaining 1,589 words occurring at least five times and preserving the
remaining 8.2% reference mass in OTHER. It completed 75 updates in **1h25m38s**
including saving and evaluation. Generated OTHER mass increased from 22.8% to
23.3%, and 20/128 final answers reached the generation cap. Its very small KL
reduction is not convincing evidence of reliable learning.

The corrected policy reward uses frozen
`Skywork/Skywork-Reward-V2-Qwen3-8B`. On the model-selection audit it reached
78.75% non-tied human-preference accuracy and passed the predefined engineering
gates; the legacy 3B scorer reached 67.14% and failed. This audit selected the
scorer, is not an independent final benchmark, and cannot rule out upstream
HelpSteer overlap. A separate Qwen2.5-7B-Instruct supplies final coherence,
readability, and fulfillment ratings; neither set of generated answers has human
quality labels.

- [Original HelpSteer preference/RL implementation](experiments/helpsteer-rl/README.md)
- [Corrected length handling, scorer audit, and broad PMF](experiments/helpsteer-rl-v2/README.md)
- [Programming selection, frequent-word PMF, and OTHER mathematics](experiments/helpsteer-programming/README.md)
- [Programming run on W&B](https://forge.coreweave.com/wandb/yash2007-university-of-washington/stochastic-llm/runs/d6hs9j2z)

### Conditional semantic-MMD objective and completed first run

The new experiment uses **102 training, 10 validation, and 19 test questions**,
retaining questions with two distinct model-written answers whose human
coherence, helpfulness, and correctness ratings are all at least 3/4. It keeps
the existing splits and assigns equal weight to those two reference prototypes.
They do not establish the true frequencies of every valid answer type.
No new teacher is trained or sampled.

Let `X_qa` be the frozen reference embeddings and `Z_qi` the embeddings of sampled
student answers to the same exact question. With `m = 2` fixed references and
`n = 4` independent student answers, the implemented cost is:

```math
\widehat d_q=
\frac{1}{n(n-1)}\sum_{i\ne j}k(Z_{qi},Z_{qj})
+\frac{1}{m^2}\sum_{a,b}k(X_{qa},X_{qb})
-\frac{2}{nm}\sum_{i,a}k(Z_{qi},X_{qa}).
```

The generated U-statistic excludes its diagonal; the fixed-reference V-statistic
includes its diagonal. The resulting estimate is unbiased for population MMD²
against this **fixed empirical reference measure**, assuming independent
conditional generation. It can be negative in finite samples and is not clipped.
It does not estimate distance to an unknown population of all valid human answers.

The batch contains four distinct questions, each with four answers. The objective
averages `mean(quality) − 0.5 × MMD²` equally across questions, with the same
β = 0.002 initial-policy regularizer. Passage distribution credit is
`n × [MMD²(without answer i) − MMD²(full question group)]`; quality leave-one-out
baselines also remain within their question. This is on-policy REINFORCE with
leave-one-out credit, without PPO clipping, GRPO normalization, or a learned critic.

The frozen, pinned MiniLM encoder processes every 256-token chunk of each answer;
chunk vectors are pooled without discarding the tail. A mixture of three Gaussian
kernels uses bandwidth calibrated from training references only. The original
word-PMF KL remains an evaluation diagnostic and does not enter the MMD reward.
Matching semantic features does not guarantee lower lexical KL or correct code.

**Job 504124 completed** with 94 updates and 1,504 sampled answers in
**81 minutes 36 seconds on two H200s**. On 19 test questions, MMD changed
0.39224 → 0.38330 and quality reward 0.66961 → 0.66314. Paired prompt-bootstrap
95% intervals for final-minus-baseline changes were [-0.03421, 0.01611] for MMD
and [-0.04043, 0.02798] for quality. These results do not establish an overall gain.

The subsequent [kCGM-inspired CPU audit](results/helpsteer-v1/kcgm-cpu-audit/README.md)
adds exact-question KL centering, frozen scale calibration, and an energy-distance
kernel. The corresponding GPU comparison **job 508718 completed**, running
Gaussian then energy sequentially in **1h43m52s on two H200s**. Both started from
the same pinned actor and reused the frozen quality scorer. No reward model or
teacher was trained. The table below records the completed comparison and follow-ups.

The calibrated objective to minimize is

```math
L(\theta)=\frac{-\mathbb E[r_{\mathrm{quality}}]+0.5s_k D_k(\pi_\theta,P)
+0.002D_{\mathrm{KL}}(\pi_\theta\|\pi_0)}{1.502},
```

with frozen scales `s_rbf = 2.29718` and `s_energy = 1.77101`, fitted on the
first ten saved training updates. The KL policy-gradient coefficient subtracts
the other same-question answers’ mean log-ratio. The energy kernel is
`k(x,y) = −||x−y||`, avoiding Gaussian bandwidth selection. Exact-gradient and
DDP checks passed; these are existing kCGM ideas adapted to our conditional
text experiment, not a claim of a new estimator. See the audit for limitations.

- [Completed first MMD run: metrics, samples, and configuration](results/helpsteer-v1/semantic-mmd/)
- [Completed comparison: metrics, samples, exact profiles, budget, and Slurm script](results/helpsteer-v1/kcgm-gpu-comparison/)
- [Full MMD mathematics, implementation, limitations, and run protocol](experiments/helpsteer-semantic-mmd/README.md)
- [Recorded CPU checks and submitted-job snapshot](results/helpsteer-v1/semantic-mmd-preparation/)

### Completed energy continuation and distribution-only diagnostic

All columns in the following table use the same 19 held-out questions and four
answers per question. Discrepancies are **unscaled** for comparison; quality is
model-judged. The longer run restores update 53, including optimizer, original
200-step cosine schedule, prompt RNG, and both ranks' RNG states.

| Metric | Base | Gaussian, 50 updates | Energy, 53 updates | Energy, 200 updates |
|---|---:|---:|---:|---:|
| Gaussian MMD ↓ | 0.3922 | 0.4077 | **0.3703** | 0.3904 |
| Energy distance ↓ | 0.4830 | 0.5082 | **0.4564** | 0.4813 |
| Quality reward ↑ | 0.6696 | 0.6615 | 0.6735 | **0.6940** |
| Independent coherence /5 ↑ | 4.517 | 4.461 | 4.494 | **4.532** |
| Independent readability /5 ↑ | 4.325 | 4.292 | 4.270 | **4.349** |
| Independent fulfillment /5 ↑ | 4.612 | 4.530 | 4.560 | **4.634** |
| Full-vocabulary word KL ↓ | 1.2104 | 1.2578 | **1.1889** | 1.2381 |

Energy training at update 53 reduced semantic discrepancies by about 5.5%, but
independent quality scores fell slightly. Continuing to update 200 increased
quality point estimates while losing most of that distribution-matching gain.
Paired question-bootstrap 95% intervals for semantic-discrepancy and quality-reward
changes include zero. One seed and 19 questions do not establish a reliable gain.
Job **509031** completed 147 additional updates / 2,352 answers in **2h00m03s**
on two H200s, ending at its 200-update target before the approved four-hour cap.

The subsequent **distribution-only diagnostic**, job **510187**, removes the
quality term completely and loads no reward scorer or judge. A fresh actor sees
four fixed training questions every update, with 16 independently sampled answers
each (64 globally, 32 per GPU). Four questions from the original validation split
are reserved for secondary unseen-question evaluation. The original test split
is unused. Selection rules and seed are recorded before training.

Its minimized objective is

```math
L_{\mathrm{diag}}(\theta)=\frac{
0.5s_E\,\mathbb E_q[D_E(P_\theta(\cdot\mid q),T_q)]
+0.002\,\mathbb E_q[D_{\mathrm{KL}}(\pi_\theta(\cdot\mid q)\|\pi_0(\cdot\mid q))]
}{1.502},\qquad s_E=1.7710087348.
```

The energy term matches answer embeddings to the two equal-weight reference
prototypes per question. The KL term limits changes from the original actor;
**it is not the earlier target-word-PMF KL**. Word-frequency KL is evaluated
post hoc and does not enter this diagnostic's reward. Coupled MMD leave-one-out
credit and same-question KL centering remain active; quality coefficients are
exactly zero. The historical divisor 1.502 is retained without rescaling the
remaining terms. Full mathematics, selection rules, and CPU checks are in the
[diagnostic guide](experiments/helpsteer-mmd-diagnostic/README.md).

The diagnostic completed **17 updates / 1,088 training answers** in **1h18m37s**
on two H200s, within its approved 90-minute cap. It evaluates 32 answers per seen
question and 16 per unseen question, before and after training.

| Energy distance ↓ | Before | After | Relative change |
|---|---:|---:|---:|
| Four seen training questions | 0.5261 | 0.5061 | 3.8% better |
| Four unseen validation questions | 0.5879 | 0.5969 | 1.5% worse |

Three of four training questions improved. Seen-question evaluations progressed
0.5261 → 0.5197 → 0.5061. The four-question bootstrap interval for the mean change
was [-0.0546, 0.0111], including zero. This is a modest fitting signal, not strong
overfitting or demonstrated generalization. Removing quality competition did not
resolve the difficulty; optimization, representation, and sparse targets remain
possible limitations. Quality was not measured in this diagnostic.

### Word-frequency audit of the diagnostic samples

A CPU-only analysis reuses the actual saved before/after generations. It compares
word counts with **each question's own two references**, then averages question
metrics equally. This differs from the older pooled programming-corpus PMF.
Forward KL uses total generated pseudocount mass 1 and retains out-of-reference
words in an OTHER category; JS and total variation are unsmoothed. Every word
participates in the numerical metrics. Reference answers are pooled by word count,
which differs from equal-answer weighting in the embedding target.

| Mean word-distribution metric ↓ | Seen: before | Seen: after | Unseen: before | Unseen: after |
|---|---:|---:|---:|---:|
| Forward unigram KL | 1.3325 | 1.2785 | 1.6379 | 1.6590 |
| Jensen–Shannon divergence | 0.2726 | 0.2712 | 0.3003 | 0.3005 |
| Total variation | 0.5478 | 0.5483 | 0.5923 | 0.5931 |

The roughly 4.1% seen-question KL reduction does not translate to an unseen gain
or consistent improvement across metrics. The plot displays the most frequent
reference content words; its stopword filtering affects display only.

![Word frequencies before and after the distribution-only diagnostic](results/helpsteer-v1/word-frequency-audit/word-frequency-comparison.png)

- [Gaussian / energy comparison records](results/helpsteer-v1/kcgm-gpu-comparison/)
- [Longer energy continuation records](results/helpsteer-v1/energy-long/)
- [Distribution-only diagnostic records and samples](results/helpsteer-v1/distribution-only-diagnostic/)
- [Word-frequency numerical report](results/helpsteer-v1/word-frequency-audit/word-frequency-report.json)
- [Readable actual sampled responses](results/helpsteer-v1/word-frequency-audit/sampled-responses.txt)
- W&B: [Gaussian](https://forge.coreweave.com/wandb/yash2007-university-of-washington/stochastic-llm/runs/9m8sxpla), [energy](https://forge.coreweave.com/wandb/yash2007-university-of-washington/stochastic-llm/runs/dyvs9hxd), [longer energy](https://forge.coreweave.com/wandb/yash2007-university-of-washington/stochastic-llm/runs/a4cbew2c), [distribution-only diagnostic](https://forge.coreweave.com/wandb/yash2007-university-of-washington/stochastic-llm/runs/wsqcmlnk).

Rebuild the word-frequency audit using the committed samples and reference inputs:

```bash
python experiments/helpsteer-mmd-diagnostic/analyze_word_frequencies.py \
  --source results/helpsteer-v1/distribution-only-diagnostic \
  --output outputs/reproduced-word-frequency-audit
```

CPU preparation can be rebuilt from the pinned public dataset. The embedding
checkpoint must be cached before the offline semantic preparation:

```bash
python experiments/helpsteer-rl-v2/prepare.py --raw outputs/helpsteer-data/raw
python experiments/helpsteer-programming/prepare.py
python -c "from huggingface_hub import snapshot_download; snapshot_download('sentence-transformers/all-MiniLM-L6-v2', revision='1110a243fdf4706b3f48f1d95db1a4f5529b4d41')"
python experiments/helpsteer-semantic-mmd/prepare.py
python -m unittest discover -s experiments/helpsteer-semantic-mmd -p 'test_mmd.py' -v
```

Preparation never submits a GPU job. Checkpoints, optimizer states, credentials,
downloaded models, raw training caches, and local deployment archives remain
outside Git. Results retain the pinned HelpSteer2 attribution and provenance.

## Unigram versus bigram passage RL

The completed unigram run changes only the distribution target from adjacent
word pairs to single words. It reuses the quality reward: the frozen
Qwen2.5-7B-Instruct judge scores coherence, fluency, and prompt fulfillment.
Both runs start fresh from the same pinned Qwen2.5-3B-Instruct model and use
the same prompts, seed, decoding settings, LoRA, and optimization settings.
The final test contains the same 16 held-out scenarios, four passages each.

| Held-out metric | Initial policy | Bigram-trained | Unigram-trained |
|---|---:|---:|---:|
| Unigram KL ↓ | 3.0598 | 2.7766 | **2.5876** |
| Bigram KL ↓ | 8.0710 | 7.9282 | **7.8699** |
| Quality reward /1 ↑ | 0.7419 | **0.8029** | 0.7874 |
| Judge coherence /5 ↑ | 3.8499 | **3.9985** | 3.9821 |
| Judge fluency /5 ↑ | 3.9618 | **4.0354** | 4.0285 |
| Judge prompt fulfillment /5 ↑ | 4.0913 | **4.6011** | 4.5827 |
| Words absent from reference ↓ | 11.37% | 9.66% | **7.88%** |
| Mean words per passage | 96.88 | 115.52 | 137.17 |
| Responses ending with EOS | 64/64 | 64/64 | 63/64 |

Unigram training reduced raw unigram KL by **15.4%** relative to the initial
policy and improved mean judge quality on **14/16 test prompts**. Its quality
reward is slightly below the bigram run, with eight prompts better and eight
worse. An exploratory paired-prompt bootstrap interval for this quality
difference includes zero; this does not establish a quality winner.

Longer outputs change histogram sample counts. Matching each policy to **6,200
word observations**, averaged over 200 subsamples without replacement, gives
unigram KL **3.0598 → 2.8589 → 2.7006** for initial, bigram-trained, and
unigram-trained policies. The unigram advantage remains about **5.5%** relative
to the bigram-trained policy. This diagnostic ignores passage dependence and
does not eliminate sparse-histogram estimation bias.

The unigram job **499702** completed successfully in **1h45m57s on two H200s**
(about **3.53 GPU-hours**). It stopped at the 105-minute training budget after
1,097 updates, then saved and evaluated; the bigram run reached 1,200 updates.
The schedules were configured identically, but actual update counts differ.
No lambda sweep was performed. Both use **lambda = 0.1** and initial-policy
KL coefficient **beta = 0.002**.

**Interpretation:** a positive distribution-control result with improved
judge-rated quality over the base model. Single-word matching gives better
measured frequency fit while retaining most of the bigram run's quality gain.
This is one seed with one reward/evaluation judge, no human validation, and
only 16 templated test scenarios. A better quality reward is a hypothesis to
test, not an improvement demonstrated by this run.

- [Unigram methods, results, and reproduction](results/passage-unigram-v1/README.md)
- [All saved test/validation passages and training metrics](results/passage-unigram-v1/)
- [Count-matched and prompt-bootstrap diagnostics](results/passage-unigram-v1/final-diagnostics.json)
- [W&B unigram run](https://forge.coreweave.com/wandb/yash2007-university-of-washington/stochastic-llm/runs/77envg9c)

## Completed bigram passage-RL results

The longer run starts from the pinned **Qwen2.5-3B-Instruct** checkpoint, not the numeric adapter or the final short-run adapter. It generates complete 80–120-word dramatic monologues. Word bigrams are measured across the generated collection; word pairs are not independently concatenated to construct the prose.

The final adapter and the original instruction policy were evaluated on the same **16 held-out scenarios × 4 generations = 64 passages per policy**, with matching decoding settings and per-rank sampling seeds. The original policy is recovered by disabling LoRA; its base weights remain frozen during training.

| Held-out metric | Before passage RL | After 1,200 updates |
|---|---:|---:|
| Pooled bigram KL ↓ | 8.0710 | **7.9282** |
| Judge coherence /5 ↑ | 3.8499 | **3.9985** |
| Judge fluency /5 ↑ | 3.9618 | **4.0354** |
| Judge prompt fulfillment /5 ↑ | 4.0913 | **4.6011** |
| Combined quality reward /1 ↑ | 0.7419 | **0.8029** |
| Out-of-reference bigram rate ↓ | 63.51% | **56.15%** |
| Mean words per passage | 96.88 | 115.52 |
| Responses ending with EOS | 64/64 | 64/64 |

Raw pooled bigram KL fell **1.77%**. Mean judge quality improved on **15 of 16 prompts**, with the largest change in prompt fulfillment. Both policies terminated every test response, so this quality gain is not the truncation artifact observed in the short run. The shorter run's KL was 7.9903 → 7.9935 and showed no clear improvement; its different prompt set prevents a compute-only comparison with the longer experiment.

The longer run completed in **1h41m21s on two H200s**, approximately **3.38 GPU-hours**, within a two-hour allocation / four-GPU-hour cap. It reached its 1,200-update limit rather than stopping on the time budget.

![Validation quality and empirical bigram KL during passage RL](results/passage-rl-v1/ddp/validation-trends.png)

Validation uses 32 passages; final test evaluation uses 64. Their absolute KL values are not directly comparable. Final outputs are about 19.2% longer, which changes the number of observed bigrams. A post hoc count-matched diagnostic still found a small reduction (8.0710 → 7.8762); it is a sampling-sensitivity check, not a correction proving exact distribution matching.

**Interpretation:** a modest positive result under this frozen judge and a small improvement in aggregate word-pair statistics. One training seed, one judge shared between reward and evaluation, only 16 templated test scenarios, and biased sparse histogram estimates limit the conclusion. Some samples remain awkward. The experiment does not establish high-quality Shakespearean writing, exact corpus sampling, absence of memorization, or superiority to continued SFT or MaD-RL.

- [Full passage report, diagnostics, and a matched sample pair](results/passage-rl-v1/ddp/RESULTS.md)
- [All 128 baseline/final test passages](results/passage-rl-v1/ddp/) and [training history](results/passage-rl-v1/ddp/history.jsonl)
- [Short-run results](results/passage-rl-v1/pilot/comparison.json) and [judge-audit records](results/passage-rl-v1/)
- [Detailed method and reproduction](docs/passage-rl-experiment.md)
- [W&B main DDP run](https://forge.coreweave.com/wandb/yash2007-university-of-washington/stochastic-llm/runs/zop09jj3) · [final analysis](https://forge.coreweave.com/wandb/yash2007-university-of-washington/stochastic-llm/runs/3buboztw)

## Passage-RL mathematics and implementation

### Reference and generated bigram distributions

The reference is the first 80% of eligible paragraphs from a pinned Tiny Shakespeare text snapshot. Eligible paragraphs have at least ten normalized words. Words are lowercased, internal apostrophes are retained, punctuation is excluded, and speaker-name words are retained. Counts do not cross paragraph boundaries. This is a discrete probability mass function, not a continuous density:

```math
p_b = \frac{C_{\mathrm{ref}}(b)}{\sum_{b'} C_{\mathrm{ref}}(b')},
\qquad p_{\mathrm{UNKNOWN}}=0.
```

There are 83,454 observed reference word pairs plus an UNKNOWN category. For a generated batch of passages $\mathcal B=(Y_1,\ldots,Y_N)$, let $n_i(b)$ count pair $b$ in passage $i$, and let $M=\sum_i\sum_b n_i(b)$. Unseen pairs contribute to UNKNOWN; an empty or one-word passage contributes one UNKNOWN observation. Generated passages are never joined to create boundary bigrams. With $B=83{,}455$ categories and total pseudocount mass $\eta=1$:

```math
a=\frac{\eta}{B},\qquad
\widehat q_b(\mathcal B)=\frac{\sum_i n_i(b)+a}{M+\eta},
\qquad
G(\mathcal B)=-\sum_{b:p_b>0}p_b\log\frac{p_b}{\widehat q_b(\mathcal B)}.
```

All generated pairs remain in the denominator. This is a smoothed, finite-batch approximation; matching it does not determine long-range meaning or all within-passage structure.

### Frozen quality reward

A pinned **Qwen2.5-7B-Instruct** evaluates coherence, fluency, and prompt fulfillment separately. It is asked for a rating from 1 to 5. Instead of parsing generated JSON, the corrected scorer reads the next-token probabilities of those five single-token digits. For dimension $d$, with frozen judge $\rho_\phi$:

```math
s_d(Y,c)=\sum_{k=1}^{5} k\,
\frac{\rho_\phi(k\mid d,c,Y)}{\sum_{j=1}^{5}\rho_\phi(j\mid d,c,Y)},
\qquad
r_Q(Y,c)=\mathbf 1_{\mathrm{EOS}}\,
\frac{\left(s_{\mathrm{coherence}}+s_{\mathrm{fluency}}+s_{\mathrm{task}}\right)/3-1}{4}.
```

The rating is conditional on those five tokens, not a calibrated human score. Their total original probability is recorded; mass below 0.01 causes failure. A truncated passage gets zero positive quality reward but still contributes its observed word pairs to the histogram. Style ratings from the initial audit do not enter the training reward. The corrected scorer passed predefined corruption checks, which do not substitute for human preference validation.

### Joint objective, coefficients, and credit assignment

With $\pi_0$ the initial instruction policy, the objective combines mean passage quality, aggregate bigram matching, and a separate reverse sequence-KL stability penalty:

```math
J(\theta)=
\mathbb E_{\mathcal C,\,\mathcal B\sim\pi_\theta}
\left[\frac1N\sum_{i=1}^{N}r_Q(Y_i,c_i)+\lambda G(\mathcal B)\right]
-\beta\,\mathbb E_c D_{\mathrm{KL}}\!\left(\pi_\theta(\cdot\mid c)\|\pi_0(\cdot\mid c)\right),
\qquad \lambda=0.1,\quad\beta=0.002.
```

The reference-policy KL is different from the target-bigram KL. **λ = 0.1 was a heuristic starting value; no λ sweep has been run.** The reward weights were identical in the short and longer passage experiments.

Write $\ell_i=\log\pi_\theta(Y_i\mid c_i)$ and $\ell_i^0=\log\pi_0(Y_i\mid c_i)$, summing completion-token log probabilities including EOS when present. The implementation assigns:

```math
A_i^Q=r_Q(Y_i,c_i)-\frac{1}{N-1}\sum_{j\ne i}r_Q(Y_j,c_j),
\qquad
A_i^D=N\left[G(\mathcal B)-G(\mathcal B\setminus\{Y_i\})\right],
```

```math
A_i=A_i^Q+\lambda A_i^D-\beta(\ell_i-\ell_i^0),
\qquad
\mathcal L_{\mathrm{RL}}=-\frac1N\sum_i\operatorname{stopgrad}(A_i)\,\ell_i.
```

Removing a passage removes all its overlapping bigrams and recomputes the denominator. The factor $N$ compensates for averaging sequence losses. There is no GRPO normalization, PPO clipping, learned value critic, or gradient through the frozen judge. The score-function estimator targets the expected finite-batch reward; it is not an unbiased estimator of exact population bigram KL. Dropout is disabled and each generated group is used for one update.

### What DDP and the longer schedule change

The longer run uses 256 training scenarios, 16 validation scenarios, and 16 test scenarios, generated from disjoint role/setting/dilemma pools. These are synthetic combinations, not 256 independently authored stories. Each update chooses two training scenarios and generates eight passages for each: **global group size 16**. Each GPU hosts an actor and frozen judge and processes eight passages. Both ranks contribute to one histogram and one leave-one-out quality baseline; rank zero broadcasts the combined credits.

DDP averages the two locally averaged gradient estimates, preserving the global mean:

```math
\frac12\left(\frac18\sum_{i\in\mathrm{rank}\,0}g_i+
\frac18\sum_{i\in\mathrm{rank}\,1}g_i\right)
=\frac1{16}\sum_{i=1}^{16}g_i.
```

Only attention LoRA weights train: rank 16, alpha 32, dropout 0, BF16 base models, AdamW learning rate $5\times10^{-6}$, 40 warmup updates, cosine decay over 1,200 updates, and gradient norm clipping at 1. Global batch size and learning rate are not doubled for DDP. Decoding uses temperature 1, top-p 1, top-k disabled, and at most 256 new tokens.

Validation and complete checkpoints run every 100 updates. Checkpoints save adapter, optimizer/scheduler, prompt-selection RNG, both ranks' RNG states, and source/config hashes. The two-hour allocation has no automatic requeue; training has a 105-minute cutoff to reserve final evaluation and saving time. No extra GPU benchmark preceded the DDP run. CPU tests verify global reward/gradient equivalence and checkpoint restoration.

### Reproduce the recorded passage analysis without a GPU

```bash
python -m pip install -e '.[dev,analysis]'
python experiments/shakespeare-quality-pilot/analyze_recorded.py
python -m unittest discover -s experiments/shakespeare-quality-pilot -p 'test_*.py' -v
```

The analysis reads committed samples and corpus counts; it does not generate text or allocate GPUs. Full training rollouts, final adapters, and source artifacts are available through the W&B run. Model weights, optimizer states, environments, and credentials remain outside Git. See [the passage guide](docs/passage-rl-experiment.md) for training commands and the [results manifest](results/passage-rl-v1/manifest.json) for checksums and provenance.

## Completed KL-reward RL results

The final 250-update RL adapter and its starting SFT adapter were evaluated on the same **100 held-out prompts × 1,000 generations per prompt**, using identical decoding settings. Lower distribution distances are better.

![KL-reward RL comparison](results/rl-v1/analysis/rl-vs-sft.png)

| Metric | SFT | SFT + KL-reward RL | Relative reduction |
|---|---:|---:|---:|
| Mean histogram KL | 0.2277 | 0.2226 | 2.2% |
| Median histogram KL | 0.1191 | 0.1057 | 11.3% |
| Median KS distance | 0.1392 | 0.1268 | 8.9% |
| Median normalized Wasserstein distance | 0.1122 | 0.1013 | 9.8% |
| Numeric-only responses with EOS | 98.68% | 98.59% | — |

This is a modest incremental improvement: KL improved on 61 of 100 prompts. Normal and uniform improved on median KS; exponential and Weibull regressed. The unseen-family aggregate worsened slightly on median KS and Wasserstein. One training seed and no equal-compute SFT continuation control limit conclusions.

The algorithm is on-policy **REINFORCE with a leave-one-out batch reward baseline**, not PPO or GRPO. It optimizes negative smoothed forward histogram KL with a separate reference-policy KL penalty. The run used 128,000 sampled training responses and took **22m34s** on one H200; evaluation took **5m43s**. Test histogram KL is computed post hoc from 1,000 draws, whereas training/validation groups contain 256; their absolute KL values are not directly comparable.

- [Full RL report](results/rl-v1/analysis/RESULTS.md)
- [Implementation, hyperparameters, and reproduction](docs/kl-rl-experiment.md)
- [Recorded results, all SFT/RL test generations, and training history](results/rl-v1/)
- [W&B comparison dashboard](https://forge.coreweave.com/wandb/yash2007-university-of-washington/stochastic-llm/runs/lyb3xv9l)
- [W&B RL training](https://forge.coreweave.com/wandb/yash2007-university-of-washington/stochastic-llm/runs/tx91cho0)
- [W&B final evaluation](https://forge.coreweave.com/wandb/yash2007-university-of-washington/stochastic-llm/runs/n9d9aegn)

## SFT results

Both checkpoints were evaluated on the same **100 held-out prompts × 1,000 independent generations**, for 200,000 model responses in total.

![SFT results: numeric response validity, distribution mismatch, and example cumulative distributions](assets/sft-results.png)

| Metric | Pretrained base | After SFT |
|---|---:|---:|
| Numeric-only responses, terminated with EOS | 6.84% | **98.68%** |
| Median normalized Wasserstein distance ↓ | 1.6858 | **0.1122** |
| Median KS distance ↓ | 0.5428 | **0.1392** |
| Out-of-support values among numeric responses | 19.87% | **1.35%** |

Lower distribution distances are better. Wasserstein is normalized by the target's 95th–5th percentile width; KS is the largest gap between the empirical and target cumulative distributions. Medians aggregate prompts.

For normal distributions, median KS improved from **0.2613 to 0.0763**. For held-out Weibull and Poisson families combined, it improved from **0.4016 to 0.1696**.

The result is encouraging, but calibration remains imperfect: a direct SciPy reference produced median KS **0.0247** and normalized Wasserstein **0.0113**, using matched valid sample counts and 20 repetitions per case.

### How to interpret these results

- Formatting is a substantial confound: 88.6% of base responses reached the 32-token limit. A format-matched control is needed to isolate the calibration effect.
- Distribution distances use only valid numeric responses, including out-of-support values. Four base prompts had no valid answers and are excluded from its distance medians.
- Numeric validity requires one finite number and EOS; it does not guarantee the number lies within the target's support.
- One training seed was used. These tests do not establish exact sampling, independence, general language retention, or an effect specifically attributable to RLHF.

## Experiment at a glance

| Setting | Value |
|---|---|
| Base model | [`Qwen/Qwen2.5-3B`](https://huggingface.co/Qwen/Qwen2.5-3B) |
| Training data | 64,000 examples: 2,000 parameter configurations × 32 draws |
| Training families | Normal, uniform, exponential, beta, gamma, binomial |
| Held-out families | Weibull, Poisson |
| Adaptation | LoRA rank 16, alpha 32, dropout 0.05 |
| Optimizer / learning rate | AdamW / `2e-4` |
| Training budget | 2,000 updates, effective batch size 32, one epoch |
| Hardware / precision | One NVIDIA H200 / BF16 |
| Training job time | 4m50s, including a 4m14s training loop |

Evaluation used temperature 1, top-p 1, top-k disabled, and no repetition penalty. Tests cover unseen parameter ranges, held-out distribution families, and new prompt wording. Invalid responses were not repaired or resampled.

## Numeric code and recorded artifacts

The repository includes the SFT training/evaluation pipeline, pinned configs, per-prompt results, the numerical-sampler reference, and the two plotted cases' raw generation logs.

- [Full SFT setup, reproduction commands, and limitations](docs/sft-experiment.md)
- [Recorded metrics and provenance](results/sft-v1/)
- [Per-prompt base results](results/sft-v1/base/results.csv) · [Per-prompt SFT results](results/sft-v1/sft/results.csv)
- [Download the figure as PDF](results/sft-v1/sampling-results.pdf)
- [W&B results dashboard](https://forge.coreweave.com/wandb/yash2007-university-of-washington/stochastic-llm/runs/zkuvylag)
- [W&B training run](https://forge.coreweave.com/wandb/yash2007-university-of-washington/stochastic-llm/runs/xcj01bg1)

Full raw generations and trained adapter artifacts are stored with the W&B runs. Model weights, credentials, environments, and scheduler logs are not checked into Git.

### Inspect the results locally

```bash
python3.12 -m venv .venv
source .venv/bin/activate
python -m pip install -e '.[dev,analysis]'
python -m pytest -q
python scripts/analyze_results.py
```

The analysis command regenerates the figure from the committed results without a GPU. Tests use a tiny local model and offline tracking. Training the full checkpoint requires CUDA; see the [SFT reproduction guide](docs/sft-experiment.md).
## Original technical overview and equations

The following retains the original research formulation. The completed implementation uses this batch-KL reward with a leave-one-out baseline and a separate reference-policy KL penalty; measured SFT and RL results are reported above.

---

# StochTune
Trying to make a LLM that can potentially generate numbers pseudo-randomly from some target distribution


### Information
# Training LLMs to Sample from Probability Distributions

This project investigates whether a language model can learn to generate numbers according to a requested probability distribution.

The goal is **distributional fidelity**: repeated, independent responses should have the correct frequencies, spread, and tail behavior.

For example, given:

> Sample exactly one number from a normal distribution with mean 0 and standard deviation 1.

Repeated responses should approximately follow:

```math
X \sim \mathcal{N}(0,1).
```

Generating individual numbers that look plausible is insufficient. Correct sampling is a property of the distribution of responses across many requests.

## Research Question

Can fine-tuning with a reward based on distributional mismatch teach an LLM to become a better conditional sampler?

We investigate generalization to:

- Unseen parameters of familiar distributions.
- Distribution families absent from training.
- Different prompt wording.
- Eventually, explicitly defined distributions over nonnumeric choices.

## Problem Formulation

Let:

- $c$ denote a prompt specifying a distribution and its parameters.
- $p(x \mid c)$ denote the requested target distribution.
- $\pi_\theta(y \mid c)$ denote the probability of generating response $y$.
- $f(y)$ parse a response into a number, or return $\bot$ for an invalid response.

The model induces an output distribution:

```math
q_\theta(x \mid c)
=
\sum_{y:f(y)=x} \pi_\theta(y \mid c).
```

Our objective is to bring this induced distribution closer to the requested target.

## Batch-Level KL Reward

For each prompt, generate $N$ independent responses:

```math
Y_1,\ldots,Y_N
\overset{\mathrm{iid}}{\sim}
\pi_\theta(\cdot \mid c),
\qquad
X_i=f(Y_i).
```

Each response is generated from a fresh context containing the same prompt.

### Discretizing the Target

We compare probability masses over bins rather than directly comparing an empirical sample with a continuous density.

For numeric bins $B_1,\ldots,B_K$, define:

```math
p_j(c) = \Pr(X \in B_j \mid c).
```

For a continuous target with density $p(x \mid c)$:

```math
p_j(c) = \int_{B_j} p(x \mid c)\,dx.
```

Add an invalid-output category, indexed by $\bot$, with target probability:

```math
p_\bot(c)=0.
```

Malformed responses, unfinished responses, and values outside the distribution's support are assigned to this category.

### Estimating the Model Distribution

Let $n_j$ be the number of generated responses assigned to category $j$.

Using additive smoothing with $\alpha>0$:

```math
\widehat{q}_j
=
\frac{n_j+\alpha}{N+\alpha(K+1)}.
```

Smoothing prevents an unobserved target bin from producing an infinite KL estimate.

### Reward Function

The batch reward is the negative forward KL divergence:

```math
R(c,Y_{1:N})
=
-D_{\mathrm{KL}}\!\left(
p(\cdot \mid c)
\,\middle\|\,
\widehat{q}
\right).
```

Explicitly:

```math
D_{\mathrm{KL}}(p\|\widehat{q})
=
\sum_{j:p_j>0}
p_j \log\frac{p_j}{\widehat{q}_j}.
```

**Higher reward means closer agreement with the target distribution.**

Invalid responses remain in the sample count and normalization. Discarding them before computing the reward would hide poor response validity.

## Reinforcement Learning Objective

We maximize expected batch reward:

```math
J(\theta)
=
\mathbb{E}_{c}
\mathbb{E}_{Y_{1:N}\sim\pi_\theta}
\left[R(c,Y_{1:N})\right].
```

A policy-gradient estimator is:

```math
\nabla_\theta J
=
\mathbb{E}\left[
\sum_{i=1}^{N}
(R-b_i)
\nabla_\theta \log\pi_\theta(Y_i \mid c)
\right].
```

One possible baseline is a leave-one-out reward:

```math
b_i = R(c,Y_{-i}),
```

computed by removing response $i$ and recomputing the histogram and its normalization.

The corresponding sampled loss is:

```math
\mathcal{L}_{\mathrm{RL}}
=
-\sum_{i=1}^{N}
\mathrm{stopgrad}(R-b_i)
\log\pi_\theta(Y_i \mid c).
```

The reward is computed from generated numbers, while gradients flow through the probabilities of their generated token sequences.

A shared batch reward must not simply be mean-centered within that same batch: identical rewards would produce zero advantages.

## Experimental Comparisons

We distinguish three training approaches:

| Method | Training signal |
|---|---|
| Sampled-target SFT | Cross-entropy on answers sampled from the target distribution |
| Direct KL supervision | Match target next-token probabilities |
| Batch-reward RL | Reward distributions estimated from the model's own generated samples |

The proposed RL experiment starts from the SFT checkpoint to investigate improvements beyond the existing formatting and sampling baseline.

The initial model is **Qwen2.5-3B**, adapted using **LoRA**.

## Evaluation

We evaluate independent samples using:

- Numeric response validity.
- Support violations.
- Empirical CDFs and quantile coverage.
- Kolmogorov–Smirnov distance.
- Wasserstein distance.
- Tail behavior and repeated-value concentration.

A conventional numerical sampler provides a reference for the error expected from finite samples.

Evaluation uses held-out prompts and parameters. Improvement on the training reward alone does not establish accurate sampling.

## Limitations

The reward measures agreement at the chosen bin resolution. It cannot fully detect errors within bins, and finite batches introduce estimation noise and smoothing bias.

Matching the training histogram is therefore insufficient: evaluation must also examine finer distributional structure.

The objective is calibrated use of inference-time sampling randomness. It does not guarantee exact continuous sampling, cryptographic randomness, or a new source of entropy.



## References and acknowledgments

These sources informed the motivation, implementation, evaluation, and related-work
comparisons. Each entry states its role; citing a method does not mean this repository
implements it or reproduces its results. Exact dataset/model revisions used in our
runs are recorded in their configurations and source manifests.

### Controlled stochasticity and distribution matching

| Reference | Role in StochTune |
|---|---|
| Kuriakose & Baldelli (2026), [Training Language Models for Controlled Stochasticity](https://www.lesswrong.com/posts/upCrcE39aF63GoLJh/training-language-models-for-controlled-stochasticity-2) | Original user-supplied blog motivating learned stochastic generation and distributional fidelity. |
| Baldelli et al. (2026), [Probabilistic Calibration Is a Trainable Capability in Language Models](https://arxiv.org/abs/2605.11845) | Calibration Fine-Tuning reference: sampled-target supervision and trie-derived soft targets. Our numeric SFT and implemented-but-unrun direct soft-target trainer relate to these two approaches; our batch-reward RL uses a different objective. |
| Diamant & Trippe (2026), [Calibrating Generative Models to Feature Distributions with MMD Finetuning](https://arxiv.org/abs/2606.19496) · [author code](https://github.com/smithhenryd/cgm) | Close prior work for MMD distribution matching with coupled leave-one-out policy gradients and initial-policy KL. Its implementation informed KL centering, frozen scale calibration, and the energy-distance comparison. These components are not claimed as new estimators. |
| Gretton et al. (2012), [A Kernel Two-Sample Test](https://jmlr.org/papers/v13/gretton12a.html) | MMD foundations and sample estimators underlying our conditional semantic-distribution objective. |
| Sejdinovic et al. (2013), [Equivalence of distance-based and RKHS-based statistics in hypothesis testing](https://arxiv.org/abs/1207.6076) | Mathematical background for the relationship between energy distance and kernel discrepancies in the Gaussian-versus-energy comparison. |

### Policy gradients and preference rewards

| Reference | Role in StochTune |
|---|---|
| Williams (1992), [Simple statistical gradient-following algorithms for connectionist reinforcement learning](https://doi.org/10.1007/BF00992696) | REINFORCE / score-function policy-gradient foundation. |
| Ahmadian et al. (2024), [Back to Basics: Revisiting REINFORCE Style Optimization for Learning from Human Feedback in LLMs](https://arxiv.org/abs/2402.14740) | REINFORCE with leave-one-out baselines in LLM post-training. Our distribution reward additionally couples multiple generated answers. |
| Bradley & Terry (1952), [Rank Analysis of Incomplete Block Designs: I. The Method of Paired Comparisons](https://doi.org/10.1093/biomet/39.3-4.324) | Pairwise preference model underlying the legacy HelpSteer reward-training loss. |
| Ouyang et al. (2022), [Training language models to follow instructions with human feedback](https://arxiv.org/abs/2203.02155) | User-supplied RLHF reference for human rankings, reward modeling, and policy optimization. Our updater uses REINFORCE rather than this paper's PPO recipe. |
| Hu et al. (2021), [LoRA: Low-Rank Adaptation of Large Language Models](https://arxiv.org/abs/2106.09685) | Adapter parameterization used to fine-tune the actors and legacy reward model. |

### Quality evaluation and human-feedback data

| Reference | Role in StochTune |
|---|---|
| Liu et al. (2023), [G-Eval: NLG Evaluation using GPT-4 with Better Human Alignment](https://arxiv.org/abs/2303.16634) | Reference for rubric-based, probability-weighted LLM evaluation. Our Qwen dimension-wise judge is a project-specific adaptation, not a reproduction of G-Eval. |
| Zheng et al. (2023), [Judging LLM-as-a-Judge with MT-Bench and Chatbot Arena](https://arxiv.org/abs/2306.05685) | Reference for judge limitations and biases, motivating corruption checks and caution about model-judged quality. |
| Wang et al. (2024), [HelpSteer2: Open-source dataset for training top-performing reward models](https://arxiv.org/abs/2406.08673) · [dataset](https://huggingface.co/datasets/nvidia/HelpSteer2) | Human attribute ratings used to select acceptable references and check quality scorers. The answers themselves are model-written. |
| Wang et al. (2024), [HelpSteer2-Preference: Complementing Ratings with Preferences](https://arxiv.org/abs/2410.01257) | Human pairwise preferences used by the HelpSteer preference-training and scorer-audit stages. |
| Liu et al. (2025), [Skywork-Reward-V2: Scaling Preference Data Curation via Human-AI Synergy](https://arxiv.org/abs/2507.01352) · [8B reward model](https://huggingface.co/Skywork/Skywork-Reward-V2-Qwen3-8B) | Source of the frozen pretrained quality scorer used after our legacy scorer audit. We did not train Skywork or establish a contamination-free benchmark. |
| [LitBench paper](https://arxiv.org/abs/2507.00769) · [published version](https://aclanthology.org/2026.eacl-long.362/) · [training data](https://huggingface.co/datasets/SAA-Lab/LitBench-Train) · [test release](https://huggingface.co/datasets/SAA-Lab/LitBench-Test) | Literary-preference dataset explored in the cancelled fantasy attempt. Training labels derive from curated Reddit-vote comparisons, not expert coherence ratings. We used internal splits, not the official test benchmark. |

### Models, embeddings, and corpus sources

| Source | Role in StochTune |
|---|---|
| [Qwen2.5 Technical Report](https://arxiv.org/abs/2412.15115) · [Qwen2.5-3B](https://huggingface.co/Qwen/Qwen2.5-3B) · [Qwen2.5-3B-Instruct](https://huggingface.co/Qwen/Qwen2.5-3B-Instruct) · [Qwen2.5-7B-Instruct](https://huggingface.co/Qwen/Qwen2.5-7B-Instruct) | Numeric base model, passage/HelpSteer actor, and instruction-model judge, respectively. |
| [all-MiniLM-L6-v2 model card](https://huggingface.co/sentence-transformers/all-MiniLM-L6-v2) · Reimers & Gurevych (2019), [Sentence-BERT: Sentence Embeddings using Siamese BERT-Networks](https://arxiv.org/abs/1908.10084) | Frozen answer embeddings and sentence-embedding background. Our full-answer chunk pooling is a project-specific extension, not a validated code-correctness metric. |
| Karpathy, [Tiny Shakespeare / char-rnn](https://github.com/karpathy/char-rnn/tree/master/data/tinyshakespeare) | Corpus source for the Shakespeare unigram and bigram target PMFs. |
| Austen, [Emma / Project Gutenberg](https://www.gutenberg.org/ebooks/158) · [pinned NLTK Gutenberg distribution](https://github.com/nltk/nltk_data/blob/5f09e470fe339e1a119dd174ca284129a3c7a5bb/packages/corpora/gutenberg.zip) | Single-novel source for the new continuation corpus and training-only word PMF. The text archive, extracted member and prepared data have recorded hashes. |
| Eldan & Li (2023), [TinyStories](https://arxiv.org/abs/2305.07759) · [dataset](https://huggingface.co/datasets/roneneldan/TinyStories) | Controlled-vocabulary synthetic-story candidate considered during corpus selection; no TinyStories training has been run here. |
| Novikova, Dušek & Rieser (2017), [The E2E Dataset: New Challenges For End-to-End Generation](https://arxiv.org/abs/1706.09254) · [author data](https://github.com/tuetschek/e2e-dataset) | Narrow restaurant-domain, multiple-reference candidate considered during corpus selection; not used in a training run. |

### Additional related work considered

These papers informed comparisons and discussion; their complete training procedures
have not been implemented or evaluated as baselines in this repository.

- Shao et al. (2024), [DeepSeekMath: Pushing the Limits of Mathematical Reasoning in Open Language Models](https://arxiv.org/abs/2402.03300): GRPO and within-prompt group advantages, informing the custom coupled-credit Emma GRPO pilot; the longer Emma runs use REINFORCE.
- Zhang et al. (2024), [Forcing Diffuse Distributions out of Language Models](https://arxiv.org/abs/2404.10859): fine-tuning for diffuse distributions over valid outputs.
- Khalifa, Elsahar & Dymetman (2021), [A Distributional Approach to Controlled Text Generation](https://arxiv.org/abs/2012.11635): pointwise and distributional constraints with divergence control relative to an initial language model.
- Anschel et al. (2025), [Group-Aware Reinforcement Learning for Output Diversity in Large Language Models](https://aclanthology.org/2025.emnlp-main.1649/): GAPO and frequency-aware group rewards for output diversity.
- Kulkarni et al. (2026), [MaD-RL: Matching Distributions for Calibrating LLMs with Reinforcement Learning](https://arxiv.org/abs/2609.31644): distribution-matching RL and divergence-based rewards for output attributes.
