# Single stronger-word-KL run: job 535088

## Final results: November 28, 2026

Job 535088 completed all 1,000 updates / 16,000 training answers on four H200s
in **4h10m50s** (16.722 GPU-hours), exit 0:0. Checkpoint 1,000 is saved, and
W&B synced. No grid jobs were submitted. Final probes use the same sixteen
test prompts × four answers as the previous runs; base passages are identical.

| Metric | Base | 500 steps, word KL 0.1 | 1,000 steps, word KL 0.5 |
|---|---:|---:|---:|
| Raw target word KL | 0.511220 | 0.528284 | **0.456778** |
| Word JSD | 0.066426 | 0.072532 | **0.061592** |
| Raw training-judge quality | 0.782605 | 0.833882 | **0.844382** |
| In 250–350-word band | 25/64 | 39/64 | 33/64 |
| Under / over band | 29 / 10 | 7 / 18 | 7 / 24 |
| Mean words | 273.531 | 320.172 | 334.953 |
| Mean length penalty | 0.065250 | 0.034531 | 0.049063 |

Raw KL is **10.65% lower than base** and **13.54% lower than
the previous length-controlled 500-step run**; JSD improves 7.28% over
base. Judge quality improves 0.061777 over base, with prompt-bootstrap 95%
interval [0.022966, 0.100142]. The quality difference versus the 500-step run
is only 0.010500, interval [-0.020257, 0.041736], so superiority on quality
over that policy is uncertain. Twelve of sixteen prompts improve versus base.

As a secondary sensitivity check, equal-17,506-word-budget KL is
0.511220 → 0.489752 (4.20% lower; 2,000 hypergeometric subsamples,
seed 20261128). Recomputing the 500-step control with this resampling seed
gives 0.558558; its earlier report used seed 20261126 and reported 0.558859.
The difference is Monte Carlo resampling variation, not changed generations.
Count matching is not a histogram-KL bias correction.

Length compliance improves over base but falls from 60.9% to 51.6% relative to
the preceding length-controlled run, with 37.5% of answers over 350 words.
All final generations terminate. This run supports the combined stronger-KL /
longer-training configuration, but cannot isolate the weight, duration or LR
schedule effects. One seed, reused test prompts, and quality assessment by the
training judge limit the conclusions; this is exploratory evidence.

`final-analysis.json` records settings, source hashes, count diagnostics and
10,000-resample prompt-bootstrap intervals. `comparison.json` contains the
original baseline/final report; validation files retain all intermediate probes.
The earlier submission record below is preserved for provenance.

Submitted November 27, 2026 (Pacific) after explicit user approval. Target:
1,000 REINFORCE steps / 16,000 training answers, four H200s, five-hour maximum
(20 GPU-hours). See status-snapshot.json for the observation time and scheduler
state; it is not live status. No model results were available at submission.

Weights: quality 1; word-distribution KL **0.5**; length **0.2 per 100 words
outside 250–350**; reference-policy KL **0.002**; nontermination penalty **0.1**.
This supersedes the unsubmitted 0.3 draft. It is one fresh actor, not a grid or
continuation. New 1,000-step cosine schedule, 50 warmup steps, learning rate
5e-6, existing LoRA rank 16 / alpha 32. Global batch 16 = four prompts × four
answers. Judge, target PMF, models/data revisions and sampling are unchanged.
Checkpoints every 25 and validation every 50; 30-minute final-reporting reserve.
The time deadline can stop training before the 1,000-step target.

All **50 CPU tests passed on Tillicum**. The installed sequence-likelihood
implementation matches the local source. W&B project access was verified;
online W&B logging is mandatory during both GPU stages. No automatic retries,
continuation, grid jobs or additional seeds are authorized/submitted.

Validation uses eight prompts × four answers; final probes use sixteen test
prompts × four answers. Judge quality is not independent human evaluation,
existing test prompts have been inspected, and changing both training horizon
and word-KL weight prevents attributing any change to just one factor.
The old 500-step runs and 100-step pilot remain recorded separately.
