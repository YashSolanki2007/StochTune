# Two-H200 length pilot: job 529144

Replaces job 529095, cancelled while pending before any training. Submitted
November 26, 2026 at the user's request to reduce the GPU count for scheduling.
No automatic continuation or retry is authorized. See status-snapshot.json for
the observation time and scheduler state; this is not live status.

100 REINFORCE updates / 1,600 training answers; global batch 16, four prompts
with four answers each. Two H200s, 16 CPUs, 128 GB RAM, 90-minute maximum
(3 GPU-hours). Stop at step 100 with the original 500-step cosine schedule
and 50 warmup steps. The length penalty remains 0.2 per 100 words outside
250–350, before prompt-LOO quality credit. Target PMF, lambda 0.1, beta 0.002,
and validation protocol are unchanged. Online W&B is required. All 43 cluster
CPU checks passed. Adapter/optimizer/scheduler/RNG state is checkpointed.
The time deadline may stop training before 100 updates; final evaluation has
20 minutes reserved. Rank-dependent sampling differs from a four-rank run.

Assess raw judge quality, length compliance, and equal-word-budget distribution
metrics separately. The submission snapshot had no results; the completed results follow below.

## Completed results

Job 529144 completed 100 updates / 1,600 training answers in 42m34s on two
H200s (1.419 GPU-hours), exit 0:0. The step-100 checkpoint and final reports
were saved and W&B synced. The before/after probes use eight validation
prompts × four answers (32 answers per policy), not the test split.

| Metric | Initial | Step 100 |
|---|---:|---:|
| Raw training-judge quality | 0.807250 | 0.807035 |
| Penalized quality reward | 0.742125 | 0.774848 |
| Answers within 250–350 words | 14/32 | 19/32 |
| Under 250 / over 350 words | 10 / 8 | 6 / 7 |
| Mean words | 300.781 | 295.281 |
| Mean length penalty | 0.065125 | 0.032188 |
| Raw word KL | 0.743017 | 0.728334 |
| Equal-9,449-word-budget KL | 0.747417 | 0.728334 |
| Word JSD | 0.085601 | 0.082696 |

Equal-word KL is 2.55% lower and JSD 3.39% lower. Count matching uses 2,000
hypergeometric subsamples, seed 20261126; it is a sensitivity diagnostic, not
KL bias correction. Raw quality change is -0.000215 with a prompt-bootstrap
95% interval [-0.053204, 0.051014] (10,000 resamples, same seed). All probes
terminated. Step 50 has lower KL than step 100, so matching is not monotonic.

This is preliminary evidence for improved length compliance with little change
in the raw quality point estimate; 13/32 answers remain outside the band.
One seed, eight validation prompts, and use of the training judge limit the
conclusion. No matched unpenalized control isolates the penalty's causal effect.
The test set was not generated for this pilot. `validation-analysis.json`
records resampling settings, limitations and source hashes; `comparison.json`
contains the final deterministic baseline/final validation probes.
