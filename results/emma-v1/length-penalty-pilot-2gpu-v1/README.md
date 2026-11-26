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
metrics separately. No training result was available at submission.
