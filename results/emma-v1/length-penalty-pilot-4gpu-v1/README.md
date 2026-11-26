# Length-penalty pilot: job 529095

Submitted November 26, 2026 after explicit user approval. At the last recorded
snapshot the job was pending scheduler priority; no training result is available.
See `status-snapshot.json` for the observation time, not a live status.

- 100 REINFORCE updates, 16 answers/update, four H200s.
- 500-step cosine schedule, 50 warmup steps, stop at step 100.
- 90-minute allocation cap, maximum 6 GPU-hours; no automatic extension or retry.
- Length penalty 0.2 per 100 words outside 250–350, before quality LOO credit.
- Existing word-distribution target, lambda 0.1 and reference beta 0.002 unchanged.
- Validation at 0, 50, 100; final pilot evaluations also use validation prompts.
- Full optimizer, scheduler, adapter and RNG checkpoints; online W&B required.
- All 42 cluster CPU checks passed before submission.

Check raw training-judge quality separately from penalized reward, compliance
and length distributions. Compare distribution metrics with equal word budgets.
Pilot validation and the training judge do not establish independent test quality.
The separate 500-step, three-hour length experiment remains unsubmitted.
