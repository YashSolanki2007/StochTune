# Emma experiment record

Job **519712** was submitted on November 24, 2026 (Pacific time). At the recorded
status check it was **pending for priority**; no policy results were available.
This is a historical snapshot, not live job monitoring.

- `submission.json`: actual Slurm submission, coefficients, resources and CPU checks.
- `status-snapshot.json`: state observed when these files were prepared.
- `data-manifest.json`: exact source revision, hashes, chapter splits and prepared counts.
- [CPU screening statistics](../../experiments/emma-word-kl/screening-results.json).
- [Protocol and reproducible scripts](../../experiments/emma-word-kl/README.md).

One job is capped at 110 minutes on two H200s, including frozen-judge checks, RL
and evaluation. Combined with the cancelled eight-minute LitBench allocation,
the maximum is 3.93 GPU-hours. Online W&B is mandatory for both GPU stages.
Model weights, raw corpora, authentication files and local environments are not
included; pinned CPU preparation reconstructs the data.
