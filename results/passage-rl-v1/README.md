# Recorded passage experiments

These are completed measurements, not new training runs. The method and
equations are in the [repository README](../../README.md) and
[passage guide](../../docs/passage-rl-experiment.md).

| Directory | Contents |
|---|---|
| [ddp](ddp/) | Completed 1,200-update, two-H200 run: 64 baseline + 64 final test passages, all saved validation passages, metrics, update history, report, figure, and run identities |
| [pilot](pilot/) | Completed 100-update, one-H200 run and its before/after evaluation |
| [quality-audit](quality-audit/) | Original JSON judge audit, including all raw scores and missing-score failures |
| [quality-calibration](quality-calibration/) | Corrected dimension-wise judge audit and its predefined perturbation checks |
| [timing](timing/) | Five-update timing measurement preceding the short run; not an independent capability result |
| [corpus](corpus/) | Pinned Tiny Shakespeare source, selected reference, held-out paragraphs, audit controls, and corpus manifest |

[provenance.json](provenance.json) identifies the model revisions, Slurm jobs,
and GPU budget. [manifest.json](manifest.json) gives SHA-256 checksums for the
recorded files and current source/documentation snapshot. The DDP run's own
[source-manifest.json](ddp/source-manifest.json) preserves its original run-time
source hashes independently of later analysis and documentation additions.

The longer run improved the frozen judge's scores and reduced raw empirical
bigram KL by 1.77%; the short run showed no clear benefit. There is one training
seed and no independent human quality evaluation. See the
[full report](ddp/RESULTS.md) for the count-matched diagnostic and limitations.

Model weights, optimizer states, full training rollouts, and original W&B source
artifacts are available through the
[training run](https://forge.coreweave.com/wandb/yash2007-university-of-washington/stochastic-llm/runs/zop09jj3).
They are not included in Git. None of the analysis commands allocate GPUs.
