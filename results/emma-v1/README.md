# Emma experiment record

## Latest status (November 28, 2026)

- **Completed:** job 535088, 1,000 updates / 16,000 training answers on four
  H200s in 4h10m50s, exit 0:0. Word-KL weight 0.5. Final raw KL improves
  0.51122 → 0.45678, JSD 0.06643 → 0.06159, judge quality 0.78261 → 0.84438.
  Compliance is 51.6%, above base but below the preceding 500-step run's 60.9%.
  W&B synced and checkpoint 1,000 saved; no grid jobs submitted. See
  [final results and analysis](stronger-word-kl05-1000-4gpu-v1/).

- **Prepared, not run:** nine-configuration reward-weight grid, with validation-only
  ranking and 150/400/1,000-step checkpoint promotions. No GPU approval or jobs.
  See [reviewable preparation](grid-search-preparation-v1/) and
  [workflow/resource ceilings](../../experiments/emma-word-kl/GRID_SEARCH.md).

- **Completed:** longer length-control job 530088, 500 updates / 8,000 answers
  on four H200s, 1h59m14s elapsed, exit 0:0. Final test compliance is 60.9%,
  versus 23.4% unpenalized; raw judge quality 0.83388 versus 0.88062 unpenalized
  and 0.78261 base. Equal-word KL 0.55886 remains above base's 0.51122;
  JSD also worsens. No joint quality/distribution success is established.
  See [final results and comparison analysis](length-penalty-long-4gpu-v1/).

- **Completed:** job 524624, 500 REINFORCE updates / 8,000 answers on four H200s,
  2h17m20s elapsed. Raw judge quality 0.7826 → 0.8806; mean words 273.5 → 416.2;
  count-matched word KL 0.5112 → 0.5661 and JSD 0.06643 → 0.06943 both worsen.
  See [saved generations, metrics and analysis](reinforce-long-4gpu-v1/).
- **Completed:** custom GRPO job 520789, 80 rollout batches / 160 optimizer steps;
  small, uncertain changes. See [results](grpo-run-v1/).
- **Completed:** length-penalty pilot job 529144; 100 updates / 1,600 answers,
  two H200s, 90-minute / 3 GPU-hour maximum. All 43 cluster CPU checks passed.
  The 500-step LR schedule is preserved; no automatic continuation or retry.
  Replaces job 529095, cancelled while pending. W&B is mandatory.
  Completed in 42m34s; validation compliance 43.8% → 59.4%, raw judge quality
  0.80725 → 0.80704, equal-word KL 0.74742 → 0.72833. W&B synced.
  See [submission and pilot protocol](length-penalty-pilot-2gpu-v1/).

Quality uses the frozen training judge, not independent human ratings. Equal-word
KL is a sensitivity diagnostic; neither completed Emma run establishes reliable
joint quality/distribution improvement. The following sections retain historical
submission snapshots and their subsequent results.

## Original 80-step REINFORCE run

Job **519712** completed all **80 updates / 1,280 training answers**, then exited
with an error during validation postprocessing, before final test evaluation.
Scheduler elapsed time was **34m09s** on two H200s. The last saved model is
checkpoint 70; the step-80 validation generations and aggregate metrics survive.

The JSONL file was valid: one answer contained U+2028, which Python's
`str.splitlines()` incorrectly treated as a record boundary inside a JSON string.
The reader now uses physical-line iteration. All 16 samples at each validation
probe (0/20/40/60/80) were recovered without modifying the raw files or generating
new answers. Future checkpoints are saved before validation, including the final
step, so diagnostics cannot discard already completed scheduled checkpoints.
The fixes were deployed to the isolated Tillicum runtime with original files
backed up. All 15 Emma CPU tests passed locally and on the cluster; 17 shared
HelpSteer/resume tests passed locally. No replacement GPU job was submitted.

| Validation metric | Initial | Step 80 |
|---|---:|---:|
| Target word KL | 0.96125 | 0.91009 |
| Word JSD | 0.09797 | 0.09228 |
| Training-judge quality | 0.82969 | 0.84662 |
| Mean words | 293.25 | 310.81 |
| Count-matched word KL (4,692 words) | 0.96125 | 0.92815 |

Count matching uses 2,000 subsamples without replacement, seed 20261124; the
roughly 3.44% reduction is a sensitivity diagnostic, not a KL-bias correction.
The eight-question quality-change bootstrap interval is [-0.04982, 0.08826],
including zero (10,000 resamples, same seed). These are validation results,
not final test results or human quality ratings; no checkpoint was selected by
these retrospective diagnostics. Step 20 has the strongest matching point
estimate, so progress is not monotonic.

`run-519712/recovery/` contains the recovered metrics, source hashes, analysis
settings and cluster CPU-test log. Reproduce from the repository root:

```bash
.venv/bin/python experiments/emma-word-kl/analyze_run.py \
  --input results/emma-v1/run-519712 \
  --output outputs/emma-recovery-reproduced
```

- `submission.json`: actual Slurm submission, coefficients, resources and CPU checks.
- `status-snapshot.json`: observed failure and verified CPU recovery/fix status.
- `data-manifest.json`: exact source revision, hashes, chapter splits and prepared counts.
- [CPU screening statistics](../../experiments/emma-word-kl/screening-results.json).
- [Protocol and reproducible scripts](../../experiments/emma-word-kl/README.md).

One job is capped at 110 minutes on two H200s, including frozen-judge checks, RL
and evaluation. Combined with the cancelled eight-minute LitBench allocation,
the maximum is 3.93 GPU-hours. Online W&B is mandatory for both GPU stages.
Model weights, raw corpora, authentication files and local environments are not
included; pinned CPU preparation reconstructs the data.

## GRPO pilot submitted

After explicit user approval, **job 520789** was submitted on November 24,
2026 (Pacific) using the prepared custom GRPO-style updater. It requests two
H200s for at most 110 minutes including judge checks and final evaluation.
The target is 80 fresh rollout batches / 1,280 training answers / 160 optimizer
steps (two passes per batch). Online W&B is mandatory. Only this GRPO job was
submitted; the matched REINFORCE control remains unsubmitted. There are no
GRPO policy results at submission. `grpo-run-v1/` records the actual authorized
configuration, CPU preflight, Slurm submission and transient queue snapshot.

### Completed GRPO outcome (job 520789)

The pilot completed successfully: 80 fresh rollout batches, 1,280 training
answers and 160 optimizer steps, in 39m39s on two H200s. Final test evaluation
uses the same 16 held-out prompts and four draws per prompt (64 answers per
policy), with matched seeds and decoding.

| Test metric | Initial | GRPO final |
|---|---:|---:|
| Target word KL | 0.48822 | 0.48570 |
| Count-matched word KL | 0.48822 | 0.49095 |
| Word JSD | 0.06441 | 0.06678 |
| Training-judge quality | 0.80106 | 0.81374 |
| Mean words | 277.09 | 284.69 |
| Termination | 100% | 100% |

Count matching uses 17,734 words and 2,000 subsamples without replacement, seed
20261124. Thus the raw 0.52% KL decrease does not survive this length/count
sensitivity check: matched KL is about 0.56% higher. The quality mean change
is +0.01268, exploratory paired 16-prompt bootstrap interval [-0.02643,0.04917]
(10,000 resamples); it includes zero. All three training-judge rubric means
increase slightly, and generated 12-word-window overlap with the novel falls
7.98% to 5.61%. These are not independent human quality or originality measures.
Second-pass task clipping activates on only about 0.138% of tokens on average.

There is no convincing joint distribution/quality gain in this pilot, and no
matched REINFORCE control was run. The earlier REINFORCE results were validation
measurements with different sample counts, not this final test comparison.
`grpo-run-v1/comparison.json` and `final-analysis.json` contain source metrics,
CPU diagnostics and sample hashes; actual baseline/final samples are included.
The prompt bootstrap conditions on these saved generations, ignores within-chapter
clustering, and does not measure training-seed uncertainty. Count matching does
not remove histogram estimator bias. No additional GPU work was started.

## Four-H200 longer REINFORCE run submitted

After the user selected four GPUs with the proposed three-hour cap, **job
524624** was submitted on November 25, 2026 (Pacific). It targets 500 REINFORCE
updates / 8,000 training answers, with the same global four-prompt by four-answer
batch. Each GPU receives one prompt and four answers. The four-rank CPU update
matches the two-rank update on the same fixed examples; all 31 staged tests
passed locally and on Tillicum. Calibration covers each example exactly once
across four ranks. This starts fresh with a 500-step learning-rate schedule.

The allocation uses four H200s, 32 CPUs and 256 GB host RAM on one node, capped
at three hours / 12 GPU-hours including judge checks, training, saving and
final evaluation. Training/setup stops by 2h30m to reserve 30 minutes for final
work, so the update target is subject to the cap. Online W&B is required.
Only this job was submitted; the two-H200 draft and additional control jobs
were not submitted. No automatic retry is configured. `reinforce-long-4gpu-v1/`
contains the authorized config, CPU test log, preflight, submission and transient
scheduler snapshot. There are no policy results at submission.

### Completed 500-step four-H200 result

Job 524624 completed successfully with all 500 REINFORCE updates / 8,000 training
answers, in 2h17m20s (about 9.16 GPU-hours, below the 12-GPU-hour cap). Baseline
and final evaluation use the same 16 test prompts / seven held-out chapters,
four draws each (64 answers per policy), under matched within-run seeds.

| Test metric | Initial | Final |
|---|---:|---:|
| Training-judge quality | 0.78261 | 0.88062 |
| Coherence /5 | 4.37541 | 4.80278 |
| Readability /5 | 3.98623 | 4.28623 |
| Prompt fulfillment /5 | 4.02962 | 4.47837 |
| Target word KL | 0.51122 | 0.50278 |
| Count-matched target word KL | 0.51122 | 0.56611 |
| Word JSD | 0.06643 | 0.06943 |
| Mean words | 273.53 | 416.22 |
| Termination | 100% | 100% |

The quality reward increases by 0.09801 (12.52% relative), with 14/16 prompt
means improving. The paired prompt-bootstrap 95% interval is [0.06419,0.13067].
A chapter-cluster bootstrap gives [0.07703,0.10937]. Both use 10,000 resamples,
seed 20261126, and condition on one seed's saved generations and the same
training judge. They do not establish human-rated quality improvements;
52.16% longer answers may affect judge preferences.

Raw target KL decreases only 1.65%. At a common 17,506 observed words, 2,000
subsamples without replacement give KL 0.51122 -> 0.56611 (10.74% worse), while
JSD rises 4.52%. Thus the raw KL decrease does not demonstrate better word
matching after count control. Count matching is a sensitivity diagnostic, not
a KL-bias correction; the nonlinear pooled-KL percentile bootstrap must not be
read as an independent calibrated matching test. Full-word diagnostic KL changes
1.17704 -> 1.13389, but generated OTHER mass increases 25.21% -> 26.20% against
reference 10.47%. Generated 12-word-window overlap with the novel falls 9.48%
to 0.43%, which alone does not exclude memorization or prove originality.

This is a strong signal under the training judge, alongside unsuccessful robust
distribution matching. It does not isolate optimizer superiority over the
80-batch GRPO run: training lengths, schedules, GPU counts and sampling streams
differ. The final checkpoint is step 500; `reinforce-long-4gpu-v1/` contains the
comparison, all final samples, validation/history metrics, source records and
`final-analysis.json` with source hashes and analysis settings. No additional
GPU run was started to analyze these results.

[W&B policy run](https://forge.coreweave.com/wandb/yash2007-university-of-washington/stochastic-llm/runs/52ahjgsi).
