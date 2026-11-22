# Recorded HelpSteer and programming experiments

These are completed-run records downloaded from Tillicum, plus CPU preparation
and records for completed conditional semantic-MMD and diagnostic experiments.
No new GPU work was started while exporting these results.

## Completed policy comparisons

Each row compares that run's initial actor with its own final adapter. All use
the pinned Qwen2.5-3B-Instruct actor and on-policy passage-level REINFORCE. The
first three rows share a subset/evaluation protocol; the broad and programming
rows use different targets and test prompts. Absolute KL values across those
protocols are not a common performance benchmark.

| Run directory | Job | λ | Updates | Test answers per policy | Initial → final target KL | Initial → final quality reward |
|---|---|---:|---:|---:|---:|---:|
| [Subset quality-only](subset-quality-only/comparison.json) | 500689 | 0 | 200 | 32 | 2.7988 → 2.8067 | 0.3663 → 0.3293 |
| [Subset quality + unigram](subset-quality-unigram/comparison.json) | 500689 | 0.1 | 200 | 32 | 2.7988 → 2.8444 | 0.3663 → 0.3623 |
| [Subset λ = 0.5](subset-lambda05/comparison.json) | 501019 | 0.5 | 200 | 32 | 2.7988 → 2.8801 | 0.3663 → 0.3461 |
| [Corrected broad-domain](broad-v2/comparison.json) | 501413 | 0.5 | 86 | 128 | 1.9163 → 1.9923 | 0.5432 → 0.5674 |
| [Programming frequent words + OTHER](programming-unigram/comparison.json) | 501852 | 0.5 | 75 | 128 | 0.5920 → 0.5880 | 0.6641 → 0.6585 |

The subset scorer was trained using human preferences in this project. Its
protocol imposed short inputs and removed positive quality reward from capped
answers. The corrected broad/programming runs reuse the audited frozen
Skywork-Reward-V2-Qwen3-8B, honor both EOS tokens, and retain content quality
with an explicit 0.1 cap penalty. Changes in scorer, reference support, context
handling, and evaluation composition prevent a simple cross-protocol ablation.

The broad run improved the reward-model quality point estimate while worsening
word KL. The programming run showed a very small KL decrease and slightly worse
quality point estimates. Neither establishes robust joint improvement. There is
one training seed, sparse-histogram bias, possible reward-model data overlap,
and no human evaluation of the generated policy answers.

For programming, the word target retains 1,589 types with at least five training
occurrences and includes the omitted mass in OTHER. Full-vocabulary KL changed
1.07295 → 1.06736. OTHER's reference probability is 8.196%; generated mass changed
22.758% → 23.338%. Final answers averaged 432.6 normalized words, and 20/128 did
not terminate before the 1,024-token ceiling. Job 501852's scheduler elapsed time
was 1h25m38s; `comparison.json` records the slightly shorter policy-process time.

## Files and provenance

Completed-run directories contain the original files when present:

- `comparison.json`, baseline/final metrics, and validation metrics.
- `baseline-samples.jsonl` and `final-samples.jsonl`: actual generations, prompt
  context, token IDs, frozen reward values, and independent model judgments.
- `history.jsonl`: per-update training measurements.
- `run_config.json`, `source-manifest.json`, and `wandb_run.json`.

Source manifests describe the code deployed for that historical run; the current
implementation may have changed since those hashes were recorded. Original
source/checkpoint artifacts remain associated with each W&B run. This export
does not contain weights, optimizer state, raw training rollouts, credentials,
or downloaded training-data caches.

The [legacy](reward-audit-legacy/checks.json) and
[Skywork](reward-audit-skywork/checks.json) audit directories record the stronger
v2 model-selection checks and saved predictions. Non-tied preference accuracy
was 67.14% and 78.75%, respectively; only Skywork passed all predeclared gates.
The validation audit selected the scorer and is not an independent final test.
Upstream HelpSteer overlap for the public pretrained reward model is unknown.

[Programming preparation](programming-preparation/) includes the fixed vocabulary,
CPU coverage report, and prepared-data manifest. The latter records hashes of
files rebuilt from the public dataset; it does not mean every training-data
file is included here.

## Conditional semantic MMD: preparation only

[Semantic-MMD preparation](semantic-mmd-preparation/) records the 102/10/19
train/validation/test questions, two reference vectors per question, fitted
training-only bandwidth, CPU corruption checks, deployed configuration, and
submitted-job snapshot. The vectors represent model-written, human-rated answers.
The two equally weighted prototypes are an experimental target, not a measured
distribution of every valid answer or semantic mode.

Job **504124** was pending priority at the November 20, 2026, 1:44 PM Pacific
check. Its request is **two H200s for 90 minutes**, at most three GPU-hours,
with a 75-minute policy-stage budget. The job was shortened in place, retaining
its original submission time. The saved start estimate is transient.
There are **no semantic-MMD policy results** in this export.

See [the MMD implementation and mathematics](../../experiments/helpsteer-semantic-mmd/README.md)
for the generated U-statistic, fixed-reference V-statistic, within-question
leave-one-out policy gradient, encoder pooling, and limitations. Word KL is
diagnostic in that experiment; the policy distribution reward is semantic MMD.

## Attribution and integrity

HelpSteer prompts, human ratings, and source references originate from
[NVIDIA HelpSteer2](https://huggingface.co/datasets/nvidia/HelpSteer2), revision
`990b2711a36180dd19d9c94b8627844866f8982a`, under
[CC BY 4.0](https://creativecommons.org/licenses/by/4.0/). Policy completions are
generated experimental outputs. MiniLM identity and reference processing are
recorded with the semantic preparation metadata.

`manifest.json` lists SHA-256 checksums of the exported files. Read-only analysis
of these records does not require a GPU or a new W&B run.

## Semantic MMD and kCGM follow-up

Job **504124** completed: [metrics, samples, training history, configuration and source hashes](semantic-mmd/). Gaussian MMD improved slightly while quality fell slightly; paired prompt-bootstrap intervals include zero. The [CPU replay audit](kcgm-cpu-audit/README.md) records KL centering, frozen scaling, energy-distance diagnostics, and their limitations. The subsequent [GPU comparison](kcgm-gpu-comparison/) completed as **job 508718**, followed by [energy continuation job 509031](energy-long/) and [distribution-only diagnostic job 510187](distribution-only-diagnostic/). The [word-frequency audit](word-frequency-audit/word-frequency-report.json) uses committed diagnostic samples and reference inputs. See the root README for mathematics, budgets, result tables, uncertainty, and limitations.
