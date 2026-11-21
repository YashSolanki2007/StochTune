# kCGM-inspired CPU replay audit

Implemented and checked on November 21, 2026. **No new GPU allocation, generation,
or policy training was performed.** These are replay diagnostics and implementation
checks, not improvements from a newly trained model.

Source: completed semantic-MMD job **504124**, 94 updates and 1,504 sampled answers,
376 exact-question groups. See the [source W&B run](https://forge.coreweave.com/wandb/yash2007-university-of-washington/stochastic-llm/runs/3q2tgmmj).
The original saved policy advantages are reproduced with maximum absolute error
**1.79e-8**. Calibration and source hashes are in [calibration.json](calibration.json);
the full numerical comparison is in [report.json](report.json).

## Three changes

1. **Initial-policy KL centering.** For each answer, subtract the mean log-ratio
   of the other answers to the same question, gathered across both DDP ranks.
   The baseline excludes the answer itself. Exact finite-action expectations and
   a real tiny-model two-rank update verify the expected gradient. This adds one
   current-policy forward pass per answer. The original run did not save individual
   policy/reference log probabilities, so its empirical KL variance reduction
   cannot be measured. Future profiles save these probabilities and coefficient RMS.
2. **Frozen scale calibration.** Use updates 1–10 (40 training-question groups)
   to set each kernel scale to the reciprocal of its mean discrepancy. Freeze
   the scale and diagnose on updates 11–94. No validation/test answers select it.
   The calibration samples come from changing early training policies, not a
   fresh fixed initial-policy sample. This is a replay-derived starting point.
3. **Energy distance.** Compare the original Gaussian mixture with
   `k(x,y) = -||x-y||`. The latter gives energy distance and needs no bandwidth.
   Both retain the generated U-statistic, fixed-reference V-statistic, and the
   same exact-question leave-one-out policy-gradient coefficients.

The prospective population objective to minimize is

```math
L(\theta)=\frac{-\mathbb E[r_{quality}]+0.5s_k D_k(\pi_\theta,P)
+0.002 D_{KL}(\pi_\theta\|\pi_0)}{1+0.5+0.002}.
```

Here `s_rbf = 2.2971824514`, `s_energy = 1.7710087348`. The fixed divisor is
**1.502**. It preserves all coefficient ratios after kernel scaling; it is not
per-batch reward standardization and does not guarantee identical optimizer
trajectories under Adam or gradient clipping. The kernel scaling deliberately
changes distribution pressure relative to quality and initial-policy KL.

## Findings

RMS of passage policy-gradient **coefficients**, measured on updates 11–94:

| Component | Original scale, before divisor | Calibrated, before divisor | Calibrated, after divisor |
|---|---:|---:|---:|
| Quality | 0.12268 | 0.12268 | 0.08168 |
| Weighted Gaussian MMD | 0.06175 | 0.14185 | 0.09444 |
| Weighted energy distance | 0.08932 | 0.15819 | 0.10532 |

Only the Gaussian column corresponds to the actual old training reward. Energy
values are counterfactual replay calculations on those same saved answers.
Gaussian and energy passage credits have correlation **0.95146**. Calibration
makes distribution coefficients comparable to quality coefficients, but coefficient
RMS is not gradient norm or gradient variance. The different questions and changing
policies also prevent interpreting between-group variability as estimator variance.

Post hoc scoring of the **old** baseline/final actors gives Gaussian discrepancy
0.39224 → 0.38330 and energy discrepancy 0.48299 → 0.47690. Their units differ;
these numbers do not establish that training with energy distance would be better.
Two reference answers per question and encoder limitations remain unresolved.

## Reproduce and use

```bash
.venv/bin/python experiments/helpsteer-semantic-mmd/audit_kcgm_replay.py \
  --source outputs/kcgm-replay --write-profiles
.venv/bin/python -m unittest discover \
  -s experiments/helpsteer-semantic-mmd -p 'test_*.py' -v
```

The compact source replay is retained locally under `outputs/`, rather than
publishing raw training rollouts in this results directory. Its decompressed
content hash, run configuration hash, and reference hashes bind this audit.

Prepared profiles: [Gaussian](../../../experiments/helpsteer-semantic-mmd/kcgm-rbf-config.json)
and [energy](../../../experiments/helpsteer-semantic-mmd/kcgm-energy-config.json).
Both have `gpu_submission_authorized: false`; staging preserves that setting.
Future runtime preflight verifies the frozen calibration and reference hashes.
At the time of this CPU audit, neither profile had been submitted.

Validation: 82 CPU checks passed across the semantic suite (13), historical DDP
suite (2), HelpSteer suite (15), programming suite (8), and root suite (44).
Checks include exact expected gradients, two-rank synchronization, checkpoint
compatibility, coefficient ratios, and staging without reusing spending approval.

Prior work: [Diamant & Trippe, Calibrating Generative Models to Feature Distributions
with MMD Finetuning](https://arxiv.org/abs/2606.19496),
[author implementation](https://github.com/smithhenryd/cgm).
These are adaptations of existing ideas, not a novelty claim.

## GPU follow-up submission

After the user authorized GPU execution, derived profiles were submitted on
November 21, 2026 at 3:36 PM Pacific as **Slurm job 508718**. At submission it
was pending priority, with no scheduler start estimate. One allocation uses
two H200s for at most two hours total (four GPU-hours), running Gaussian then
energy sequentially. Each variant has a 45-minute policy-stage budget and a
55-minute controller cap including evaluation and saving. Both use the frozen
calibration above, initial-policy KL centering, and required online W&B logging.
They start independently from the same pinned initial actor. A failure in the
first variant stops the job before the second; there are no automatic retries.

Runtime directories on Tillicum are `experiments/helpsteer-kcgm-rbf-v2` and
`experiments/helpsteer-kcgm-energy-v2`. CPU preflight passed for each deployed
runtime, and W&B project authentication was verified before submission.
The replay statistics above remain CPU findings; no new training result is
claimed by this submission record.
