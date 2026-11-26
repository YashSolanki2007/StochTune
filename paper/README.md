# Paper source

**Learning to Match Output Distributions with Batch-Level Rewards: Numerical
Results, Human Preferences, and Conditional Semantic MMD**

Yash Solanki, University of Washington. Revised November 22, 2026.

This report now includes all completed numeric, Shakespeare unigram/bigram,
HelpSteer preference/word-target, conditional semantic-MMD, calibrated
Gaussian/energy, energy-continuation, and distribution-only experiments.
It includes scorer-selection audits, achieved budgets, paired question-bootstrap
uncertainty, and post hoc word- and character-frequency diagnostics.

The later text experiments do not establish reliable joint quality/distribution
gains. All 19-question semantic/quality mean-change intervals include zero.
One training seed, fixed small reference sets and no new human ratings limit
the conclusions. Larger frozen-teacher reference sets, conditional-JSD training,
and full-model direct supervised soft-KL remain unrun. No new GPU experiment
was started to prepare this document.

## Compile

Run `bash build.sh` in this directory. It uses Tectonic or latexmk and writes
`controlled-stochasticity.pdf`. On Overleaf, upload the source ZIP and select
`main.tex` as the main document, using pdfLaTeX. The tables, PDF figures, `.bib`,
and compiled `.bbl` are included; compilation needs no experiment data, shell
escape, credentials, or custom document class.

## Regenerate tables and figures

From the full repository root, after installing the `analysis` extra:

```bash
python paper/generate_assets.py
python paper/generate_new_results.py
```

The numeric script requires records in `results/sft-v1` and `results/rl-v1` and
the installed `stochastic_llm` package. The new-results script reads completed
text records, saved embeddings and the local character-audit report under
`outputs/mmd-diagnostic-character-audit/`. It verifies source hashes and computes
paired question-bootstrap intervals. Neither script trains a model or creates W&B runs.
The scripts in the standalone source ZIP are provided for provenance;
regenerating assets requires the full repository checkout.

`supplementary/per-prompt-results.csv` contains all 100 test prompts' parameters
and recorded base/SFT/RL metrics. Base histogram KL was not part of the recorded
post-hoc RL comparison and is left blank. Null base distances reflect prompts
with zero valid responses. `numerical-summary.json` contains the aggregate
comparison and bootstrap intervals. `supplementary/source-records.json` lists
recorded source hashes used to assemble this manuscript.
The supplementary files also include all 261 training/probe history entries,
the completed SFT/RL metrics, and the recorded training and decoding configs.
`helpsteer.tex` preserves the original larger protocol; actual subset overrides
and completed trials appear in `new_results.tex`. Its source config, compact dataset
manifest, and implementation hashes are captured in
`supplementary/helpsteer-protocol.json`. The historical preparation snapshot is not a current performance report.
`mmd.tex` contains the MMD mathematics. `supplementary/mmd-protocol.json` captures
its config, compact manifest, encoder metadata, CPU report, and source hashes.
The pure mathematical estimator is included as
`supplementary/mmd-estimator-snapshot.py`; it does not load or train a model.
The larger frozen-teacher reference experiment remains a proposal. The current
implementation instead uses two existing model-written, human-rated references
for each retained exact question and generates no new teacher answers.

The source ZIP is suitable for editing or adapting to a submission format; the
author should select any hosting or submission license and venue requirements.

Completed comparison JSONs, reward audits, configs, implementation/W&B records,
and the local character-frequency report are included under
`supplementary/new-results/`. Its `source-records.json` lists SHA-256 hashes.
`semantic-question-statistics.csv` contains all question-level before/after
estimates. `analysis-summary.json` records 10,000 paired bootstrap resamples
with seed 20261122. These intervals condition on the saved generations and
references and exclude training-seed uncertainty. Raw generations and model
weights are not duplicated in the source ZIP; sample hashes point to repository
records. The character-audit script is included as a snapshot and its report
hashes are checked against the saved samples.

The energy 200-update result resumes update 53 (147 new updates), not an
independent trial. Completed text records extend through commit `deb5c8e`,
with the local character audit separately hashed. kCGM-inspired estimator and
calibration components are credited to prior work and the author implementation.
