# Reward-weight grid search — implemented, awaiting GPU approval

This is a CPU-only preparation/selection workflow. **No sweep jobs have been
submitted.** It has no scheduler submission command. `run.py` continues to require
an explicitly authorized deployed configuration before any GPU training starts.
Approval from earlier experiments is never copied into a new trial or promotion.

## Frozen experiment

Nine fresh Qwen2.5-3B-Instruct actors use the Cartesian product:

| Setting | Values |
|---|---|
| Word-KL weight | 0.1, 0.3, 1.0 |
| Length weight per 100 words outside 250–350 | 0, 0.2, 0.4 |
| Quality weight | 1, implicit in the unchanged reward implementation |
| Reference-policy KL weight | 0.002 |
| Global batch | Four prompts × four answers = 16 |
| GPUs per trial | Four H200s; four DDP ranks throughout promotions |
| Optimizer | Existing AdamW, LR 5e-6, LoRA rank 16 / alpha 32 |
| Schedule | 1,000-step cosine schedule, 50 warmup steps, from the outset |
| Seed | 42 for initial screening; further seeds require a separate experiment |
| Validation | Same 16 prompts × four answers per probe; expanded from eight prompts |
| Checkpoint / validation intervals | 25 / 50 steps |
| Rungs | 150 → 400 → 1,000 cumulative updates |

The target word PMF, judge pins, generation protocol, EOS penalty and credit
calculation stay fixed. The default control is `kl010-len020` (word KL 0.1,
length 0.2). Keep it at every rung even if an early validation probe is weak.
The population progresses **nine → three → two**: control plus the best two
other configurations, then control plus the best remaining alternative.

Every rung uses validation for initial, periodic and final probes. No test
answers are generated or used for selection. The plan freezes target/validation
data hashes and the training/reward implementation hashes. Changing training
code between rungs requires a new plan; formatting a result or editing a source
runtime after ranking invalidates its decision hashes.
The installed `stochastic_llm.rl` likelihood implementation is checked as well;
deploy/install the matching repository source before cluster preflight. The Slurm
launcher finds the ancestor containing `.venv-h200`, including nested grid folders,
or accepts an explicit `STOCH_PROJECT_DIR`. The frozen launcher-body hash excludes
allocation-specific `#SBATCH` lines; the complete staged script is separately
bound by the deployment manifest.

## Common selection rule

The defaults below are **proposed criteria to review before approval**, not
claims about which policy should ultimately be preferred. They can be changed
when preparing a new plan, but the resulting plan is then frozen for that search.

1. Raw judge quality must be at least the identical initial validation baseline
   minus **0.01** (one absolute point on the normalized 0–1 scale).
2. At least **70%** of generated answers must be in the 250–350-word band.
3. Among feasible trials, minimize **raw word KL**. Break exact ties by higher
   raw quality, higher compliance, then deterministic trial ID.

Do not compare `joint_reward` between trials with different weights. Ranking
recomputes raw quality, word counts, compliance and word KL from the saved
validation samples. It checks all 64 unique sample IDs, the exact prompt set,
the completed rung, deployed/logged configuration agreement and identical base
generations across candidates. Incomplete/missing trials or test-split results
are rejected, not silently dropped.

Infeasible trials rank after feasible trials by the fixed violation score

```math
v=\frac{\max(0,Q_{\mathrm{base}}-0.01-Q)}{0.01}
  +\frac{\max(0,0.70-C)}{0.10}.
```

The numerator thresholds follow any explicitly changed plan defaults; the two
denominator units remain fixed. Raw KL breaks violation-score ties. If too few
trials are feasible, remaining promotion slots may go to the least violating
trials; reports explicitly list `selected_infeasible`. If no final trial meets
both constraints, the report returns **no feasible winner**. It never relaxes
the thresholds automatically. These are noisy point-estimate screening rules;
finalists still need additional seeds and independent quality evaluation.

## Resource proposals, not authorization

| Stage | Trials | Additional updates per trial | Wall cap each | Maximum GPU-hours for stage |
|---|---:|---:|---:|---:|
| Initial | 9 | 150 | 90 minutes | 54 |
| Intermediate | 3 | 250 | 2 hours | 24 |
| Finalists | 2 | 600 | 5 hours | 40 |
| Total | 14 allocations | 3,300 aggregate updates | — | **118** |

These are requested ceilings, including judge calibration, model loading,
checkpointing and evaluation. They are deliberately above the rough **13-hour
four-GPU training-only estimate** extrapolated from the last 500-step run;
expanded validation and repeated setup add overhead, and new weights can change
generation length/speed. Stage one reserves 20 minutes for final reporting,
stage two 20 minutes, and finalists 30 minutes. Deadlines can stop a trial before
its rung; such a trial cannot be ranked/promoted. There are no automatic retries.
Confirmation seeds are neither staged nor included in these ceilings.

## CPU-only commands

Prepare locally from the repository root, after reconstructing the existing Emma
data and installing the project environment. The existing `.venv` already has
the required dependencies; no HPO service or new library is needed.

```bash
.venv/bin/python experiments/emma-word-kl/grid_search.py prepare \
  --output outputs/emma-grid-v1 --stage \
  --quality-tolerance 0.01 --compliance-floor 0.70
```

This writes a frozen `grid-search-plan.json`, nine unapproved configs and nine
isolated runtimes. `preparation.json` records zero jobs submitted and the stage
ceiling. Keep these outputs outside Git; the reviewed plan/config copies are
archived in `results/emma-v1/grid-search-preparation-v1/`.

After approval and actual completion of stage one, create an immutable JSON
index mapping **all nine trial IDs to their actual policy output directories**,
using absolute paths on the machine holding the checkpoints. For example one
entry is `"kl010-len020": "/.../kl010-len020-rung150/outputs/quality-unigram-JOBID"`.
The file must contain every required trial, not just that example entry.
Each runtime's original `run-config.json` and artifacts must remain intact.

```bash
python grid_search.py rank \
  --plan /path/to/grid-search-plan.json --index /path/to/index150.json \
  --rung 150 --output /path/to/decision150.json

python grid_search.py promote \
  --plan /path/to/grid-search-plan.json --decision /path/to/decision150.json \
  --output /path/to/grid-rung400 --stage
```

Run those commands from a staged runtime, or use the repository script's full
path. On Tillicum use the existing project `.venv-h200/bin/python`. They perform
only CPU reads/preparation: **promotion does not submit or authorize jobs**.

For the next rank, supply only the three selected outputs in `index400.json`:

```bash
python grid_search.py rank \
  --plan /path/to/grid-search-plan.json --index /path/to/index400.json \
  --rung 400 --previous-decision /path/to/decision150.json \
  --output /path/to/decision400.json

python grid_search.py promote \
  --plan /path/to/grid-search-plan.json --decision /path/to/decision400.json \
  --output /path/to/grid-rung1000 --stage

python grid_search.py rank \
  --plan /path/to/grid-search-plan.json --index /path/to/index1000.json \
  --rung 1000 --previous-decision /path/to/decision400.json \
  --output /path/to/decision1000.json
```

The final rank consumes only the two finalists and prepares no further stage.
Do not overwrite earlier indexes/decisions. Reports bind their input hashes and
absolute artifact paths; run ranking/promotion where the original artifacts are
available. Copying a report alone to another machine is insufficient to promote.

Promotions bind the source configuration, adapter weights, optimizer, scheduler,
per-rank RNG and prompt RNG checkpoint files. The same four-rank world size and
1,000-step schedule are required. Reward weights, data and model settings cannot
change during promotion. New configs reset all GPU spending approval fields.

All training stages require online W&B in the existing `stochastic-llm` project,
with the plan digest, trial ID and rung included in the run config/source artifact.
The 150-step screen is deliberately inexpensive and can misrank late learners;
retain the control and treat this as a resource-limited search, not proof of a
global optimum. Seed confirmation, independent quality judging and a later
frozen test protocol remain separate follow-up work.
