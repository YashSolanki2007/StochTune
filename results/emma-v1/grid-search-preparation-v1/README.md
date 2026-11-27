# Reward-weight grid preparation — not run

Code and proposed configurations prepared November 27, 2026. **No GPU jobs
have been submitted or authorized.** Execution waits for the user's approval.
The JSON files here are configuration/provenance records, not model results.

Nine initial trials cover word-KL weights 0.1/0.3/1.0 and length weights
0/0.2/0.4. Quality weight is 1 and policy KL weight 0.002. All use a fresh
actor, 1,000-step LR horizon, 50 warmup steps, four GPUs and global batch 16.
The initial allocation stops at 150 updates. Planned promotions retain the
current control and progress nine → three at 400 → two at 1,000, using bound
adapter/optimizer/scheduler/RNG checkpoints without changing reward settings.

Selection uses raw KL on the same sixteen validation prompts × four answers,
subject to baseline-relative raw quality and length-compliance thresholds.
Proposed defaults: quality may fall at most 0.01 from the initial validation
baseline; compliance must reach 70%. These can be revised in a new plan before
approval. The common constraints and fallback rule are frozen in the plan.
No test generations enter selection. A final feasible winner need not exist.

All **49 local CPU tests passed**, including all three rungs using synthetic
fixtures; all nine staged profiles passed CPU preflight. No test loads a large
model or requests a GPU allocation. See cpu-validation.json for coverage and
source hashes, and all-trials-cpu-preflight.json for individual profile checks.

Proposed maximum allocations: initial stage 54 GPU-hours, intermediate stage
24, finalist stage 40; full conditional search ceiling **118 GPU-hours**. These
are upper limits including setup/evaluation, not predicted usage or spending
approval. Additional-seed confirmation is separate and not automatically staged.

See [workflow, commands and resource table](../../../experiments/emma-word-kl/GRID_SEARCH.md).
Regenerate fresh isolated runtimes from the committed code with the documented
prepare command; the generated outputs/ directory is intentionally not committed.
