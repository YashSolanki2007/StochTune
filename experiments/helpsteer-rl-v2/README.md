# Corrected HelpSteer experiment: completed shared-budget run

## Recorded results and historical shared-budget protocol

The user subsequently authorized **at most two GPUs for at most two hours**.
The launcher for this recorded run is `budget.sbatch`: ONE two-H200/two-hour allocation
including all scorer validation and the one proposed policy run, maximum4
GPU-hours. No other standalone launchers are submitted by this controller.

It first audits the existing3B scorer and the pretrained
[Skywork-Reward-V2-Qwen3-8B](https://huggingface.co/Skywork/Skywork-Reward-V2-Qwen3-8B)
concurrently, one GPU each. The latter is a Bradley-Terry reward model trained
on human/AI preference data, pinned to revision
`6f19fdefb933293d4898bdb59a96f7223d998659`. Its weights are downloaded and
tokenization checked before allocation. GPU processes run offline. Candidate
validation re-encodes the raw conversations using each model's own tokenizer.
Upstream overlap with HelpSteer cannot be ruled out; this audit is not a clean
external benchmark claim.

Both candidates must meet the same predeclared validation gates below. Among
passing models, the controller selects the higher human-preference accuracy,
breaking ties in favor of the smaller legacy scorer. Each writes a separate
certificate; no race or relaxed threshold creates policy eligibility. If both
fail, the allocation exits early. No scorer training is performed.

One quality+unigram policy then targets200 updates using the selected frozen
reward, lambda0.5, and the corrections below. There is no quality-only control
or automatic follow-up job. A shared deadline, policy training cutoff, process
group cleanup, Slurm hard limit, and no-requeue setting bound total compute.
Final evaluation/checkpoint time is reserved. W&B logs both audits and any policy.
The older ten-minute audit-only proposal was superseded by this authorization.

This separate directory preserves the completed v1 source and runs. It addresses
length handling, a narrow reference PMF, correlated prompt batches, and inadequate
reward validation. Slurm job **501413** was submitted for the authorized shared
budget. It validates two scorer candidates, then runs one corrected quality+KL
policy only if a scorer passes. The allocation completed successfully in 1h21m01s. The selected Skywork scorer
passed its audit, and the policy completed 86 updates. Full-vocabulary KL changed
1.9163 → 1.9923 while quality reward changed 0.5432 → 0.5674; this does not
establish joint quality/distribution improvement. See the
[recorded results](../../results/helpsteer-v1/README.md).
There is no additional policy submission or reward retraining.

## Implemented corrections

| Setting | Previous subset | This profile |
|---|---:|---:|
| Generated-token ceiling | 384 | 1,024 |
| Total reward input ceiling | 1,024 | 2,048 |
| Independent auditor input ceiling | 2,048 | 4,096 |
| Global distribution batch | 2 prompts x8 answers | 16 prompts x1 answer |
| Reference answers | 128 | 4,948 |
| Reference word occurrences | 13,194 | 1,202,701 |
| Policy-training prompts | 128 | 128, length-stratified |
| Treatment of cutoff answers | Entire quality score set to0 | Keep quality; subtract a separate penalty |

The target uses all eligible human-preferred original-training answers with
human coherence, helpfulness, and correctness ratings >=3/4. There is no
256-token response filter. Source/model revisions are unchanged and pinned in
`config.json`. Reward-training prompts may supply PMF reference text; no reward-
validation or policy-evaluation prompt supplies reference text. This is a broad
mixed-domain marginal, not a guarantee that every answer should use the same words.

Policy validation and test retain the original 16/32 held-out prompt IDs, preventing
the old reward model's model-selection prompts from silently becoming policy tests.
The corrected reward dataset has 5,219 training pairs and 372 validation pairs.
Validation includes64 original-training conversations exceeding1,024 tokens,
reserved from BOTH reward training and the PMF. The old reward model did not see
these inputs during training. No source text is silently truncated.

The policy quality term is now

```math
r_Q(c,Y)=\sigma((r_\phi(c,Y)-m)/s)-\kappa\mathbf1_{\mathrm{generation\ cap}},
\qquad \kappa=0.1.
```

Both checkpoint EOS tokens are honored when stopping and recording termination.
Neither alternate EOS explains the recorded v1 test truncations (checked from
saved token IDs). The cap penalty is an explicit starting choice, not a validated optimum. No
clipping or dropping erases the content score. Raw scores, bounded quality before
penalty, cutoff penalty, and final quality are logged separately. The scorer sees
the full visible answer. This still cannot recover the unseen continuation.

The policy objective remains quality minus **0.5 x full-support unigram forward
KL**, with a separate0.002 initial-policy sequence-KL penalty. DDP and passage-level
leave-one-out credit are unchanged. The optional prepared policy job has200
updates, warmup20, batch16, microbatch1, temperature1, top-p1, and top-k0.
Smaller microbatches accommodate longer sequences. The recorded allocation runtime includes both concurrent scorer audits and the
policy stage; no separate GPU benchmark was submitted.

## Stronger scorer validation

`audit_reward.py` loads the existing adapter, freezes it, and scores human-labeled
pairs without training any weights. Every candidate's raw score is saved.
`validation.py` reports overall preference accuracy, training-majority and
longer-answer baselines, response-length-matched accuracy, human-coherence
direction accuracy, and accuracy on >1,024-token inputs.

The new policy eligibility checks require:

- At least100 non-tied preference pairs.
- Positive lower95% paired-bootstrap improvement over BOTH simple baselines.
- At least30 length-matched pairs with lower95% accuracy bound above50%.
- At least30 human-coherence-different pairs with lower95% bound above50%.
- At least20 long-input pairs with lower95% accuracy bound above50%.

These are conservative engineering gates, not proof of human-perceived quality.
The intervals are exploratory validation statistics; selecting a model on the same
validation data prevents treating them as independent final-test evidence.
Overall preferences and coherence can conflict because correctness and usefulness
also matter. Individual predictions are retained to investigate such failures.

The previous `passed: true` certificate is rejected. A passed certificate must
use protocol2 and match the validation-data and adapter checksums. A failed audit
does not create a ready marker. `preflight.py` checks this BEFORE requesting a
policy allocation; the policy entry point repeats the checks. A failed audit
does not automatically launch another model, retraining, or an RL job.

## CPU verification and measured data coverage

```bash
python experiments/helpsteer-rl-v2/prepare.py --raw outputs/helpsteer-data/raw
python experiments/helpsteer-rl-v2/diagnose_data.py --raw outputs/helpsteer-data/raw
python -m unittest discover -s experiments/helpsteer-rl-v2 -p test_helpsteer.py -v
```

[DATA_AUDIT.json](DATA_AUDIT.json) records a CPU-only comparison. On the same32
human-preferred test answers, out-of-reference word mass falls **19.41% -> 2.57%**.
This is reference coverage, not evidence of improved model generation. The target
now has30,078 observed word types. Of4,948 reference answers,34 exceed the proposed
1,024-output-token ceiling; the median/p90/p99 are292/572/971 tokens.

The human validation set contains280 non-tied preferences,73 length-matched
non-tied pairs,146 coherence-different pairs, and77 long-input pairs including
ties. Counts used in each gate are recorded separately by the model audit.
The new longer-answer baseline is63.93%, with training-majority baseline56.79%.
Skywork non-tied preference accuracy was 78.75%; the legacy scorer reached
67.14%. Only Skywork passed all declared gates; these are model-selection
audit results, with possible upstream HelpSteer overlap.

15 CPU tests passed, covering preserved truncated-answer quality, both EOS tokens, explicit context
limits, distinct-prompt assignment, reference/evaluation separation, rejection
of length-only scorers and old certificates, gradient direction, DDP credit,
and real tiny-model reward-head export. They do not replace a scheduled model audit.

## Distribution evaluation

The primary loss remains full-vocabulary KL. Saved evaluation includes missing
reference mass, its KL contribution, and smoothed Jensen-Shannon divergence.
Common1,000 words plus OTHER JS is diagnostic only. Rare words remain in the
primary objective; no favorable metric replaces it silently.

Final evaluation uses32 held-out prompts x4 generations =128 responses per policy,
with matching initial/final seeds. Responses can finish naturally up to1,024 tokens.
This increases coverage; post hoc comparisons still need matched word counts and
prompt-level uncertainty. A separate7B auditor supplies coherence/readability/
fulfillment scores and never supplies training reward. Public-data pretraining,
mixed domains, one seed, finite-batch bias, and lack of human generation ratings
remain limitations.

## Completed submission and budget

Job **501413** uses the isolated cluster directory
`/gpfs/projects/mathai/permuformer2/yash_data/stoch/experiments/helpsteer-budget-v2`.
All data hashes, cached candidate weights, tokenizer lengths, scalar score-head
shape, and the Slurm resource cap were checked before submission. Fifteen CPU
checks passed. The maximum is two H200s for two hours TOTAL across all phases,
not two hours per phase. Downloads finished before allocation.

```bash
# The ONE job submitted under the approved shared budget:
sbatch --account=mathai budget.sbatch
```

The standalone audit/policy scripts remain reproduction templates. They were
not submitted in addition to the shared job. Both scorer audits have a ten-minute
stage cutoff and run concurrently. A selected scorer must pass protocol2 before
one quality+unigram run can start. If neither passes, the allocation exits early.
A shared deadline reserves saving/evaluation time. Actual update count may be
less than200 if the time budget binds. There is no quality-only control, retry,
requeue, extra allocation, or follow-up job. W&B logging is mandatory throughout.
