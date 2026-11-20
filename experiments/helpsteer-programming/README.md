# Programming-only word-distribution experiment

Status: **Run 501852 completed successfully in 1h25m38s**, with 75 policy
updates and 1,200 training answers. On 32 held-out questions × four answers,
bucketed KL changed 0.59196 → 0.58803, full-vocabulary KL 1.07295 → 1.06736,
and quality reward 0.66412 → 0.65851. These point estimates do not establish a
reliable improvement. [Recorded results and generations](../../results/helpsteer-v1/README.md).
The historical allocation requested two H200s for at most two hours, with
75 minutes budgeted for its policy stage. The scorer was reused without
additional reward training or a new scorer audit.

`stage_runtime.py` builds the isolated runtime deployed at
`/gpfs/projects/mathai/permuformer2/yash_data/stoch/experiments/helpsteer-programming-v1`.
`policy.sbatch` submits its single policy run through `run_policy.py`.
The prepared `config.json` remains a CPU snapshot; the staged runtime adds the
new explicit authorization and records its own configuration hash in the manifest.
W&B is required online in the `stochastic-llm` project, with group
`helpsteer-programming-unigram-v1`. The W&B connection was verified before submission.

The proposed next policy run keeps the existing actor, frozen quality scorer,
passage-level leave-one-out credit, and reward weights. It changes the prompt
domain and replaces the full word vocabulary with a fixed frequent-word
vocabulary plus OTHER. Both changes are combined here, so this experiment alone
would not distinguish their individual effects.

## Prepared data

| Item | Count |
|---|---:|
| Keyword candidates inspected | 412 |
| Topic-included candidates | 287 |
| Included candidates reserved for existing scorer-audit families | 2 |
| Distinct remaining conversation families | 276 |
| Policy training prompts / reference answers | 228 |
| Policy validation prompts / held-out reference answers | 16 |
| Policy test prompts / held-out reference answers | 32 |
| Words in training references | 61,775 |
| Distinct training word types | 4,400 |
| Retained word categories, occurring at least five times | 1,589 |
| Retained reference probability mass | 91.804% |
| OTHER reference probability mass | 8.196% |

The 10,000-word maximum does not bind: there are fewer than 10,000 types even
before the frequency threshold. The minimum count of five pools the sparse tail.
It is an initial setting, not a demonstrated optimum. The CPU report also records
the vocabulary and coverage obtained at minimum counts 1, 2, 5, and 10.

Sources are the cached pinned HelpSteer2 files and the previously prepared v2
reference corpus. That corpus selects preferred original-training answers with
human coherence, helpfulness, and correctness all at least 3/4. Its existing
prompt and complete-answer length eligibility limits remain in force; this step
does not broaden the experiment to previously excluded long inputs.

`topic_filter.py` searches broadly for programming languages, algorithms, and
development tools. `topic-review.json` records an assistant inclusion/exclusion
decision for every candidate, with a prompt hash and a source-reference hash.
Keyword matching alone never includes an example. The review excludes incidental
mentions, unrelated content requested in HTML, career/marketing writing, generic
role acknowledgements, and unrelated conversation continuations. It is not an
independent human annotation, and the candidate search can miss relevant prompts.

Same and near-identical conversation openings are grouped before splitting.
One representative per family is selected by a seeded ID hash, independently of
answer length. All other continuations in that family are withheld rather than
placed in a different split. Families represented in the existing scorer audit
are reserved as well. The split uses seed 42, with 16 validation and 32 test
families. Every remaining family is used for policy training; the old 128-prompt
subsample limit is removed to align the policy prompt set with the reference set.

The vocabulary and PMF use **only the 228 policy-training reference answers**.
Validation/test answers are held out from both. These are fresh splits within the
original HelpSteer training partition, not the dataset's official validation
partition or previously untouched data across every historical experiment. A
future comparison should initialize from the pinned actor without prior adapters.

## Distribution and reward

Normalize words using the existing lowercase alphabetic-word extractor, retaining
internal apostrophes. Code operators and numerals are ignored, and identifiers
containing underscores are split. This models a word PMF, not a code lexer or the
actor's BPE-token distribution.

Let `c(w)` be training counts and `V` the most frequent at most 10,000 words with
`c(w) >= 5`. Ties are broken lexically. For `T = sum_w c(w)`, define

\[
P(w)=c(w)/T \quad(w\in V),\qquad
P(\mathrm{OTHER})=\sum_{w\notin V}c(w)/T.
\]

Both reference-tail words and entirely novel generated words map to OTHER. The
tail is never discarded, and the head is never renormalized in isolation.

For a batch of complete passages with pooled counts `C`, use additive smoothing
with total mass one, `alpha = 1 / (|V| + 1)`:

\[
\widehat Q_C(k)=\frac{C_k+\alpha}{\sum_j C_j+1},\qquad
K(C)=\sum_{k:P(k)>0}P(k)\log\frac{P(k)}{\widehat Q_C(k)}.
\]

The batch objective remains

\[
\mathbb E\left[\frac1N\sum_i r_{\rm quality}(Y_i)-0.5K(C)\right],
\]

alongside the existing actor-to-initial-policy regularizer with coefficient
0.002. The scorer remains the audited frozen
`Skywork/Skywork-Reward-V2-Qwen3-8B`, revision
`6f19fdefb933293d4898bdb59a96f7223d998659`, using its existing calibration and
0.1 nontermination penalty. This CPU step does not retrain or rescore that model.
The original general-domain audit file is preserved byte-for-byte for its scorer
certificate; no programming-specific audit has been performed, and upstream
scorer training overlap is unknown.

With 16 distinct training prompts per update, aggregate the resulting 16 complete
passages across both ranks. Passage `i` receives distribution advantage

\[
A_i^{\rm distribution}=N\,[K(C-C_i)-K(C)].
\]

Add 0.5 times this to the unchanged quality leave-one-out advantage. The optimizer
averages the 16 passage contributions. There is no word-by-word reward, and the
histogram is recomputed per batch rather than accumulated across the entire run.
Generation remains capped at 1,024 new model tokens. OTHER does not improve
credit assignment inside a passage; the existing passage-level estimator remains.

The v2 runtime's `make_distribution` factory selects this mode from the new
configuration. The existing v2 profile still defaults to its full vocabulary.
Evaluation in the new mode reports both the optimized bucketed KL and a separate
full-vocabulary KL. Lower bucketed KL alone would not demonstrate improvement on
the original full-word objective: differences within OTHER remain unconstrained.

## CPU checks and reproduction

From the repository root, with the existing v2 reference data and cached pinned
actor tokenizer:

```bash
.venv/bin/python experiments/helpsteer-programming/prepare.py
.venv/bin/python -m unittest discover -s experiments/helpsteer-programming -p 'test_*.py' -v
```

Preparation is offline and loads only a tokenizer. The raw preference cache
defaults to `outputs/helpsteer-data/raw`; paths can be overridden with `--raw`,
`--source-data`, and `--out`. Changed source references or unreviewed candidates
cause an error rather than silent inclusion. Rebuilding the upstream v2 corpus
is documented in `../helpsteer-rl-v2/README.md`.

Artifacts in ignored `data/` include the split files, training references,
held-out gold answers, `vocabulary.json`, the complete candidate review inventory,
`cpu-report.json`, and a manifest of hashes and conversation-family memberships.
Held-out gold diagnostics measure corpus coverage; **they are not generated model
results**. The test suite checks mass conservation, novel-word counting, a fixed
vocabulary, deterministic ties, the KL formula, exact expected policy gradients,
split separation, and the saved-data hashes.

Programming remains a broad domain spanning different languages and tasks. Even
human reference batches can differ from the pooled training PMF. This preparation
does not establish that the new target improves generation or removes the
sample-size sensitivity of empirical KL. No new checkpoint or W&B run existed
at submission; the run creates them when it starts.
