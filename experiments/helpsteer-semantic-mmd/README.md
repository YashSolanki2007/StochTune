# Prompt-conditioned semantic MMD

Slurm job **504124** was submitted on November 20, 2026. At the last check
(November 20, 1:44 PM Pacific), it remained pending priority. Scheduler estimates
are transient; this document records no MMD policy-training result.
The isolated runtime is deployed at
`/gpfs/projects/mathai/permuformer2/yash_data/stoch/experiments/helpsteer-semantic-mmd-v1`.
The deployed CPU preflight, encoder parity, and W&B connection checks passed.
All 30 relevant CPU tests passed (7 MMD tests plus 23 existing regression checks).

This experiment replaces the word-PMF reward with answer-embedding MMD computed
separately for each exact question. It retains the frozen audited Skywork quality
scorer and the pinned Qwen2.5-3B-Instruct actor. No teacher is trained or sampled.
The user requested this experiment after discussing semantic distribution matching;
the request was subsequently reduced to two H200s for 90 minutes (three GPU-hours).
The 75-minute policy-stage budget is retained; 15 minutes remain for final evaluation
and saving. The previous comparable run finished in 85 minutes 38 seconds total.

## Reference data

Use the existing programming splits without moving any questions between them.
Keep only questions with two distinct original HelpSteer2 answers whose human
coherence, helpfulness, and correctness ratings are all at least 3/4:

| Split | Questions | Reference answers |
|---|---:|---:|
| Train | 102 | 204 |
| Validation | 10 | 20 |
| Test | 19 | 38 |

These are **model-written, human-rated** answers, not human-authored answers or
known samples from a human response distribution. Each question's target gives
its two reference prototypes equal weight. The experiment tests matching this
fixed empirical target; it cannot establish coverage of all valid semantic modes
or their real-world frequencies.

Training uses four distinct questions with four independent sampled answers each,
for 16 generated answers per update across two ranks. Quality leave-one-out
baselines also stay within each question. A question's reference answers never
serve as targets for another question. Held-out reference vectors are available
only to evaluation, with an explicit training-ID whitelist in policy credit.

## Representation and kernel

The frozen encoder is `sentence-transformers/all-MiniLM-L6-v2`, revision
`1110a243fdf4706b3f48f1d95db1a4f5529b4d41`. It is a small sentence/paragraph
encoder, not an independently validated judge of code correctness.

Tokenize the entire answer, split into nonoverlapping chunks of at most 254
content tokens plus special tokens, and encode every chunk. Use attention-masked
mean pooling and unit normalization per chunk. Combine chunk vectors with weights
equal to their content-token counts and normalize the combined vector. This is
a documented long-answer extension of the short-paragraph encoder, with no text
tail discarded. It may blur local semantics or code details.

For unit answer vectors, use the average of three Gaussian kernels:

\[
k(x,y)=\frac13\sum_{s\in\{0.5,1,2\}}
\exp\left[-\frac{\|x-y\|^2}{2s^2b}\right].
\]

The squared bandwidth `b = 0.2786315083503723` is the median positive
within-question squared distance between the two **training** reference vectors.
Validation and test vectors do not calibrate the kernel.

The CPU corruption checks prefer a balanced collection of both reference
prototypes over repeated copies of one prototype and over a wrong-question
collection, for all 102 training questions. These are engineering checks, not
independent proof that the encoder measures meaningful semantic diversity.

## Reward and credit

For generated vectors `Y_1,...,Y_n` and the two fixed references `X_1,X_2`, use

\[
\widehat d_q=
\frac{1}{n(n-1)}\sum_{i\ne j}k(Y_i,Y_j)
+\frac14\sum_{a,b=1}^2 k(X_a,X_b)
-\frac1n\sum_{i=1}^n\sum_{a=1}^2 k(Y_i,X_a).
\]

The reference term includes its diagonal because the target is the fixed,
equal-weight empirical reference measure. The generated term excludes its
diagonal. Under independent conditional generation, this is unbiased for
population MMD² against that empirical target. It is not an unbiased estimate
of distance to the unknown population of all acceptable answers. A finite
U-statistic can be negative: **do not clip it**. We also log the nonnegative,
diagonal-including V-statistic as a separate diagnostic.

The batch reward averages equally across its four questions:

\[
R=\frac14\sum_q\left[\frac14\sum_{i=1}^4 r_{\rm quality}(Y_{qi})
-0.5\,\widehat d_q\right].
\]

For answer `i` to question `q`, distribution credit is

\[
A^{\rm MMD}_{qi}=4\,[\widehat d_q(Y_{q,-i})-\widehat d_q(Y_q)].
\]

The three remaining answers suffice for the leave-one-out U-statistic. Add 0.5
times this credit to the within-question quality advantage, then average the
16 passage policy gradients using the existing DDP updater. Retain initial-actor
KL regularization with coefficient 0.002, learning rate 5e-6, LoRA rank 16, and
the existing 1,024-token generation cap. The coefficient 0.5 is an initial setting;
it has different units from 0.5 times word KL and is not known to be optimal.

Word-level KL against the previous training-only programming PMF is retained
**only as an evaluation diagnostic**. Semantic MMD is the actual distribution
reward; this is not a quality-only run. Nearest-reference prototype coverage is
also logged, with no claim that the two prototypes are true discrete modes.

## Run and validation

Prepare from cached pinned data and encoder files using CPU:

```bash
.venv/bin/python experiments/helpsteer-semantic-mmd/prepare.py
.venv/bin/python -m unittest discover -s experiments/helpsteer-semantic-mmd -p 'test_mmd.py' -v
```

Tests verify the population MMD expectation, exact expected policy gradients,
collapse and mean-only counterexamples, question-specific baselines, held-out
reference rejection, full-text chunk coverage, and cross-rank credit gathering.
The staged runtime reuses the existing scorer certificate byte-for-byte and
verifies data/config hashes and training-only kernel calibration before submission.

One policy run is submitted through `policy.sbatch` and `run_policy.py`, with
75 minutes budgeted for its policy stage and a hard 90-minute Slurm wall limit.
There is no separate reward-training, teacher-generation, or GPU pilot job.
All metrics, source hashes, reference vectors, sampled answers, and final
comparison files are logged online to W&B in group
`helpsteer-conditional-semantic-mmd-v1`.

Evaluation compares the initialized actor and final actor on the same 19 questions,
four samples each. It reports conditional MMD U/V, prototype coverage, frozen
quality reward, independent coherence/readability/fulfillment, truncation, length,
and the unchanged word-PMF diagnostics. The 7B auditor is independent of the
optimized MiniLM embedding and Skywork reward. Generated answers lack human
evaluation. No quality-only control or second training seed is included in this
budget, and several design choices change from the previous unigram experiment.
Consequently cross-run differences cannot isolate MMD's causal effect.

References: [MiniLM model card](https://huggingface.co/sentence-transformers/all-MiniLM-L6-v2),
[Gretton et al., A Kernel Two-Sample Test](https://jmlr.org/papers/v13/gretton12a.html).
