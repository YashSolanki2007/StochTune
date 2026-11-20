# Human-preference reward and HelpSteer word-distribution control

This experiment uses pinned NVIDIA HelpSteer2 human preferences to train a
scalar reward model, then compares two fresh policy adapters: quality-only
RL and the same quality reward plus unigram KL. The base instruction model
is the implicit third arm, evaluated with adapters disabled. Neither the
Shakespeare adapter nor its corpus is used as training data.


The subset quality-only and λ=0.1 arms (job 500689) and the λ=0.5 arm
(job 501019) completed 200 updates each. Their measured KL did not improve
relative to the common initial policy. Recorded metrics and generations are in
[results/helpsteer-v1](../../results/helpsteer-v1/README.md). The protocol below
is historical; it does not describe a currently running reward-training job.

## Completed subset checks and historical launch protocol

The two longer queued policy jobs (500683 and 500684) were cancelled at the
user's request. The reward-model job (500682) completed and was reused rather than retrained. The smaller comparison uses
128 policy-training prompts and their 128 preferred PMF responses, eight
validation prompts, 16 test prompts, and 200 updates per arm. Reward-model
training still uses its original 5,095 human-preference pairs.

`prepare_subset.py` creates an isolated deployment from the pinned prepared
data and `subset-config.json`. Both fresh policy arms execute sequentially
inside ONE two-H200 allocation capped at **1h45m**, at most 3.5 GPU-hours.
Each arm has a 40-minute training cutoff to reserve final checkpointing,
independent evaluation, and W&B upload time. Actual counts must be checked.
The larger policy experiment is not automatically queued after this check.

```bash
../../.venv/bin/python prepare_subset.py --output ../../outputs/helpsteer-subset-deploy
```

The earlier larger-pipeline cap of nine GPU-hours means up to five hours of
wall-clock allocations, not nine wall-clock hours. Queue waits are separate.

## Data

Dataset: `nvidia/HelpSteer2`, revision
`990b2711a36180dd19d9c94b8627844866f8982a` (CC-BY-4.0).
Source: https://huggingface.co/datasets/nvidia/HelpSteer2

Canonical conversation hashes group whitespace-equivalent prompts before
splitting. Original training prompts are divided 75% to reward training and
25% to policy training. Original validation prompts supply disjoint reward
validation, policy validation, and policy test sets. Evaluation references
and labels never enter reward training or PMF construction.

The prepared dataset has 5,095 reward-training pairs, 295 reward-validation
pairs, 533 policy-training prompts, 16 policy-validation prompts, and 32
policy-test prompts. The reference PMF counts only the 533 human-preferred
policy-training responses whose coherence, correctness, and helpfulness
ratings are all >=3/4. Tied preferences do not supply a PMF reference.

Both candidate answers must have >=90% ASCII among alphabetic characters.
This is a language-selection heuristic, not a language classifier. Prompts
are at most 384 Qwen tokens; complete prompt/response inputs for reward
training are at most 1,024 tokens; PMF references are at most 256 response
tokens. No source answer is silently truncated. Full multi-turn context is
preserved by parsing HelpSteer's User/Assistant markers. The selection
defines a short-answer subset and limits generalization to the full dataset.

## Learned quality reward

A separate Qwen2.5-3B-Instruct sequence classifier is adapted with attention
LoRA (rank16, alpha32) plus a trainable scalar score head. It is trained for
one shuffled epoch, effective batch32 pairs, learning rate2e-5. Positive
HelpSteer preference strength prefers response TWO; zero denotes a tie.

```math
P_\phi(y_2\succ y_1|c)=\sigma(r_\phi(c,y_2)-r_\phi(c,y_1)),\qquad
\mathcal L=-t\log P_\phi-(1-t)\log(1-P_\phi),
\quad t=\frac{1+\operatorname{sign}(s)}2.
```

Preference strength magnitude is not treated as a probability; only sign
and ties enter this initial likelihood. Model selection uses reward-validation
loss. Policy training requires non-tied reward-validation accuracy >50%,
nonzero score variance, and at least one reward-training update. This is a
technical gate; it does not certify writing quality or statistical significance.

Selected-model validation scores fix center m and scale s. The frozen policy
reward is `sigmoid((raw_score-m)/s)`, multiplied by EOS termination. This
bounded scalar is not a calibrated human win probability. The RM scores
complete prompt/answer inputs without access to human rationales at inference.

## Controlled policy comparison

Both policies start from the same pinned 3B instruction model, seed42, LoRA,
prompts, and schedule: 600 updates, 16 global passages/update across two GPUs,
learning rate5e-6, warmup40, cosine decay, and microbatch2. They use independent
adapters. Sampling uses temperature1, top-p1, top-k0, up to384 new tokens.

```math
J_\lambda=\mathbb E\left[\frac1{16}\sum_i r_Q(c_i,Y_i)
-\lambda D_{\mathrm{KL}}(p_{\mathrm{preferred\ words}}\|\widehat q_\mathcal B)\right]
-0.002\,\mathbb E_cD_{\mathrm{KL}}(\pi_\theta(\cdot|c)\|\pi_0(\cdot|c)),
\qquad \lambda\in\{0,0.1\}.
```

The same word normalization, UNKNOWN category, total smoothing mass1, and
global leave-one-passage-out REINFORCE estimator as the unigram experiment
are reused. Every answer, including an unfinished one, remains in the PMF.
Quality and weighted distribution advantage RMS are logged separately.
Lambda0.1 is a starting setting, not a tuned optimum for the new reward.

Validation uses16 prompts x2 draws; final evaluation32 prompts x2 draws,
with matched per-rank seeds for initial/final policies. A separate pinned
Qwen2.5-7B-Instruct auditor scores coherence, readability, and fulfillment
on final test generations; its output never enters training. It remains
a same-family model judge, not human evaluation. Finite-count KL and length
remain confounded and require count-matched post hoc analysis.

## Execution and budget

`prepare.py` is CPU-only. `reward.sbatch` requests one H200 for at most one
hour. Two sequential `policy.sbatch` jobs each request two H200s for at most
two hours. Total allocation cap:9 GPU-hours, never more than two concurrent
GPUs. There are no extra benchmark or pilot GPU jobs and no automatic requeue.
Each policy has a105-minute training cutoff reserving final evaluation time;
if a cutoff is reached, actual update counts can differ and must be reported.

W&B online logging is mandatory in the existing `stochastic-llm` project,
group `helpsteer-human-preference-v1`. Raw actions, preference rewards,
validation/test generations, auditor scores, checkpoints, selected reward
adapter, and source/data hashes are retained. Environments, weights, raw
dataset downloads, and generated training data are outside Git.

```bash
# Locally, from this folder with the project environment:
../../.venv/bin/python prepare.py
../../.venv/bin/python -m unittest -v test_helpsteer.py
# On Tillicum, from the uploaded folder:
sbatch --account=mathai reward.sbatch
# Submit policy jobs with afterok dependencies on reward then prior policy.
sbatch --account=mathai --dependency=afterok:REWARD_JOB policy.sbatch quality-only
sbatch --account=mathai --dependency=afterok:CONTROL_JOB policy.sbatch quality-unigram
```
