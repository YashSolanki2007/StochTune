# LitBench fantasy writing: learned preferences + word-frequency KL

**Recorded outcome:** job 519547 was cancelled after eight minutes on two H200s,
before RL. Initial validation reporting failed because NumPy boolean checks were
not JSON-serializable. No usable literary reward or policy result was exported.
The source now converts checks to native booleans and tests report serialization;
the cancelled runtime was not restarted. See the [attempt record](../../results/litbench-v1/README.md).
The following describes the intended pipeline, not completed training results.

This experiment returns to the original corpus word-frequency objective in a
creative-writing domain. It trains a new literary Bradley–Terry reward adapter,
then freezes it and trains a separate, fresh Qwen2.5-3B-Instruct policy adapter.
It does not reuse the HelpSteer/Skywork reward or the semantic MMD/energy objective.

The source is [LitBench-Train](https://huggingface.co/datasets/SAA-Lab/LitBench-Train),
pinned in `config.json`. LitBench's chosen/rejected labels are curated implicit
Reddit upvote preferences, not expert coherence ratings. Full attribution and
limitations are described in the [LitBench paper](https://arxiv.org/abs/2507.00769)
and [published version](https://aclanthology.org/2026.eacl-long.362/).
The public [official test](https://huggingface.co/datasets/SAA-Lab/LitBench-Test)
contains comment IDs rather than full stories; we do not use or reconstruct it.
All evaluation here is internal held-out evaluation, not the official benchmark.
Underlying stories remain the work of their original authors.

## Data and held-out checks

Before filtering, union-find groups canonical prompts and shared story texts.
Components are deterministically split into reward training, model selection,
reward audit, policy training, policy validation and policy test. Components
cannot cross partitions. Reward selection and audit use one pair per component;
their sampling intentionally includes length-matched pairs. This is a restricted
evaluation population, not an estimate of accuracy on the entire dataset.

Both stories must have 150–650 words and fit the scorer's 3,072-token context
without truncation. The scorer uses broad short-fiction pairs; the policy uses
only prompts matching an explicit fantasy keyword filter in `prepare.py`.
The target word PMF counts unique chosen fantasy stories from training
partitions only. Retain up to 10,000 words occurring at least five times and
aggregate all remaining words into OTHER, preserving their total probability.

Reward training minimizes

\[
 L_{BT}=-\log\sigma(s_\phi(q,y^+)-s_\phi(q,y^-)).
\]

LoRA rank 16 / alpha 32 trains attention projections and the saved scalar score
head, with AdamW learning rate 2e-5 and global batch 32 on two DDP ranks.
Model selection minimizes held-out pairwise loss. A separate audit must pass
all declared checks: at least 128 independent components, accuracy >=55%, a
bootstrap lower confidence bound above chance, a positive paired lower bound
over a longer-answer baseline, and >=55% accuracy on at least 50 length-matched
pairs. At least 25 training updates must have completed. Failure stops the job
before policy training; a failed scorer is never silently substituted.

## Policy objective and credit

For a batch of generated stories, optimize

\[
 J=\mathbb E\left[\frac1N\sum_i r_\phi(q_i,Y_i)
       -0.1 D_{KL}(P_{words}\|\widehat Q_{Y_{1:N}})\right]
       -0.002\,\mathbb E_q D_{KL}(\pi_\theta(\cdot|q)\|\pi_0(\cdot|q)).
\]

The two KL terms are different: target-to-generated **word PMF** fidelity and
policy-to-base **sequence distribution** regularization. The empirical generated
PMF has smoothing mass 1.0. The scalar scorer is transformed by a sigmoid using
center/scale from the model-selection partition; this is not calibrated human
preference probability. Unterminated stories incur a separate 0.1 penalty.

On-policy REINFORCE uses whole-story leave-one-out quality and coupled word-PMF
credit: \(A_i=A_i^{quality}+0.1N(D_{-i}-D_{full})\). Distribution statistics are
pooled across the full batch, not recomputed as a reward after every word.
Base-policy KL credit uses a same-prompt leave-one-out baseline. The batch has
four distinct prompts and four independently sampled stories per prompt.
Policy LoRA uses learning rate 5e-6, generation temperature 1, no top-k/top-p
truncation and a maximum 1,024 new tokens. Lambda 0.1 is an initial experimental
setting, not a claim of optimality or calibrated gradient balance.

## One bounded allocation

The user approved **one job, two H200s, maximum two hours total (4 GPU-hours)**,
including reward training, policy training, evaluations and logging. Reward
updates have a 30-minute deadline; reward validation/export must finish within
45 minutes. RL gets at most 60 minutes and stops by allocation minute 100,
leaving time for final evaluation. Hard Slurm wall time is two hours; there are
no automatic restarts or additional GPU jobs.

Both stages require online W&B in the configured `stochastic-llm` project.
Artifacts include source/data hashes, reward checks, calibration and adapters,
policy rollout logs, word KL, quality, truncation, individual credit RMS values,
and base-policy KL. Baseline and final evaluation use identical held-out prompts,
sample counts, decoding and RNG seeds. A separate frozen Qwen2.5-7B fiction
rubric judge scores coherence, readability and premise fulfillment; it is not
human ground truth or the training reward. Longer outputs and scorer rewards
alone do not establish an improvement.

Run CPU preparation and tests locally, build a fresh isolated runtime with
`stage_runtime.py`, run `preflight.py` on the cluster login node, then submit
`policy.sbatch` once. `run.py` stops if a stale reward certificate exists.
The runtime deployment manifest records exact source/data bytes.

This is an exploratory single-seed budgeted experiment, not a reproduction of
LitBench's full-finetuning protocol or evidence of a new general method.

Related implementation lineage: [controlled stochasticity blog](https://www.lesswrong.com/posts/upCrcE39aF63GoLJh/training-language-models-for-controlled-stochasticity-2),
[InstructGPT / Bradley–Terry preference rewards](https://arxiv.org/abs/2203.02155).
