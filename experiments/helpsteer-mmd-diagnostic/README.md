# Fixed-question distribution-only fitting diagnostic

This diagnostic asks whether on-policy energy-distance training can fit a small
fixed set of **training** questions. It removes the competing quality reward and
increases sampling within each conditional distribution. It does not evaluate
writing quality or reproduce the full kCGM paper.

The user approved **two H200s for at most 90 minutes total**, including evaluation,
on November 22, 2026: at most three GPU-hours. One job only, no retries or extra
allocations. The policy stage has a 70-minute deadline; 20 minutes are reserved
for final evaluation, checkpointing, and W&B upload. There is no scorer training,
reward-model inference, teacher sampling, or separate GPU benchmark.

**Slurm job 510187** was submitted on November 22, 2026 at 1:32 AM Pacific.
At submission it was pending priority without a scheduler start estimate.
Deployed CPU preflight, runtime imports, and W&B authentication passed; all
16 relevant CPU tests passed (3 diagnostic checks plus 13 semantic checks).
The job subsequently completed 17 updates / 1,088 answers in 1h18m37s. Seen
energy distance changed 0.5261 → 0.5061; unseen changed 0.5879 → 0.5969. This
is a modest, uncertain fitting signal without demonstrated generalization. See
[results and actual samples](../../results/helpsteer-v1/distribution-only-diagnostic/)
and the [word-frequency audit](../../results/helpsteer-v1/word-frequency-audit/word-frequency-report.json).

## Fixed targets and question selection

Retain the existing two equally weighted, human-rated HelpSteer references per
question and the pinned reference embeddings. Restrict eligibility to original
training questions with a single-turn prompt of 5–100 whitespace words, both
reference answers of 80–350 words, and a reference length ratio at most two.
Sort eligible question IDs and select four with Python RNG seed 42. Select four
unseen questions from the original validation split with seed 43, independently
of model scores. These rules simplify the fitting diagnostic; they do not make
the targets representative of all valid answers. Final test questions are unused.

Every update uses the same four training questions, with **16 independent answers
per question**, giving **64 answers globally** and 32 per GPU. Each rank handles
two complete question groups. All similarity comparisons and leave-one-out
baselines stay inside the exact question. Unseen references never enter gradients.

The actor starts fresh from pinned Qwen2.5-3B-Instruct. It does not continue the
previous 200-update adapter. MiniLM, full-answer chunk pooling, the two reference
prototypes, energy kernel, frozen kernel scale 1.7710087347953098, distribution
weight 0.5, and initial-policy KL coefficient 0.002 are retained.

## Objective and optimization

The objective to minimize is

```math
L(\theta)=\frac{0.5s_E\;\mathbb E_q[D_E(\pi_\theta(\cdot\mid q),P_q)]
+0.002\;\mathbb E_q[D_{KL}(\pi_\theta(\cdot\mid q)\|\pi_0(\cdot\mid q))]}{1.502}.
```

Quality coefficients are **exactly zero**, with no reward scorer or judge loaded.
The historical fixed divisor 1.502 is retained to avoid rescaling the other terms
when quality is removed. The generated U-statistic may be negative and is not
clipped. MMD coupled leave-one-out credit and initial-policy KL centering are
unchanged. LoRA rank 16/alpha 32, learning rate 5e-6, weight decay 0.01, and the
1,024-token generation cap are retained. This fresh short run uses five warm-up
updates and a 100-update cosine schedule, stopping earlier at its time cap.

## Evaluation and interpretation

Before and after training, generate **32 answers per seen question** (128 total)
and **16 per unseen question** (64 total), using matched evaluation RNG seeds
without advancing training RNG. Repeat seen-question evaluation every ten updates.
Save texts, embeddings, per-question U/V discrepancies, both unscaled Gaussian
and energy scores, prototype coverage, length, termination rate, and training
credits/log ratios. Source, data, configuration, metrics, samples, and final
adapter are logged online to W&B.

The primary comparison is the same four seen questions before versus after
training. A large consistent decrease supports the ability to fit this empirical
target. A seen-only improvement motivates studying generalization. Failure or
small changes under this budget are inconclusive: they could reflect optimization,
representation, regularization, or sampling noise, rather than impossibility.
No writing-quality improvement may be claimed from this diagnostic.

CPU tests exercise deterministic score-independent split selection, budget and
quality guards, 16-answer exact-question groups across two ranks, and the actual
tiny-model DDP update against its global analytical gradient with quality zero
and initial-policy KL active. No GPU pre-test is required.

References: [kCGM paper](https://arxiv.org/abs/2606.19496),
[original author implementation](https://github.com/smithhenryd/cgm),
[previous CPU audit](../../results/helpsteer-v1/kcgm-cpu-audit/README.md).
