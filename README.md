# llm-random-number-generator
Trying to make a LLM that can potentially generate numbers pseudo-randomly from some target distribution


### Information
# Training LLMs to Sample from Probability Distributions

This project investigates whether a language model can learn to generate numbers according to a requested probability distribution.

The goal is **distributional fidelity**: repeated, independent responses should have the correct frequencies, spread, and tail behavior.

For example, given:

> Sample exactly one number from a normal distribution with mean 0 and standard deviation 1.

Repeated responses should approximately follow:

```math
X \sim \mathcal{N}(0,1).
```

Generating individual numbers that look plausible is insufficient. Correct sampling is a property of the distribution of responses across many requests.

## Research Question

Can fine-tuning with a reward based on distributional mismatch teach an LLM to become a better conditional sampler?

We investigate generalization to:

- Unseen parameters of familiar distributions.
- Distribution families absent from training.
- Different prompt wording.
- Eventually, explicitly defined distributions over nonnumeric choices.

## Problem Formulation

Let:

- $c$ denote a prompt specifying a distribution and its parameters.
- $p(x \mid c)$ denote the requested target distribution.
- $\pi_\theta(y \mid c)$ denote the probability of generating response $y$.
- $f(y)$ parse a response into a number, or return $\bot$ for an invalid response.

The model induces an output distribution:

```math
q_\theta(x \mid c)
=
\sum_{y:f(y)=x} \pi_\theta(y \mid c).
```

Our objective is to bring this induced distribution closer to the requested target.

## Batch-Level KL Reward

For each prompt, generate $N$ independent responses:

```math
Y_1,\ldots,Y_N
\overset{\mathrm{iid}}{\sim}
\pi_\theta(\cdot \mid c),
\qquad
X_i=f(Y_i).
```

Each response is generated from a fresh context containing the same prompt.

### Discretizing the Target

We compare probability masses over bins rather than directly comparing an empirical sample with a continuous density.

For numeric bins $B_1,\ldots,B_K$, define:

```math
p_j(c) = \Pr(X \in B_j \mid c).
```

For a continuous target with density $p(x \mid c)$:

```math
p_j(c) = \int_{B_j} p(x \mid c)\,dx.
```

Add an invalid-output category, indexed by $\bot$, with target probability:

```math
p_\bot(c)=0.
```

Malformed responses, unfinished responses, and values outside the distribution's support are assigned to this category.

### Estimating the Model Distribution

Let $n_j$ be the number of generated responses assigned to category $j$.

Using additive smoothing with $\alpha>0$:

```math
\widehat{q}_j
=
\frac{n_j+\alpha}{N+\alpha(K+1)}.
```

Smoothing prevents an unobserved target bin from producing an infinite KL estimate.

### Reward Function

The batch reward is the negative forward KL divergence:

```math
R(c,Y_{1:N})
=
-D_{\mathrm{KL}}\!\left(
p(\cdot \mid c)
\,\middle\|\,
\widehat{q}
\right).
```

Explicitly:

```math
D_{\mathrm{KL}}(p\|\widehat{q})
=
\sum_{j:p_j>0}
p_j \log\frac{p_j}{\widehat{q}_j}.
```

**Higher reward means closer agreement with the target distribution.**

Invalid responses remain in the sample count and normalization. Discarding them before computing the reward would hide poor response validity.

## Reinforcement Learning Objective

We maximize expected batch reward:

```math
J(\theta)
=
\mathbb{E}_{c}
\mathbb{E}_{Y_{1:N}\sim\pi_\theta}
\left[R(c,Y_{1:N})\right].
```

A policy-gradient estimator is:

```math
\nabla_\theta J
=
\mathbb{E}\left[
\sum_{i=1}^{N}
(R-b_i)
\nabla_\theta \log\pi_\theta(Y_i \mid c)
\right].
```

One possible baseline is a leave-one-out reward:

```math
b_i = R(c,Y_{-i}),
```

computed by removing response $i$ and recomputing the histogram and its normalization.

The corresponding sampled loss is:

```math
\mathcal{L}_{\mathrm{RL}}
=
-\sum_{i=1}^{N}
\mathrm{stopgrad}(R-b_i)
\log\pi_\theta(Y_i \mid c).
```

The reward is computed from generated numbers, while gradients flow through the probabilities of their generated token sequences.

A shared batch reward must not simply be mean-centered within that same batch: identical rewards would produce zero advantages.

## Experimental Comparisons

We distinguish three training approaches:

| Method | Training signal |
|---|---|
| Sampled-target SFT | Cross-entropy on answers sampled from the target distribution |
| Direct KL supervision | Match target next-token probabilities |
| Batch-reward RL | Reward distributions estimated from the model's own generated samples |

The proposed RL experiment starts from the SFT checkpoint to investigate improvements beyond the existing formatting and sampling baseline.

The initial model is **Qwen2.5-3B**, adapted using **LoRA**.

## Evaluation

We evaluate independent samples using:

- Numeric response validity.
- Support violations.
- Empirical CDFs and quantile coverage.
- Kolmogorov–Smirnov distance.
- Wasserstein distance.
- Tail behavior and repeated-value concentration.

A conventional numerical sampler provides a reference for the error expected from finite samples.

Evaluation uses held-out prompts and parameters. Improvement on the training reward alone does not establish accurate sampling.

## Limitations

The reward measures agreement at the chosen bin resolution. It cannot fully detect errors within bins, and finite batches introduce estimation noise and smoothing bias.

Matching the training histogram is therefore insufficient: evaluation must also examine finer distributional structure.

The objective is calibrated use of inference-time sampling randomness. It does not guarantee exact continuous sampling, cryptographic randomness, or a new source of entropy.


(Summary Generated by ChatGPT)
