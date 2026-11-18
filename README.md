# llm-random-number-generator
Trying to make a LLM that can potentially generate numbers pseudo-randomly from some target distribution


### Information
# Training LLMs to Sample from Probability Distributions

This project investigates whether a language model can learn to generate numbers according to a requested probability distribution.

The goal is **distributional fidelity**: when repeatedly asked to sample from a distribution, the model should produce values with the correct frequencies, spread, and tail behavior.

For example, given:

> Sample exactly one number from a normal distribution with mean 0 and standard deviation 1.

Repeated, independent responses should approximately follow:

$$
X \sim \mathcal{N}(0,1).
$$

Generating individual numbers that look plausible is insufficient. Correct sampling is a property of the distribution of responses across many requests.

## Research Question

Can fine-tuning with a reward based on distributional mismatch teach an LLM to become a better conditional sampler?

We also investigate whether this behavior generalizes to:

- Unseen parameters of familiar distributions.
- Distribution families absent from training.
- Different prompt wording.
- Eventually, explicitly defined distributions over nonnumeric choices.

## Problem Formulation

Let:

- $c$ denote a prompt specifying a distribution and its parameters.
- $p(x \mid c)$ denote the requested target distribution.
- $\pi_\theta(y \mid c)$ denote the model's probability of generating response $y$.
- $f(y)$ parse the response into a number, or return $\bot$ for an invalid response.

The model induces an output distribution:

$$
q_\theta(x \mid c)
=
\sum_{y:f(y)=x}
\pi_\theta(y \mid c).
$$

Our objective is to bring this induced distribution closer to the requested target.

## Batch-Level KL Reward

For each prompt, generate $N$ independent responses:

$$
Y_1,\ldots,Y_N
\overset{\mathrm{iid}}{\sim}
\pi_\theta(\cdot \mid c),
\qquad
X_i=f(Y_i).
$$

Each response is generated from a fresh context containing the same prompt.

### Discretizing the Target

Because model outputs are finite numeric strings, we compare probability masses over bins rather than directly comparing an empirical sample with a continuous density.

For numeric bins $B_1,\ldots,B_K$, define:

$$
p_j(c)
=
P(X \in B_j \mid c).
$$

For a continuous target with density $p(x \mid c)$:

$$
p_j(c)
=
\int_{B_j} p(x \mid c)\,dx.
$$

Add an invalid-output category, indexed by $\bot$, with target probability:

$$
p_\bot(c)=0.
$$

Malformed responses, unfinished responses, and values outside the distribution's support are assigned to this category.

### Estimating the Model Distribution

Let $n_j$ be the number of generated responses assigned to category $j$.

Using additive smoothing with $\alpha>0$:

$$
\widehat{q}_j
=
\frac{n_j+\alpha}
{N+\alpha(K+1)}.
$$

Smoothing prevents an unobserved target bin from producing an infinite KL estimate.

The batch reward is:

$$
R(c,Y_{1:N})
=
-
D_{\mathrm{KL}}
\left(
p(\cdot \mid c)
\;\middle\|\;
\widehat{q}
\right),
$$

where:

$$
D_{\mathrm{KL}}(p\|\widehat{q})
=
\sum_{j:p_j>0}
p_j
\log\frac{p_j}{\widehat{q}_j}.
$$

Higher reward means closer agreement with the target distribution.

Invalid responses remain in the sample count and normalization. Discarding them before computing the reward would allow poor response validity to be hidden.

## Reinforcement Learning Objective

We maximize expected batch reward:

$$
J(\theta)
=
\mathbb{E}_{c}
\mathbb{E}_{Y_{1:N}\sim\pi_\theta}
\left[
R(c,Y_{1:N})
\right].
$$

A policy-gradient estimator is:

$$
\nabla_\theta J
=
\mathbb{E}
\left[
\sum_{i=1}^{N}
(R-b_i)
\nabla_\theta
\log \pi_\theta(Y_i \mid c)
\right].
$$

One possible variance-reduction baseline is a leave-one-out reward:

$$
b_i
=
R(c,Y_{-i}),
$$

computed by removing response $i$ and recomputing the histogram and its normalization.

The corresponding sampled loss is:

$$
\mathcal{L}_{\mathrm{RL}}
=
-
\sum_{i=1}^{N}
\operatorname{stopgrad}(R-b_i)
\log \pi_\theta(Y_i \mid c).
$$

The reward is computed from generated numbers, while gradients flow through the probabilities of their generated token sequences.

A shared batch reward must not simply be mean-centered within that same batch: identical rewards would then produce zero advantages.

## Experimental Comparisons

We distinguish three training approaches:

| Method | Training signal |
|---|---|
| Sampled-target SFT | Cross-entropy on answers sampled from the target distribution |
| Direct KL supervision | Match target next-token probabilities |
| Batch-reward RL | Reward distributions estimated from the model's own generated samples |

The proposed RL experiment starts from the SFT checkpoint to study whether the batch reward improves calibration beyond the existing formatting and sampling baseline.

The initial model is **Qwen2.5-3B**, adapted using LoRA.

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

Consequently, matching the training histogram is insufficient: evaluation must also examine finer distributional structure.

The objective is calibrated use of inference-time sampling randomness. It does not claim to create a new entropy source, guarantee exact continuous sampling, or establish cryptographic randomness.



(Summary Generated by ChatGPT)
