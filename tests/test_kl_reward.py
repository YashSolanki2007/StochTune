import itertools

import numpy as np
import pytest
import torch

from stochastic_llm.distributions import distribution
from stochastic_llm.kl_reward import HistogramTarget, group_kl_reward, histogram_kl


def test_bin_masses_are_exact_cdf_differences_and_reject_invalid_outputs():
    case = {"family": "binomial", "parameters": {"n": 5, "p": 0.3}}
    target = HistogramTarget.from_case(case, num_bins=32)
    np.testing.assert_allclose(target.probabilities[:-1], distribution(case).pmf(np.arange(6)))
    assert target.probabilities[-1] == 0
    for bad in (None, float("inf"), -1, 1.5, 6):
        assert target.assign(bad) == target.invalid_bin
    assert target.assign(2, terminated=False) == target.invalid_bin
    assert target.assign(2) != target.invalid_bin


def test_continuous_bins_keep_tails_and_penalize_invalid_support():
    normal = HistogramTarget.from_case({"family": "normal", "parameters": {"mean": 0, "std": 1}})
    assert normal.assign(-1e50) == 0
    assert normal.assign(1e50) == normal.invalid_bin - 1
    assert normal.probabilities.sum() == pytest.approx(1)
    uniform = HistogramTarget.from_case({"family": "uniform", "parameters": {"low": 0, "high": 1}})
    assert uniform.assign(-0.5) == uniform.invalid_bin
    assert uniform.assign(1.5) == uniform.invalid_bin


def test_spread_has_better_reward_than_collapsed_or_invalid_groups():
    p = np.array([0.5, 0.5, 0])
    spread = group_kl_reward([0] * 16 + [1] * 16, p)
    collapsed = group_kl_reward([0] * 32, p)
    invalid = group_kl_reward([2] * 32, p)
    assert spread["reward"] > collapsed["reward"] > invalid["reward"]
    assert np.all(invalid["advantages"] < 0)
    assert np.isfinite(invalid["kl"])


def test_fast_leave_one_out_credit_matches_explicit_recomputed_rewards():
    indices = np.array([0, 0, 0, 1, 2])
    p = np.array([0.25, 0.75, 0.0])
    reward = group_kl_reward(indices, p, pseudocount=0.1)
    for i in range(len(indices)):
        other = np.delete(indices, i)
        baseline = -histogram_kl(np.bincount(other, minlength=3), p, 0.1)
        assert reward["advantages"][i] == pytest.approx(
            len(indices) * (reward["reward"] - baseline)
        )


def test_exact_policy_gradient_matches_enumeration_of_the_group_objective():
    # Enumerate every possible pair of actions. This catches the shared-reward
    # normalization trap, the sign, the leave-one-out baseline, and N scaling.
    logits = torch.tensor([0.9, -0.4, -0.2], dtype=torch.float64, requires_grad=True)
    probabilities = logits.softmax(0)
    target = np.array([0.3, 0.7, 0.0])
    exact_objective = 0.0
    estimator = 0.0
    for pair in itertools.product(range(3), repeat=2):
        probability = probabilities[list(pair)].prod()
        group = group_kl_reward(pair, target, 0.2)
        exact_objective = exact_objective + probability * group["reward"]
        credit = torch.tensor(group["advantages"], dtype=torch.float64)
        estimator = (
            estimator + probability.detach() * (credit * probabilities[list(pair)].log()).mean()
        )
    exact_grad = torch.autograd.grad(exact_objective, logits, retain_graph=True)[0]
    estimated_grad = torch.autograd.grad(estimator, logits)[0]
    torch.testing.assert_close(exact_grad, estimated_grad, atol=1e-12, rtol=1e-12)
    assert exact_grad.norm() > 0


def test_reward_optimization_learns_target_and_reduces_invalid_mass():
    torch.manual_seed(21)
    logits = torch.nn.Parameter(torch.tensor([1.5, -0.5, 0.5]))
    optimizer = torch.optim.Adam([logits], lr=0.06)
    target = np.array([0.25, 0.75, 0.0])
    for _ in range(180):
        samples = torch.distributions.Categorical(logits=logits).sample((256,))
        group = group_kl_reward(samples.numpy(), target, 0.1)
        credit = torch.tensor(group["advantages"], dtype=logits.dtype)
        loss = -(credit * logits.log_softmax(0)[samples]).mean()
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()
    learned = logits.softmax(0).detach().numpy()
    np.testing.assert_allclose(learned[:2], target[:2], atol=0.06)
    assert learned[2] < 0.02
