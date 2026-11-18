import numpy as np
import pytest

from stochastic_llm.distributions import (
    HELD_OUT_FAMILIES,
    TRAIN_FAMILIES,
    distribution,
    finite_target,
    format_number,
    sample_parameters,
)


@pytest.mark.parametrize("family", TRAIN_FAMILIES + HELD_OUT_FAMILIES)
def test_finite_targets_are_normalized_unique_and_ordered(family):
    case = {"family": family, "parameters": sample_parameters(family, np.random.default_rng(4))}
    outputs, masses = finite_target(case, max_bins=101)
    assert len(outputs) <= 101
    assert len(set(outputs)) == len(outputs)
    assert np.all(masses > 0)
    assert masses.sum() == pytest.approx(1)
    assert np.all(np.diff([float(x) for x in outputs]) > 0)
    assert np.isfinite(distribution(case).rvs(random_state=np.random.default_rng(4)))


def test_negative_zero_is_not_a_separate_output():
    assert format_number(-0.00000001, "normal") == "0.00000"


def test_binomial_keeps_its_entire_support_when_it_fits():
    case = {"family": "binomial", "parameters": {"n": 8, "p": 0.3}}
    outputs, masses = finite_target(case)
    assert outputs == [str(x) for x in range(9)]
    np.testing.assert_allclose(masses, distribution(case).pmf(np.arange(9)))
