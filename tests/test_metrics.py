import numpy as np
import pytest

from stochastic_llm.metrics import cdf_sides, parse_number, score_samples, summarize


def rows(values):
    return [{"text": str(v), "value": v, "valid": True, "terminated": True} for v in values]


@pytest.mark.parametrize("text", ["3 bananas", "1, 2", "NaN", "inf", "1e999", "", "<x>3"])
def test_parser_rejects_non_numbers(text):
    assert parse_number(text) is None


def test_parser_accepts_numeric_notation():
    assert parse_number(" -1.3e-2\n") == -0.013


def test_discrete_ks_handles_cdf_jumps():
    case = {"family": "binomial", "parameters": {"n": 1, "p": 0.5}}
    assert score_samples(case, rows([0, 1]))["ks_statistic"] == pytest.approx(0)
    assert score_samples(case, rows([0, 0]))["ks_statistic"] == pytest.approx(0.5)
    assert score_samples(case, rows([0.5, 0.5]))["support_violation_rate"] == 1


def test_bad_support_values_are_not_filtered_out_of_fidelity():
    case = {"family": "uniform", "parameters": {"low": 0, "high": 1}}
    result = score_samples(case, rows([0.5, 100]))
    assert result["numeric_valid_rate"] == 1
    assert result["support_violation_rate"] == 0.5
    assert result["wasserstein_quantile"] > 40


def test_zero_valid_outputs_do_not_look_like_perfect_calibration():
    case = {"family": "normal", "parameters": {"mean": 0, "std": 1}}
    result = score_samples(
        case, [{"text": "word", "value": None, "valid": False, "terminated": True}]
    )
    assert result["normalized_wasserstein"] is None
    summary = summarize([result | {"family": "normal", "category": "test"}])["all"]
    assert summary["numeric_valid_rate"] == 0
    assert summary["prompts_without_valid_samples"] == 1


def test_continuous_reference_accounts_for_rounding():
    case = {"family": "uniform", "parameters": {"low": 0, "high": 1}}
    left, right = cdf_sides(case, [0.5], decimals=1)
    np.testing.assert_allclose(left, [0.45])
    np.testing.assert_allclose(right, [0.55])
    values = np.round((np.arange(10000) + 0.5) / 10000, 5)
    assert score_samples(case, rows(values))["ks_statistic"] < 0.001


def test_huge_finite_outputs_cannot_overflow_json_reports():
    import json

    case = {"family": "uniform", "parameters": {"low": 0, "high": 0.1}}
    result = score_samples(case, rows([1e308, 1e308]))
    assert result["mean"] == 1e308
    assert result["std"] == 0
    assert result["support_violation_rate"] == 1
    assert result["metric_overflow"]
    json.dumps(result, allow_nan=False)
