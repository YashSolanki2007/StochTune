"""Numeric validity and distribution fidelity, including discrete/rounded CDFs."""

import re

import numpy as np

from .distributions import DISCRETE_FAMILIES, distribution, format_number

NUMBER = re.compile(r"[+-]?(?:\d+(?:\.\d*)?|\.\d+)(?:[eE][+-]?\d+)?", flags=re.ASCII)


def stable_mean(values):
    scale = float(np.max(np.abs(values)))
    return float(np.mean(values / scale) * scale) if scale else 0.0


def stable_std(values):
    scale = float(np.max(np.abs(values)))
    return float(np.std(values / scale) * scale) if scale else 0.0


def parse_number(text):
    if not NUMBER.fullmatch(text.strip()):
        return None
    try:
        value = float(text.strip())
    except (ValueError, OverflowError):
        return None
    return value if np.isfinite(value) else None


def cdf_sides(case, values, decimals=5):
    """F(x-) and F(x) for the actual rounded/discrete reference law."""
    law = distribution(case)
    values = np.asarray(values, dtype=np.float64)
    if case["family"] in DISCRETE_FAMILIES:
        left = law.cdf(np.ceil(values) - 1)
        right = law.cdf(np.floor(values))
    else:
        step = 10.0 ** (-decimals)
        # Small tolerance only compensates floating-point error at decimal grid points.
        # Huge finite outputs still need a score, even if x / step overflows.
        grid = np.clip(values, -1e290, 1e290) / step
        tolerance = np.minimum(1e-5, np.abs(grid) * np.finfo(float).eps * 4 + 1e-8)
        below = (np.ceil(grid - tolerance) - 0.5) * step
        through = (np.floor(grid + tolerance) + 0.5) * step
        left, right = law.cdf(below), law.cdf(through)
    return left, right


def score_samples(case, rows, decimals=5):
    if not rows:
        raise ValueError("Cannot score zero samples")
    # No filtering of valid but out-of-support values: they must affect fidelity.
    values = np.array([r["value"] for r in rows if r["valid"]], dtype=np.float64)
    result = {
        "requested": len(rows),
        "valid": int(len(values)),
        "numeric_valid_rate": len(values) / len(rows),
        "truncated_rate": sum(not r["terminated"] for r in rows) / len(rows),
        "canonical_rate": sum(
            r["valid"] and r["text"].strip() == format_number(r["value"], case["family"], decimals)
            for r in rows
        )
        / len(rows),
        "wasserstein_quantile": None,
        "normalized_wasserstein": None,
        "ks_statistic": None,
        "support_violation_rate": None,
        "unique_rate": None,
        "mean": None,
        "std": None,
        "metric_overflow": False,
    }
    if not len(values):
        return result
    law = distribution(case)
    low, high = law.support()
    margin = 0 if case["family"] in DISCRETE_FAMILIES else 0.5 * 10.0 ** (-decimals)
    violations = (values < low - margin) | (values > high + margin)
    if case["family"] in DISCRETE_FAMILIES:
        violations |= values != np.floor(values)
    ordered = np.sort(values)
    probabilities = (np.arange(len(values)) + 0.5) / len(values)
    reference = law.ppf(probabilities)
    if case["family"] not in DISCRETE_FAMILIES:
        reference = np.round(reference, decimals)
    # Midpoint quantile approximation, not an exact integral for continuous laws.
    wasserstein = stable_mean(np.abs(ordered - reference))
    width = float(law.ppf(0.95) - law.ppf(0.05))
    unique, counts = np.unique(ordered, return_counts=True)
    cumulative = np.cumsum(counts) / len(values)
    before = (np.cumsum(counts) - counts) / len(values)
    target_left, target_right = cdf_sides(case, unique, decimals)
    ks = float(max(np.max(np.abs(before - target_left)), np.max(np.abs(cumulative - target_right))))
    with np.errstate(over="ignore"):
        normalized = float(np.float64(wasserstein) / width) if width > 0 else None
    overflow = normalized is not None and not np.isfinite(normalized)
    result.update(
        {
            "wasserstein_quantile": wasserstein,
            "normalized_wasserstein": None if overflow else normalized,
            "ks_statistic": ks,
            "support_violation_rate": float(np.mean(violations)),
            "unique_rate": len(unique) / len(values),
            "mean": stable_mean(values),
            "std": stable_std(values),
            "metric_overflow": bool(overflow),
        }
    )
    return result


def summarize(results):
    groups = {"all": results}
    for result in results:
        for field in ("family", "category"):
            groups.setdefault(f"{field}:{result[field]}", []).append(result)
    summary = {}
    for name, rows in groups.items():
        requested = sum(r["requested"] for r in rows)
        finite = [
            r["normalized_wasserstein"] for r in rows if r["normalized_wasserstein"] is not None
        ]
        ks = [r["ks_statistic"] for r in rows if r["ks_statistic"] is not None]
        summary[name] = {
            "prompts": len(rows),
            "requested": requested,
            "numeric_valid_rate": sum(r["valid"] for r in rows) / requested,
            "prompts_without_valid_samples": sum(r["valid"] == 0 for r in rows),
            "prompts_with_metric_overflow": sum(r.get("metric_overflow", False) for r in rows),
            "median_normalized_wasserstein": float(np.median(finite)) if finite else None,
            "median_ks_statistic": float(np.median(ks)) if ks else None,
        }
    return summary
