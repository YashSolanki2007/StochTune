"""A genuine group-level empirical KL reward and its REINFORCE credit assignment."""

from dataclasses import dataclass

import numpy as np

from .distributions import DISCRETE_FAMILIES, distribution


@dataclass
class HistogramTarget:
    edges: np.ndarray
    probabilities: np.ndarray
    low: float
    high: float
    discrete: bool
    decimals: int

    @classmethod
    def from_case(cls, case, num_bins=32, decimals=5):
        if num_bins < 2 or decimals < 0:
            raise ValueError("num_bins must be >=2 and decimals must be nonnegative")
        law = distribution(case)
        low, high = law.support()
        discrete = case["family"] in DISCRETE_FAMILIES
        quantiles = law.ppf(np.arange(1, num_bins) / num_bins)
        # Boundaries lie between representable target outputs. Exact CDF
        # differences account for discrete atoms and rounding of continuous laws.
        if discrete:
            if np.isfinite(low) and np.isfinite(high) and high - low + 1 <= num_bins:
                edges = np.arange(low, high) + 0.5
            else:
                edges = np.unique(quantiles) + 0.5
        else:
            edges = np.unique(np.round(quantiles, decimals)) + 0.5 * 10.0 ** (-decimals)
        cdf = law.cdf(edges)
        keep = (cdf > 0) & (cdf < 1)
        edges, cdf = edges[keep], cdf[keep]
        keep = np.diff(np.concatenate(([0.0], cdf))) > 0
        edges, cdf = edges[keep], cdf[keep]
        masses = np.diff(np.concatenate(([0.0], cdf, [1.0])))
        # Last category is malformed/truncated/out-of-support output, with P=0.
        return cls(edges, np.append(masses, 0.0), float(low), float(high), discrete, decimals)

    @property
    def invalid_bin(self):
        return len(self.probabilities) - 1

    def assign(self, value, terminated=True):
        if value is None or not terminated or not np.isfinite(value):
            return self.invalid_bin
        tolerance = 0 if self.discrete else 0.5 * 10.0 ** (-self.decimals)
        if value < self.low - tolerance or value > self.high + tolerance:
            return self.invalid_bin
        if self.discrete and value != np.floor(value):
            return self.invalid_bin
        return int(np.searchsorted(self.edges, value, side="right"))


def histogram_kl(counts, probabilities, pseudocount=0.1):
    """KL(P || smoothed empirical Q), including invalid draws in Q's denominator."""
    counts = np.asarray(counts, dtype=np.float64)
    probabilities = np.asarray(probabilities, dtype=np.float64)
    if (
        counts.shape != probabilities.shape
        or np.any(counts < 0)
        or counts.sum() < 1
        or pseudocount <= 0
        or np.any(probabilities < 0)
        or not np.isclose(probabilities.sum(), 1)
    ):
        raise ValueError("Invalid histogram, probabilities, or pseudocount")
    observed = (counts + pseudocount) / (counts.sum() + pseudocount * len(counts))
    positive = probabilities > 0
    return float(
        np.sum(probabilities[positive] * np.log(probabilities[positive] / observed[positive]))
    )


def group_kl_reward(bin_indices, probabilities, pseudocount=0.1):
    """Return group reward -KL and N-scaled leave-one-out advantages.

    R_N is shared by all N actions. A baseline R_{N-1}(other actions) is independent
    of action i and therefore preserves the exact group policy gradient. Scaling
    advantages by N compensates for averaging sequence losses instead of summing
    them. There is no within-group mean/std normalization (which would erase a
    shared group reward).
    """
    indices = np.asarray(bin_indices, dtype=np.int64)
    probabilities = np.asarray(probabilities, dtype=np.float64)
    if (
        indices.ndim != 1
        or len(indices) < 2
        or np.any(indices < 0)
        or np.any(indices >= len(probabilities))
    ):
        raise ValueError("A group needs at least two valid category indices")
    counts = np.bincount(indices, minlength=len(probabilities))
    cost = histogram_kl(counts, probabilities, pseudocount)
    n, k = len(indices), len(probabilities)
    common = np.log((n - 1 + pseudocount * k) / (n + pseudocount * k))
    selected = counts[indices].astype(np.float64)
    differences = common + probabilities[indices] * np.log(
        (selected + pseudocount) / (selected - 1 + pseudocount)
    )
    return {"reward": -cost, "kl": cost, "counts": counts, "advantages": n * differences}
