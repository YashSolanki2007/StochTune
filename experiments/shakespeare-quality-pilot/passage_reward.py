"""Full word-bigram reward with exact leave-one-passage-out credit assignment."""

import numpy as np
from pilot import pooled_counts


class PassageDistribution:
    def __init__(self, reference_passages, smoothing_mass=1.0):
        reference = pooled_counts(reference_passages)
        if not reference or smoothing_mass <= 0:
            raise ValueError("Need nonempty reference and positive smoothing mass")
        self.index = {pair: i for i, pair in enumerate(sorted(reference))}
        self.unknown = len(self.index)
        self.probabilities = np.array(
            [reference[pair] for pair in sorted(reference)] + [0.0], dtype=np.float64
        )
        self.probabilities /= self.probabilities.sum()
        self.smoothing_mass = smoothing_mass
        self.alpha = smoothing_mass / len(self.probabilities)

    def counts(self, passages):
        result = np.zeros((len(passages), len(self.probabilities)), dtype=np.float64)
        for i, passage in enumerate(passages):
            pairs = pooled_counts([passage])
            for pair, count in pairs.items():
                result[i, self.index.get(pair, self.unknown)] += count
            if not pairs:
                result[i, self.unknown] = 1  # Empty/one-word outputs are not dropped.
        return result

    def kl(self, counts):
        total = counts.sum()
        if total < 1 or np.any(counts < 0):
            raise ValueError("Invalid counts")
        q = (counts + self.alpha) / (total + self.smoothing_mass)
        mask = self.probabilities > 0
        return float(np.sum(self.probabilities[mask] * np.log(self.probabilities[mask] / q[mask])))

    def credit(self, passages):
        if len(passages) < 2:
            raise ValueError("Leave-one-out reward requires at least two passages")
        counts = self.counts(passages)
        total = counts.sum(0)
        cost = self.kl(total)
        # Reward=-KL. A_i=N*(R(group)-R(without passage i)).
        credit = len(passages) * np.array([self.kl(total - row) - cost for row in counts])
        return {
            "kl": cost,
            "advantages": credit,
            "unknown_bigram_rate": float(total[-1] / total.sum()),
        }


def quality_advantages(rewards):
    values = np.asarray(rewards, dtype=np.float64)
    if len(values) < 2 or not np.all(np.isfinite(values)):
        raise ValueError("Need finite rewards for at least two passages")
    return values - (values.sum() - values) / (len(values) - 1)
