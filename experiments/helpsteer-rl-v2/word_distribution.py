"""A fixed training-word vocabulary with probability-preserving OTHER aggregation."""

import numpy as np
from passage_reward import PassageDistribution


class BucketedWordDistribution(PassageDistribution):
    def __init__(self, reference_passages, max_words=10000, min_count=5, smoothing_mass=1.0):
        if max_words < 1 or min_count < 1:
            raise ValueError("Vocabulary size and minimum frequency must be positive")
        references = list(reference_passages)
        super().__init__(references, smoothing_mass=smoothing_mass, order=1)
        self.full_target = PassageDistribution(references, smoothing_mass=smoothing_mass, order=1)
        reference = self.pooled_counts(references)
        ranked = sorted(reference, key=lambda word: (-reference[word], word))
        retained = sorted([word for word in ranked if reference[word] >= min_count][:max_words])
        if not retained:
            raise ValueError("No words meet the vocabulary frequency threshold")
        self.index = {word: i for i, word in enumerate(retained)}
        self.unknown = len(self.index)  # Internal index: all excluded AND unseen words.
        other_count = sum(count for word, count in reference.items() if word not in self.index)
        self.probabilities = np.array(
            [reference[word] for word in retained] + [other_count], dtype=np.float64
        )
        self.probabilities /= self.probabilities.sum()
        self.alpha = smoothing_mass / len(self.probabilities)
        self.name = "bucketed_unigram"

    def metrics(self, result):
        return {
            "distribution_kl": result["kl"],
            "bucketed_unigram_kl": result["kl"],
            "other_word_rate": result["unknown_rate"],
            "reference_other_mass": float(self.probabilities[-1]),
        }
