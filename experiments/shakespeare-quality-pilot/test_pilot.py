"""Checks for metric and score-validation failure modes; no model downloads."""

import json
import unittest
from collections import Counter

from judge import parse_score
from pilot import distribution_metrics, pooled_counts, words


class PilotTests(unittest.TestCase):
    def test_no_cross_passage_bigrams(self):
        self.assertEqual(pooled_counts(["a b", "c d"]), Counter({("a", "b"): 1, ("c", "d"): 1}))

    def test_unknown_words_are_retained(self):
        ref = Counter({("a", "b"): 10})
        faithful = distribution_metrics(ref, ref)
        unknown = distribution_metrics(ref, Counter({("c", "d"): 10}))
        self.assertEqual(unknown["unknown_bigram_rate"], 1)
        self.assertGreater(unknown["forward_kl"], faithful["forward_kl"])

    def test_normalization_and_tokenization(self):
        self.assertEqual(words("THOU’RT, king!"), ["thou'rt", "king"])
        ref = Counter({("a", "b"): 10, ("b", "c"): 4})
        result = distribution_metrics(ref, ref)
        self.assertGreaterEqual(result["forward_kl"], 0)
        self.assertLess(result["js_divergence"], 0.01)

    def test_boolean_score_rejected(self):
        raw = dict(
            coherence=True,
            fluency=5,
            prompt_fulfillment=5,
            shakespearean_style=4,
            rationale="Example",
        )
        with self.assertRaises(ValueError):
            parse_score(json.dumps(raw))

    def test_source_has_no_overall_task_reward(self):
        raw = dict(
            coherence=5,
            fluency=5,
            prompt_fulfillment=None,
            shakespearean_style=5,
            rationale="Example",
        )
        self.assertIsNone(parse_score(json.dumps(raw), False)["quality_reward"])
        with self.assertRaises(ValueError):
            parse_score(json.dumps(raw), True)


if __name__ == "__main__":
    unittest.main()
