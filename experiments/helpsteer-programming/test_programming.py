"""CPU correctness checks for bucket mass, policy credit, and dataset leakage."""

import hashlib
import itertools
import json
import sys
import unittest
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parent
sys.path.insert(0, str(ROOT))
sys.path.insert(0, str(ROOT.parent / "helpsteer-rl-v2"))
from common import distribution_diagnostics, make_distribution, read_rows  # noqa: E402
from topic_filter import prompt_groups  # noqa: E402
from word_distribution import BucketedWordDistribution  # noqa: E402


class ProgrammingTests(unittest.TestCase):
    def test_tail_and_novel_words_are_counted_in_other(self):
        target = BucketedWordDistribution(["a a a b b tail"], max_words=2, min_count=2)
        np.testing.assert_allclose(target.probabilities, [3 / 6, 2 / 6, 1 / 6])
        np.testing.assert_array_equal(target.counts(["a tail novel b"]), [[1, 1, 2]])
        self.assertEqual(target.counts([""])[0, -1], 1)

    def test_top_limit_is_deterministic_and_does_not_renormalize_head(self):
        target = BucketedWordDistribution(["c b a"], max_words=2, min_count=1)
        self.assertEqual(list(target.index), ["a", "b"])
        np.testing.assert_allclose(target.probabilities, [1 / 3] * 3)

    def test_vocabulary_stays_fixed_during_generation(self):
        target = BucketedWordDistribution(["a a b tail"], min_count=2)
        original = target.probabilities.copy()
        target.credit(["novel " * 100, "b a"])
        self.assertEqual(list(target.index), ["a"])
        np.testing.assert_array_equal(target.probabilities, original)

    def test_kl_uses_full_probability_mass_including_other(self):
        target = BucketedWordDistribution(["a a b tail"], min_count=2)
        counts = target.counts(["a a a b"])[0]
        q = (counts + 0.5) / 5
        expected = np.sum(np.array([0.5, 0.5]) * np.log(0.5 / q))
        self.assertAlmostEqual(target.kl(counts), expected)

    def test_leave_one_out_credit_matches_exact_expected_policy_gradient(self):
        import torch

        target = BucketedWordDistribution(["a a b tail"], max_words=2, min_count=1)
        passages = ["a a", "b", "novel"]
        logits = torch.tensor([0.2, -0.4, 0.5], dtype=torch.float64, requires_grad=True)
        log_probs = logits.log_softmax(0)
        objective = logits.sum() * 0
        estimate = logits.sum() * 0
        for indices in itertools.product(range(3), repeat=3):
            indices = list(indices)
            probability = log_probs[indices].sum().exp()
            credit = target.credit([passages[i] for i in indices])
            objective = objective - probability * credit["kl"]
            estimate = (
                estimate
                + probability.detach()
                * (log_probs[indices] * torch.tensor(credit["advantages"]) / 3).sum()
            )
        exact = torch.autograd.grad(objective, logits, retain_graph=True)[0]
        actual = torch.autograd.grad(estimate, logits)[0]
        torch.testing.assert_close(actual, exact, rtol=1e-10, atol=1e-10)

    def test_factory_keeps_old_default_and_reports_full_diagnostic(self):
        full = make_distribution({}, ["a a b"])
        self.assertEqual(full.name, "unigram")
        target = make_distribution(
            {
                "word_distribution": "frequency-capped-other",
                "vocabulary_max_words": 10,
                "vocabulary_min_count": 2,
            },
            ["a a b"],
        )
        result = distribution_diagnostics(target, ["a b novel"])
        self.assertIn("full_vocabulary_unigram_kl", result)
        self.assertAlmostEqual(result["other_word_rate"], 2 / 3)
        self.assertRaises(ValueError, make_distribution, {"word_distribution": "typo"}, ["a"])

    def test_conversation_continuations_cannot_cross_splits(self):
        rows = [
            {"id": "a", "prompt": "Teach me Python."},
            {"id": "b", "prompt": "teach me python<extra_id_1>Assistant OK<extra_id_1>User next"},
            {"id": "c", "prompt": "Explain Rust references"},
        ]
        groups = prompt_groups(rows)
        self.assertEqual(sorted(len(g) for g in groups), [1, 2])

    def test_prepared_target_and_splits_match_manifest(self):
        data = ROOT / "data"
        if not (data / "manifest.json").exists():
            self.skipTest("Run prepare.py before checking the prepared corpus")
        manifest = json.loads((data / "manifest.json").read_text())
        for name, expected in manifest["prepared_sha256"].items():
            self.assertEqual(hashlib.sha256((data / name).read_bytes()).hexdigest(), expected, name)
        sets = [set(ids) for ids in manifest["split_prompt_ids"].values()]
        family_members = [
            {key for members in families.values() for key in members}
            for families in manifest["conversation_families"].values()
        ]
        for groups in (sets, family_members):
            for i, first in enumerate(groups):
                for second in groups[i + 1 :]:
                    self.assertFalse(first & second)
                self.assertFalse(first & set(manifest["reserved_conversation_member_ids"]))
        self.assertEqual(
            set(manifest["reference_prompt_ids"]), set(manifest["split_prompt_ids"]["train"])
        )
        cfg = json.loads((ROOT / "config.json").read_text())
        self.assertFalse(cfg["gpu_submission_authorized"])
        target = make_distribution(
            cfg, [r["response"] for r in read_rows(data / "reference.jsonl")]
        )
        vocabulary = json.loads((data / "vocabulary.json").read_text())
        np.testing.assert_array_equal(
            target.probabilities, [c["probability"] for c in vocabulary["categories"]]
        )
        self.assertEqual(
            (data / "reward_validation.jsonl").read_bytes(),
            (ROOT.parent / "helpsteer-rl-v2/data/reward_validation.jsonl").read_bytes(),
        )


if __name__ == "__main__":
    unittest.main()
