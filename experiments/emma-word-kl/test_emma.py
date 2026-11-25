"""Tests of real prepared text, chapter boundaries, target isolation and screening units."""

import json
import re
import unittest
from collections import Counter

from common import ROOT, words
from preflight import preflight, rows
from prepare import chapters, split_chapters
from screen import batch_diagnostic


class EmmaTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.cfg = json.loads((ROOT / "config.json").read_text())
        cls.chapters = chapters((ROOT / "data/emma.txt").read_text())

    def test_all_chapters_unique_despite_repeated_roman_numerals(self):
        self.assertEqual(len(self.chapters), 55)
        self.assertEqual(
            Counter(r["volume"] for r in self.chapters),
            {"VOLUME I": 18, "VOLUME II": 18, "VOLUME III": 19},
        )
        self.assertEqual(len({r["id"] for r in self.chapters}), 55)
        self.assertTrue(all("VOLUME" not in r["text"] for r in self.chapters))

    def test_splits_do_not_depend_on_input_order(self):
        a = split_chapters(self.chapters, self.cfg)
        b = split_chapters(self.chapters[::-1], self.cfg)
        self.assertEqual(
            {k: {r["id"] for r in v} for k, v in a.items()},
            {k: {r["id"] for r in v} for k, v in b.items()},
        )

    def test_prefix_and_continuation_are_adjacent_original_paragraphs(self):
        chapters_by_id = {r["id"]: r for r in self.chapters}
        for name in ("train", "validation", "test"):
            ends = {}
            for case in rows(ROOT / f"data/policy_{name}.jsonl"):
                chapter = chapters_by_id[case["group_id"]]
                paras = [
                    " ".join(p.split()) for p in re.split(r"\n\s*\n", chapter["text"]) if p.strip()
                ]
                a, b = case["source_paragraph_start"], case["source_paragraph_end"]
                expected = "\n\n".join(paras[a:b])
                self.assertEqual(case["prefix"] + "\n\n" + case["response"], expected)
                self.assertGreaterEqual(a, ends.get(case["group_id"], 0))
                ends[case["group_id"]] = b
                self.assertGreaterEqual(len(words(case["prefix"])), 80)
                self.assertLessEqual(len(words(case["prefix"])), 180)
                self.assertGreaterEqual(len(words(case["response"])), 250)
                self.assertLessEqual(len(words(case["response"])), 450)

    def test_real_preflight_and_training_only_pmf(self):
        result = preflight()
        self.assertTrue(result["passed"])
        self.assertEqual(
            result["continuation_counts"], {"train": 168, "validation": 35, "test": 29}
        )
        self.assertFalse(result["gpu_submission_authorized"])
        json.loads(json.dumps(result))

    def test_screen_uses_equal_word_batches_and_is_reproducible(self):
        cfg = {
            **self.cfg,
            "screen_passage_words": 4,
            "screen_resamples": 5,
            "vocabulary_min_count": 1,
        }
        groups = [
            ("a b c a " * 20).strip(),
            ("b a d a " * 20).strip(),
            ("c b d a " * 20).strip(),
            ("c c d a " * 20).strip(),
        ]
        a = batch_diagnostic(groups, groups, cfg)
        b = batch_diagnostic(groups, groups, cfg)
        self.assertEqual(a, b)
        self.assertEqual(a["words_per_batch"], 64)
        self.assertEqual(a["groups_with_at_least_four_passages"], 4)
        self.assertGreaterEqual(a["random_16_passages"]["kl_nats"]["median"], 0)
        json.loads(json.dumps(a))

    def test_report_does_not_claim_training_results(self):
        result = json.loads((ROOT / "screening-results.json").read_text())
        self.assertFalse(result["trained_model_results"])
        self.assertFalse(result["gpu_used"])
        comparison = result["matched_section_comparison"]
        self.assertEqual(
            comparison["emma"]["shared_1000_words_plus_other"]["word_categories"],
            comparison["tiny_shakespeare"]["shared_1000_words_plus_other"]["word_categories"],
        )


if __name__ == "__main__":
    unittest.main()
