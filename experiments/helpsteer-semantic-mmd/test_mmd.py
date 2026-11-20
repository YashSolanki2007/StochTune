"""CPU mathematical, conditioning, full-text, and two-rank credit checks."""

import itertools
import json
import sys
import tempfile
import unittest
from pathlib import Path

import numpy as np
import torch
import torch.distributed as dist
import torch.multiprocessing as mp

ROOT = Path(__file__).resolve().parent
sys.path.insert(0, str(ROOT))
sys.path.insert(0, str(ROOT.parent / "helpsteer-rl-v2"))
from embedding import content_chunks  # noqa: E402
from mmd import kernel, mmd2, passage_credit  # noqa: E402
from semantic_runtime import ConditionalSemanticTarget  # noqa: E402
from train_policy import global_credit, training_assignment  # noqa: E402


def dummy_target():
    target = ConditionalSemanticTarget.__new__(ConditionalSemanticTarget)
    target.references = {
        "q1": np.array([[-1.0, 0], [1.0, 0]]),
        "q2": np.array([[0.0, -1], [0.0, 1]]),
    }
    target.train_ids = {"q1", "q2"}
    target.bandwidth_squared = 1.0
    return target


def credit_worker(rank, folder):
    torch.set_num_threads(1)
    dist.init_process_group("gloo", init_method=f"file://{folder}/store", rank=rank, world_size=2)
    try:
        target = dummy_target()
        rows = [
            {"case_id": q, "passage": f"{q}-{i}", "semantic_embedding": v.tolist()}
            for q, values in target.references.items()
            for i, v in enumerate(values[[0, 1, 0, 0]])
        ]
        quality = np.array([0.1, 0.5, 0.2, 0.3, 0.8, 0.2, 0.4, 0.9])
        expected_qa, expected_da, _ = target.training_credit(rows, quality)
        credit, metrics = global_credit(
            rows[rank::2], quality[rank::2], target, 0.5, torch.device("cpu")
        )
        np.testing.assert_allclose(
            credit.numpy(), (expected_qa + 0.5 * expected_da)[rank::2], atol=1e-7
        )
        assert metrics["mmd_questions"] == 2 and metrics["answers_per_question"] == 4
        assert "conditional_mmd2_u" in metrics and "unigram_kl" not in metrics
    finally:
        dist.destroy_process_group()


class MMDTests(unittest.TestCase):
    def test_full_text_chunking_never_drops_or_duplicates_tokens(self):
        for n in (0, 1, 254, 255, 1025):
            ids = list(range(n))
            chunks = content_chunks(ids, 254)
            self.assertEqual([i for chunk in chunks for i in chunk], ids)
            self.assertTrue(all(len(chunk) <= 254 for chunk in chunks))

    def test_distribution_matching_detects_collapse_and_mean_only_failure(self):
        reference = np.array([[-1.0], [1.0]])
        matched = reference[[0, 0, 1, 1]]
        collapsed = reference[[0, 0, 0, 0]]
        same_mean_but_wrong = np.zeros((4, 1))
        self.assertLess(mmd2(matched, reference, 1), mmd2(collapsed, reference, 1))
        self.assertGreater(mmd2(same_mean_but_wrong, reference, 1, unbiased=False), 0)
        self.assertAlmostEqual(mmd2(matched, reference, 1, unbiased=False), 0)
        self.assertLess(mmd2(matched, reference, 1), 0)  # Never clip a negative U estimate.

    def test_unbiased_cost_and_exact_expected_policy_gradient(self):
        values = np.array([[-1.0], [0.2], [1.0]])
        reference = np.array([[-1.0], [1.0]])
        logits = torch.tensor([0.2, -0.4, 0.5], dtype=torch.float64, requires_grad=True)
        logp = logits.log_softmax(0)
        objective, estimate = logits.sum() * 0, logits.sum() * 0
        for indices in itertools.product(range(3), repeat=3):
            indices = list(indices)
            probability = logp[indices].sum().exp()
            cost, advantage = passage_credit(values[indices], reference, 0.7)
            objective = objective - probability * cost
            estimate = (
                estimate
                + probability.detach() * (logp[indices] * torch.tensor(advantage) / 3).sum()
            )
        exact = torch.autograd.grad(objective, logits, retain_graph=True)[0]
        actual = torch.autograd.grad(estimate, logits)[0]
        torch.testing.assert_close(actual, exact, rtol=1e-10, atol=1e-10)
        p = logp.detach().exp().numpy()
        population = p @ kernel(values, values, 0.7) @ p + kernel(reference, reference, 0.7).mean()
        population -= 2 * p @ kernel(values, reference, 0.7).mean(1)
        self.assertAlmostEqual(-float(objective.detach()), population)

    def test_quality_and_mmd_are_conditioned_on_the_exact_question(self):
        target = dummy_target()
        rows = [
            {"case_id": q, "semantic_embedding": v.tolist()}
            for q, ref in target.references.items()
            for v in ref[[0, 1, 0, 1]]
        ]
        quality = np.array([0, 1, 2, 3, 10, 11, 12, 13], dtype=float)
        qa, da, _ = target.training_credit(rows, quality)
        np.testing.assert_allclose(qa[:4], qa[4:])
        self.assertAlmostEqual(float(da[:4].sum()), 0)
        self.assertAlmostEqual(float(da[4:].sum()), 0)
        rows[0]["case_id"] = "held-out"
        self.assertRaises(ValueError, target.training_credit, rows, quality)

    def test_four_prompts_produce_sixteen_answers_across_two_ranks(self):
        cfg = {"prompts_per_group": 4, "global_group_size": 16}
        cases = [{"id": str(i)} for i in range(4)]
        for rank in (0, 1):
            local, count = training_assignment(cases, cfg, rank, 2)
            self.assertEqual(len(local), 2)
            self.assertEqual(count, 4)

    def test_two_rank_credit_gathers_all_answers_before_comparison(self):
        with tempfile.TemporaryDirectory() as folder:
            mp.spawn(credit_worker, args=(folder,), nprocs=2, join=True)

    def test_prepared_references_preserve_splits_and_human_quality(self):
        data = ROOT / "data"
        if not (data / "manifest.json").exists():
            self.skipTest("Run prepare.py first")
        manifest = json.loads((data / "manifest.json").read_text())
        ids = manifest["prompt_ids"]
        for a, b in (("train", "validation"), ("train", "test"), ("validation", "test")):
            self.assertFalse(set(ids[a]) & set(ids[b]))
        for line in (data / "semantic-references.jsonl").read_text().splitlines():
            row = json.loads(line)
            self.assertEqual(len(set(row["responses"])), 2)
            self.assertIn(row["case_id"], ids[row["split"]])
            for attributes in row["human_attributes"]:
                self.assertGreaterEqual(min(attributes.values()), 3)
        metadata = json.loads((data / "semantic-metadata.json").read_text())
        self.assertEqual(set(metadata["training_prompt_ids"]), set(ids["train"]))


if __name__ == "__main__":
    unittest.main()
