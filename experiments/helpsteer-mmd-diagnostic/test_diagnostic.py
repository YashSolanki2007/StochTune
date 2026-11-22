"""Zero-quality objective, split guards, and actual 64-answer two-rank updates."""

import json
import shutil
import sys
import tempfile
import unittest
from pathlib import Path

import numpy as np
import torch
import torch.distributed as dist
import torch.multiprocessing as mp
from torch.nn.parallel import DistributedDataParallel as DDP

ROOT = Path(__file__).resolve().parent
sys.path.insert(1, str(ROOT.parent / "helpsteer-rl-v2"))
sys.path.insert(1, str(ROOT.parent / "helpsteer-semantic-mmd"))
sys.path.insert(0, str(ROOT))
from common import PassageDistribution  # noqa: E402,F401
from ddp_passage_rl import SequencePolicy, distributed_update, prompt_loo_baselines  # noqa: E402
from diagnostic import select_cases, select_questions, zero_quality  # noqa: E402
from mmd import conditional_credit  # noqa: E402
from preflight import check  # noqa: E402
from semantic_runtime import ConditionalSemanticTarget  # noqa: E402
from test_ddp import tiny_actor  # noqa: E402
from train_policy import global_credit  # noqa: E402

from stochastic_llm.rl import completion_log_probs  # noqa: E402


def actor_for_test():
    actor = tiny_actor()
    with torch.no_grad():
        for name, p in actor.named_parameters():
            if "lora_B" in name:
                p.fill_(0.1)
    return actor


def test_records():
    return [
        {
            "case_id": f"q{q}",
            "prompt_ids": [1, q + 3],
            "completion_ids": [j % 4 + 8, 2],
            "passage": "irrelevant",
            "quality_reward": 1000000 * (j + 1),
            "semantic_embedding": [float(j / 20), float(q / 4)],
        }
        for q in range(4)
        for j in range(16)
    ]


def target_for_test():
    target = ConditionalSemanticTarget.__new__(ConditionalSemanticTarget)
    target.references = {f"q{q}": np.array([[0, q / 4], [1, q / 4]]) for q in range(4)}
    target.train_ids = set(target.references)
    target.bandwidth_squared = 0.2786
    target.kernel_name = "energy"
    target.kernel_scale = 1.7710087347953098
    return target


def worker(rank, folder):
    torch.set_num_threads(1)
    dist.init_process_group("gloo", init_method=f"file://{folder}/store", rank=rank, world_size=2)
    try:
        rows = [r for r in test_records() if int(r["case_id"][1:]) % 2 == rank]
        policy = DDP(SequencePolicy(actor_for_test()), broadcast_buffers=False)
        optimizer = torch.optim.SGD([p for p in policy.parameters() if p.requires_grad], lr=0.01)
        credit, metrics = global_credit(
            rows, zero_quality(rows), target_for_test(), 0.5, torch.device("cpu")
        )
        update = distributed_update(
            policy,
            optimizer,
            rows,
            credit,
            0,
            1,
            0.002,
            reference_kl_centering="prompt-loo",
            loss_divisor=1.502,
        )
        torch.save(
            {
                "credit": credit,
                "metrics": metrics,
                "update": update,
                "parameters": {
                    name: p.detach()
                    for name, p in policy.module.actor.named_parameters()
                    if p.requires_grad
                },
            },
            Path(folder) / f"rank-{rank}.pt",
        )
    finally:
        dist.destroy_process_group()


class DiagnosticTests(unittest.TestCase):
    def test_selection_is_training_only_reproducible_and_score_independent(self):
        data = ROOT.parent / "helpsteer-semantic-mmd/data"
        refs = [
            json.loads(s) for s in (data / "semantic-references.jsonl").read_text().splitlines()
        ]
        selection = select_questions(refs)
        self.assertEqual(selection, select_questions(list(reversed(refs))))
        self.assertEqual(len(selection["seen_ids"]), 4)
        train = {r["case_id"] for r in refs if r["split"] == "train"}
        self.assertLessEqual(set(selection["seen_ids"]), train)
        self.assertFalse(set(selection["unseen_ids"]) & train)
        for r in refs:
            r["model_reward"] = 9999
        self.assertEqual(selection, select_questions(refs))
        with self.assertRaises(ValueError):
            select_cases([{"id": "a"}], ["b"])

    def test_preflight_rejects_quality_and_excess_budget(self):
        cfg = json.loads((ROOT / "config.json").read_text())
        source = ROOT.parent / "helpsteer-semantic-mmd/data"
        with tempfile.TemporaryDirectory() as folder:
            data = Path(folder) / "data"
            shutil.copytree(source, data)
            refs = [
                json.loads(s) for s in (data / "semantic-references.jsonl").read_text().splitlines()
            ]
            (data / "diagnostic-selection.json").write_text(json.dumps(select_questions(refs)))
            calibration = ROOT.parents[1] / cfg["semantic_calibration_report"]
            shutil.copy2(calibration, data / "semantic-calibration.json")
            self.assertEqual(check(cfg, data)["quality_weight"], 0)
            for changes in (
                {"quality_weight": 1},
                {"approved_allocation_seconds": 7200},
                {"train_budget_seconds": 5400},
                {"samples_per_question": 4},
            ):
                with self.assertRaises(ValueError):
                    check({**cfg, **changes}, data)
            selection = select_questions(refs)
            selection["seen_ids"][0] = selection["unseen_ids"][0]
            (data / "diagnostic-selection.json").write_text(json.dumps(selection))
            with self.assertRaises(ValueError):
                check(cfg, data)

    def test_two_rank_update_has_no_quality_and_matches_global_gradient(self):
        torch.set_num_threads(1)
        rows = test_records()
        target = target_for_test()
        qa, da, _ = conditional_credit(
            rows,
            zero_quality(rows),
            target.references,
            target.bandwidth_squared,
            kernel_name="energy",
            kernel_scale=target.kernel_scale,
        )
        np.testing.assert_array_equal(qa, np.zeros(64))
        actor = actor_for_test()
        optimizer = torch.optim.SGD([p for p in actor.parameters() if p.requires_grad], lr=0.01)
        current = torch.cat([completion_log_probs(actor, [r], 0) for r in rows])
        with torch.no_grad(), actor.disable_adapter():
            base = torch.cat([completion_log_probs(actor, [r], 0) for r in rows])
        delta = current.detach() - base
        baseline = torch.tensor(prompt_loo_baselines(delta.numpy(), [r["case_id"] for r in rows]))
        coefficient = torch.tensor(0.5 * da) - 0.002 * (delta - baseline)
        loss = -(coefficient.detach() * current).mean() / 1.502
        loss.backward()
        torch.nn.utils.clip_grad_norm_([p for p in actor.parameters() if p.requires_grad], 1)
        optimizer.step()
        expected = {
            name: p.detach().clone() for name, p in actor.named_parameters() if p.requires_grad
        }
        with tempfile.TemporaryDirectory() as folder:
            mp.spawn(worker, args=(folder,), nprocs=2, join=True)
            for rank in (0, 1):
                result = torch.load(Path(folder) / f"rank-{rank}.pt", weights_only=True)
                ids = [i for i, r in enumerate(rows) if int(r["case_id"][1:]) % 2 == rank]
                np.testing.assert_allclose(result["credit"].numpy(), (0.5 * da)[ids], atol=1e-7)
                self.assertEqual(result["metrics"]["quality_advantage_rms"], 0)
                self.assertEqual(result["metrics"]["quality_reward"], 0)
                self.assertEqual(result["metrics"]["answers_per_question"], 16)
                self.assertGreater(result["update"]["reference_kl_credit_rms_raw"], 0)
                for name, value in result["parameters"].items():
                    torch.testing.assert_close(value, expected[name], rtol=2e-5, atol=2e-7)


if __name__ == "__main__":
    unittest.main()
