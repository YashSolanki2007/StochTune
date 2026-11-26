"""Four real CPU ranks: fixed-batch credit, calibration coverage and gradient scaling."""

import json
import tempfile
import unittest
from pathlib import Path
from unittest.mock import patch

import numpy as np
import preflight_gpu
import run
import torch
import torch.distributed as dist
import torch.multiprocessing as mp
from calibrate import calibration_assignment
from common import make_distribution
from ddp_passage_rl import (
    SequencePolicy,
    configured_world_size,
    distributed_update,
    gather_objects,
)
from prepare_long import long_profile
from stage_runtime import stage
from test_ddp import tiny_actor
from torch.nn.parallel import DistributedDataParallel as DDP
from train_policy import global_credit, training_assignment


def fixed_rows():
    return [
        {
            "id": f"q{q}-{s}",
            "case_id": f"q{q}",
            "prompt_ids": [1, 4 + q],
            "completion_ids": [5 + 4 * q + s] + ([6] if s % 2 else []) + [2],
            "passage": ["a a", "b", "a c", "d c"][s],
        }
        for q in range(4)
        for s in range(4)
    ]


def actor():
    model = tiny_actor()
    with torch.no_grad():
        for name, value in model.named_parameters():
            if "lora_B" in name:
                value.normal_(std=0.03)
    return model


def four_rank_worker(rank, directory, world):
    torch.set_num_threads(1)
    root = Path(directory)
    dist.init_process_group("gloo", init_method=f"file://{root}/store", rank=rank, world_size=world)
    try:
        examples = [{"id": f"q{i}"} for i in range(4)]
        assignment, count = training_assignment(
            examples, {"prompts_per_group": 4, "global_group_size": 16}, rank, world
        )
        assert count == 4
        local_ids = {r["id"] for r in assignment}
        rows = [r for r in fixed_rows() if r["case_id"] in local_ids]
        assert len(rows) == 16 // world
        quality = np.array([0.1 + 0.2 * int(r["id"][-1]) for r in rows])
        target = make_distribution({"word_distribution": "full-vocabulary"}, ["a a b c d"])
        credit, metrics = global_credit(
            rows, quality, target, 0.1, torch.device("cpu"), quality_baseline="prompt-loo"
        )
        gathered = gather_objects(
            [{"id": r["id"], "credit": float(c)} for r, c in zip(rows, credit)]
        )
        calibration = [{"id": str(i)} for i in range(24)]
        assigned = calibration_assignment(calibration, rank, world)
        all_calibration = [r for part in gather_objects(assigned) for r in part]
        assert len(all_calibration) == len({r["id"] for r in all_calibration}) == 24
        model = actor()
        policy = DDP(SequencePolicy(model), broadcast_buffers=False)
        optimizer = torch.optim.SGD([p for p in model.parameters() if p.requires_grad], lr=0.03)
        result = distributed_update(
            policy, optimizer, rows, credit, 0, 1, 0.02, reference_kl_centering="prompt-loo"
        )
        torch.save(
            {n: p.detach().clone() for n, p in model.named_parameters() if p.requires_grad},
            root / f"rank-{rank}.pt",
        )
        if rank == 0:
            (root / "metrics.json").write_text(
                json.dumps(
                    {
                        "reward": metrics,
                        "update": result,
                        "credits": {r["id"]: r["credit"] for part in gathered for r in part},
                    },
                    allow_nan=False,
                )
            )
    finally:
        dist.destroy_process_group()


class FourRankTests(unittest.TestCase):
    def test_four_rank_update_matches_same_global_batch_on_two_ranks(self):
        results = {}
        with tempfile.TemporaryDirectory() as directory:
            for world in (2, 4):
                root = Path(directory) / str(world)
                root.mkdir()
                mp.spawn(four_rank_worker, args=(str(root), world), nprocs=world, join=True)
                initial = torch.load(root / "rank-0.pt", weights_only=True)
                for rank in range(1, world):
                    other = torch.load(root / f"rank-{rank}.pt", weights_only=True)
                    for name in initial:
                        torch.testing.assert_close(initial[name], other[name], rtol=0, atol=0)
                results[world] = (initial, json.loads((root / "metrics.json").read_text()))
            for name, value in results[2][0].items():
                torch.testing.assert_close(value, results[4][0][name], rtol=1e-5, atol=1e-7)
            self.assertEqual(set(results[2][1]["credits"]), set(results[4][1]["credits"]))
            for key, value in results[2][1]["credits"].items():
                self.assertAlmostEqual(value, results[4][1]["credits"][key], places=6)
            self.assertAlmostEqual(
                results[2][1]["reward"]["unigram_kl"],
                results[4][1]["reward"]["unigram_kl"],
                places=12,
            )

    def test_four_gpu_resource_caps_slurm_and_controller_are_consistent(self):
        cfg = long_profile(gpus=4)
        self.assertEqual(configured_world_size(cfg), 4)
        self.assertEqual(cfg["steps"] * cfg["global_group_size"], 8000)
        self.assertEqual(cfg["requested_gpus"] * cfg["requested_allocation_seconds"] / 3600, 12)
        self.assertFalse(cfg["gpu_submission_authorized"])
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            path = root / "config.json"
            path.write_text(json.dumps(cfg))
            runtime = root / "runtime"
            stage(runtime, path)
            text = (runtime / "policy.sbatch").read_text()
            for value in ["--gres=gpu:4", "--cpus-per-task=32", "--mem=256G", "--time=03:00:00"]:
                self.assertIn(value, text)
            with patch.object(preflight_gpu, "ROOT", runtime):
                report = preflight_gpu.preflight_gpu(require_authorization=False)
                self.assertEqual(report["gpus"], 4)
                self.assertEqual(report["gpu_wall_seconds"], 10800)
                self.assertRaises(PermissionError, run.run)

    def test_calibration_assignment_covers_each_example_once(self):
        rows = [{"id": i} for i in range(24)]
        for world in (2, 4):
            values = [r for rank in range(world) for r in calibration_assignment(rows, rank, world)]
            self.assertEqual(sorted(r["id"] for r in values), list(range(24)))
        for cfg in [{"distributed_world_size": 3}, {"distributed_world_size": True}]:
            self.assertRaises(ValueError, configured_world_size, cfg)


if __name__ == "__main__":
    unittest.main()
