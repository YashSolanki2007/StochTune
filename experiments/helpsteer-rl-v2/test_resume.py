"""Verify a real adapter/optimizer/scheduler/RNG continuation before paid training."""

import hashlib
import json
import random
import tempfile
import unittest
from pathlib import Path

import torch
import torch.distributed as dist
from common import PassageDistribution  # noqa: F401
from ddp_passage_rl import SequencePolicy, distributed_update, save_checkpoint
from resume_policy import (
    restore_policy_checkpoint,
    validate_resume_checkpoint,
    validate_resume_config,
)
from test_ddp import tiny_actor
from torch.nn.parallel import DistributedDataParallel as DDP
from transformers import get_cosine_schedule_with_warmup


class ResumeTests(unittest.TestCase):
    def test_cannot_change_reward_or_schedule_when_extending_wall_time(self):
        source = {"steps": 200, "distribution_weight": 0.5, "train_budget_seconds": 2700}
        cfg = {**source, "train_budget_seconds": 13200, "resume_checkpoint": "/checkpoint"}
        validate_resume_config(cfg, source)
        for key, value in (("steps", 400), ("distribution_weight", 1.0)):
            with self.assertRaises(ValueError):
                validate_resume_config({**cfg, key: value}, source)

    def test_restored_next_update_and_random_streams_match_uninterrupted_training(self):
        torch.set_num_threads(1)
        with tempfile.TemporaryDirectory() as folder:
            root = Path(folder)
            dist.init_process_group(
                "gloo", init_method=f"file://{root}/store", rank=0, world_size=1
            )
            try:
                actor = tiny_actor()
                policy = DDP(SequencePolicy(actor), broadcast_buffers=False)
                params = [p for p in actor.parameters() if p.requires_grad]
                optimizer = torch.optim.AdamW(params, lr=0.001)
                scheduler = get_cosine_schedule_with_warmup(optimizer, 0, 4)
                prompt_rng = random.Random(333)
                rows = [
                    {"case_id": "q", "prompt_ids": [1, 4], "completion_ids": [i, 2]}
                    for i in (5, 6, 7, 8)
                ]
                credit = torch.tensor([0.1, -0.2, 0.3, -0.1])

                def update(current_policy, current_optimizer, current_scheduler):
                    distributed_update(
                        current_policy,
                        current_optimizer,
                        rows,
                        credit,
                        0,
                        1,
                        0.002,
                        reference_kl_centering="prompt-loo",
                        loss_divisor=1.502,
                    )
                    current_scheduler.step()

                update(policy, optimizer, scheduler)
                torch.manual_seed(777)
                random.seed(555)
                source = {"steps": 4, "seed": 42, "distribution_weight": 0.5}
                source_file = root / "source.json"
                source_file.write_text(json.dumps(source))
                checkpoint = save_checkpoint(
                    policy,
                    NoneTokenizer(),
                    optimizer,
                    scheduler,
                    prompt_rng,
                    1,
                    root,
                    hashlib.sha256(json.dumps(source, sort_keys=True).encode()).hexdigest(),
                    {"arm": "quality-semantic-mmd"},
                    torch.device("cpu"),
                )
                expected_torch = torch.rand(3)
                expected_python = random.random()
                expected_prompt = prompt_rng.sample(range(100), 4)
                update(policy, optimizer, scheduler)
                expected = {
                    name: p.detach().clone()
                    for name, p in actor.named_parameters()
                    if p.requires_grad
                }
                cfg = {
                    **source,
                    "resume_checkpoint": str(checkpoint),
                    "resume_step": 1,
                    "resume_source_config": str(source_file),
                    "resume_source_config_sha256": hashlib.sha256(
                        source_file.read_bytes()
                    ).hexdigest(),
                    "resume_checkpoint_sha256": {
                        str(p.relative_to(checkpoint)): hashlib.sha256(p.read_bytes()).hexdigest()
                        for p in checkpoint.rglob("*")
                        if p.is_file()
                    },
                }
                state = validate_resume_checkpoint(cfg, "quality-semantic-mmd", world=1)
                restored = tiny_actor()
                restored_policy = DDP(SequencePolicy(restored), broadcast_buffers=False)
                restored_optimizer = torch.optim.AdamW(
                    [p for p in restored.parameters() if p.requires_grad], lr=0.001
                )
                restored_scheduler = get_cosine_schedule_with_warmup(restored_optimizer, 0, 4)
                restored_rng = random.Random(999)
                step = restore_policy_checkpoint(
                    restored,
                    restored_optimizer,
                    restored_scheduler,
                    restored_rng,
                    state,
                    checkpoint,
                    0,
                    torch.device("cpu"),
                )
                self.assertEqual(step, 1)
                torch.testing.assert_close(torch.rand(3), expected_torch)
                self.assertEqual(random.random(), expected_python)
                self.assertEqual(restored_rng.sample(range(100), 4), expected_prompt)
                update(restored_policy, restored_optimizer, restored_scheduler)
                for name, p in restored.named_parameters():
                    if p.requires_grad:
                        torch.testing.assert_close(p, expected[name], rtol=1e-6, atol=1e-8)
                self.assertEqual(scheduler.state_dict(), restored_scheduler.state_dict())
                with self.assertRaises(ValueError):
                    validate_resume_checkpoint(
                        {**cfg, "distribution_weight": 1.0}, "quality-semantic-mmd", world=1
                    )
            finally:
                dist.destroy_process_group()


class NoneTokenizer:
    def save_pretrained(self, path):
        pass


if __name__ == "__main__":
    unittest.main()
