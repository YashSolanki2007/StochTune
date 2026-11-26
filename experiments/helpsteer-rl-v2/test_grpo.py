"""Exact expected gradients, clipping, real tiny-model updates and two-rank GRPO."""

import itertools
import json
import tempfile
import unittest
from pathlib import Path

import numpy as np
import torch
import torch.distributed as dist
import torch.multiprocessing as mp
from common import PassageDistribution, quality_advantages
from ddp_passage_rl import SequencePolicy, distributed_update, save_checkpoint
from grpo import distributed_grpo_update, token_surrogate, validate_update_config
from resume_policy import restore_policy_checkpoint, validate_resume_checkpoint
from test_ddp import tiny_actor
from torch.nn.parallel import DistributedDataParallel as DDP
from train_policy import global_credit

from stochastic_llm.rl import completion_log_probs, completion_token_log_probs


def rows_and_quality():
    rows = [
        {"case_id": "a", "prompt_ids": [1, 4], "completion_ids": [5, 6, 2], "passage": "a a"},
        {"case_id": "b", "prompt_ids": [1, 7], "completion_ids": [8, 2], "passage": "b"},
        {"case_id": "a", "prompt_ids": [1, 4], "completion_ids": [9, 2], "passage": "a c"},
        {"case_id": "b", "prompt_ids": [1, 7], "completion_ids": [2], "passage": "c c"},
    ]
    return rows, np.array([0.2, 0.7, 0.9, 0.3])


def nonzero_actor():
    actor = tiny_actor()
    # Exercise the reference penalty away from its initial zero-log-ratio state.
    with torch.no_grad():
        for name, parameter in actor.named_parameters():
            if "lora_B" in name:
                parameter.normal_(mean=0.0, std=0.05)
    return actor


def trainable_values(actor):
    return {n: p.detach().clone() for n, p in actor.named_parameters() if p.requires_grad}


def multi_rank_worker(rank, folder):
    torch.set_num_threads(1)
    dist.init_process_group("gloo", init_method=f"file://{folder}/store", rank=rank, world_size=2)
    try:
        actor = nonzero_actor()
        policy = DDP(SequencePolicy(actor), broadcast_buffers=False)
        rows, quality = rows_and_quality()
        target = PassageDistribution(["a a b c"], order=1)
        local = rows[rank * 2 : rank * 2 + 2]
        credit, metrics = global_credit(
            local,
            quality[rank * 2 : rank * 2 + 2],
            target,
            0.1,
            torch.device("cpu"),
            quality_baseline="prompt-loo",
        )
        # The same-prompt pairs are split across ranks deliberately.
        expected_quality = np.array([-0.7, 0.4, 0.7, -0.4])
        expected_credit = (
            expected_quality + 0.1 * target.credit([r["passage"] for r in rows])["advantages"]
        )
        np.testing.assert_allclose(
            credit.numpy(), expected_credit[rank * 2 : rank * 2 + 2], atol=1e-7
        )
        optimizer = torch.optim.SGD([p for p in actor.parameters() if p.requires_grad], lr=0.2)
        scheduler = torch.optim.lr_scheduler.LambdaLR(optimizer, lambda _: 1.0)
        result = distributed_grpo_update(
            policy,
            optimizer,
            local,
            credit,
            0,
            1,
            0.02,
            passes=2,
            clip_epsilon=0.2,
            reference_kl_centering="prompt-loo",
            scheduler=scheduler,
            record_log_probs=True,
        )
        assert result["grpo_pass_1_task_clip_fraction"] == 0.0
        assert result["grpo_pass_2_old_policy_log_ratio_rms"] > 0.0
        assert scheduler.last_epoch == 2
        torch.save(trainable_values(actor), Path(folder) / f"rank-{rank}.pt")
        (Path(folder) / f"metrics-{rank}.json").write_text(json.dumps(result, allow_nan=False))
    finally:
        dist.destroy_process_group()


class GRPOTests(unittest.TestCase):
    def test_tokens_include_eos_without_prompt_or_padding(self):
        actor = nonzero_actor()
        rows, _ = rows_and_quality()
        values = completion_token_log_probs(actor, rows, 0)
        self.assertEqual(len(values), sum(len(r["completion_ids"]) for r in rows))
        summed = torch.stack(
            [p.sum() for p in values.split([len(r["completion_ids"]) for r in rows])]
        )
        torch.testing.assert_close(summed, completion_log_probs(actor, rows, 0))
        for row, packed in zip(rows, values.split([len(r["completion_ids"]) for r in rows])):
            inputs = torch.tensor([row["prompt_ids"] + row["completion_ids"]])
            scores = actor(input_ids=inputs).logits.float().log_softmax(-1)
            start = len(row["prompt_ids"]) - 1
            expected = scores[0, start:-1].gather(1, inputs[0, start + 1 :, None]).squeeze(1)
            torch.testing.assert_close(packed, expected)

    def test_clipping_blocks_only_the_favored_direction_and_detaches_snapshot(self):
        current = torch.tensor([1.5, 0.5, 0.9, 1.1], dtype=torch.float64).log().requires_grad_()
        old = torch.zeros(4, dtype=torch.float64, requires_grad=True)
        advantage = torch.tensor([1.0, -1.0, 1.0, -1.0], dtype=torch.float64, requires_grad=True)
        kl = torch.zeros_like(advantage, requires_grad=True)
        loss, ratio, active = token_surrogate(current, old, advantage, kl, 0.0, 0.2)
        loss.sum().backward()
        torch.testing.assert_close(
            current.grad, torch.tensor([0.0, 0.0, -0.9, 1.1], dtype=torch.float64)
        )
        self.assertEqual(active.tolist(), [True, True, False, False])
        self.assertIsNone(old.grad)
        self.assertIsNone(advantage.grad)
        self.assertIsNone(kl.grad)

    def test_exact_expected_coupled_distribution_gradient(self):
        logits = torch.tensor([0.2, -0.1], dtype=torch.float64, requires_grad=True)
        probabilities = logits.softmax(0)
        target = PassageDistribution(["a a b"], order=1)
        exact_loss = 0.0
        surrogate = 0.0
        for actions in itertools.product(range(2), repeat=3):
            choices = torch.tensor(actions)
            mass = probabilities[choices].prod()
            quality = np.array([0.2 if a == 0 else 0.8 for a in actions])
            fidelity = target.credit(["a" if a == 0 else "b" for a in actions])
            task = torch.tensor(
                quality_advantages(quality) + 0.1 * fidelity["advantages"], dtype=torch.float64
            )
            exact_loss = exact_loss + mass * (-quality.mean() + 0.1 * fidelity["kl"])
            current = probabilities[choices].log()
            losses, _, _ = token_surrogate(
                current, current.detach(), task, torch.zeros_like(task), 0.0, 0.2
            )
            surrogate = surrogate + mass.detach() * losses.mean()
        expected = torch.autograd.grad(exact_loss, logits, retain_graph=True)[0]
        actual = torch.autograd.grad(surrogate, logits)[0]
        torch.testing.assert_close(actual, expected, atol=1e-10, rtol=1e-9)
        self.assertGreater(expected.norm().item(), 0.0)

    def test_first_real_update_matches_reinforce_with_centered_reference_penalty(self):
        torch.set_num_threads(1)
        with tempfile.TemporaryDirectory() as folder:
            dist.init_process_group(
                "gloo", init_method=f"file://{folder}/store", rank=0, world_size=1
            )
            try:
                rows, _ = rows_and_quality()
                credit = torch.tensor([0.1, -0.2, 0.3, -0.1])
                actors = [nonzero_actor(), nonzero_actor()]
                outputs = []
                for actor, method in zip(actors, ["reinforce", "grpo"]):
                    policy = DDP(SequencePolicy(actor), broadcast_buffers=False)
                    optimizer = torch.optim.SGD(
                        [p for p in actor.parameters() if p.requires_grad], lr=0.02
                    )
                    function = (
                        distributed_update if method == "reinforce" else distributed_grpo_update
                    )
                    extra = {} if method == "reinforce" else {"passes": 1}
                    metrics = function(
                        policy,
                        optimizer,
                        rows,
                        credit,
                        0,
                        1,
                        0.2,
                        reference_kl_centering="prompt-loo",
                        loss_divisor=1.502,
                        **extra,
                    )
                    outputs.append(trainable_values(actor))
                for name in outputs[0]:
                    torch.testing.assert_close(
                        outputs[0][name], outputs[1][name], rtol=1e-5, atol=1e-7
                    )
                self.assertGreater(metrics["reference_kl_credit_rms_raw"], 0.0)
                self.assertEqual(metrics["grpo_pass_1_task_clip_fraction"], 0.0)
            finally:
                dist.destroy_process_group()

    def test_two_pass_two_rank_update_matches_single_rank_global_update(self):
        with tempfile.TemporaryDirectory() as folder:
            mp.spawn(multi_rank_worker, args=(folder,), nprocs=2, join=True)
            distributed = torch.load(Path(folder) / "rank-0.pt", weights_only=True)
            other = torch.load(Path(folder) / "rank-1.pt", weights_only=True)
            for name in distributed:
                torch.testing.assert_close(distributed[name], other[name], rtol=0, atol=0)
            dist.init_process_group(
                "gloo", init_method=f"file://{folder}/single", rank=0, world_size=1
            )
            try:
                actor = nonzero_actor()
                policy = DDP(SequencePolicy(actor), broadcast_buffers=False)
                rows, quality = rows_and_quality()
                target = PassageDistribution(["a a b c"], order=1)
                credit, _ = global_credit(
                    rows, quality, target, 0.1, torch.device("cpu"), quality_baseline="prompt-loo"
                )
                optimizer = torch.optim.SGD(
                    [p for p in actor.parameters() if p.requires_grad], lr=0.2
                )
                scheduler = torch.optim.lr_scheduler.LambdaLR(optimizer, lambda _: 1.0)
                metrics = distributed_grpo_update(
                    policy,
                    optimizer,
                    rows,
                    credit,
                    0,
                    1,
                    0.02,
                    passes=2,
                    clip_epsilon=0.2,
                    reference_kl_centering="prompt-loo",
                    scheduler=scheduler,
                )
                for name, value in trainable_values(actor).items():
                    torch.testing.assert_close(value, distributed[name], rtol=1e-5, atol=1e-7)
                self.assertEqual(scheduler.last_epoch, 2)
                self.assertEqual(metrics["optimizer_steps_this_batch"], 2)
            finally:
                dist.destroy_process_group()

    def test_normalization_and_invalid_settings_are_rejected(self):
        for extra in [
            {"grpo_passes": 0},
            {"grpo_passes": True},
            {"grpo_clip_epsilon": float("nan")},
            {"advantage_normalization": "group-std"},
            {"grpo_loss_normalization": "answer-length"},
        ]:
            with self.assertRaises(ValueError):
                validate_update_config({"policy_update": "grpo", **extra})
        self.assertEqual(validate_update_config({}), "reinforce")
        with self.assertRaises(FloatingPointError):
            token_surrogate(
                torch.tensor([1000.0]), torch.zeros(1), torch.ones(1), torch.zeros(1), 0.0, 0.2
            )

    def test_resume_restores_two_pass_optimizer_scheduler_and_weights(self):
        import hashlib
        import random
        from types import SimpleNamespace

        with tempfile.TemporaryDirectory() as folder:
            root = Path(folder)
            dist.init_process_group(
                "gloo", init_method=f"file://{root}/store", rank=0, world_size=1
            )
            try:
                cfg = {"policy_update": "grpo", "grpo_passes": 2, "steps": 3, "seed": 42}
                source = root / "source.json"
                source.write_text(json.dumps(cfg))
                actor = nonzero_actor()
                policy = DDP(SequencePolicy(actor), broadcast_buffers=False)
                optimizer = torch.optim.AdamW(
                    [p for p in actor.parameters() if p.requires_grad], lr=0.001
                )
                scheduler = torch.optim.lr_scheduler.LambdaLR(optimizer, lambda _: 1.0)
                rows, _ = rows_and_quality()
                credit = torch.tensor([0.1, -0.2, 0.3, -0.1])
                prompt_rng = random.Random(42)

                def update(current_policy, current_optimizer, current_scheduler):
                    distributed_grpo_update(
                        current_policy,
                        current_optimizer,
                        rows,
                        credit,
                        0,
                        1,
                        0.02,
                        passes=2,
                        reference_kl_centering="prompt-loo",
                        scheduler=current_scheduler,
                    )

                update(policy, optimizer, scheduler)
                checkpoint = save_checkpoint(
                    policy,
                    SimpleNamespace(save_pretrained=lambda _: None),
                    optimizer,
                    scheduler,
                    prompt_rng,
                    1,
                    root,
                    hashlib.sha256(json.dumps(cfg, sort_keys=True).encode()).hexdigest(),
                    {"arm": "quality-unigram"},
                    torch.device("cpu"),
                )
                continuation = {
                    **cfg,
                    "resume_checkpoint": str(checkpoint),
                    "resume_step": 1,
                    "resume_source_config": str(source),
                    "resume_source_config_sha256": hashlib.sha256(source.read_bytes()).hexdigest(),
                    "resume_checkpoint_sha256": {
                        str(p.relative_to(checkpoint)): hashlib.sha256(p.read_bytes()).hexdigest()
                        for p in checkpoint.rglob("*")
                        if p.is_file()
                    },
                }
                state = validate_resume_checkpoint(continuation, "quality-unigram", world=1)
                self.assertEqual(state["scheduler"]["last_epoch"], 2)
                update(policy, optimizer, scheduler)
                expected = trainable_values(actor)
                restored = nonzero_actor()
                restored_policy = DDP(SequencePolicy(restored), broadcast_buffers=False)
                restored_optimizer = torch.optim.AdamW(
                    [p for p in restored.parameters() if p.requires_grad], lr=0.001
                )
                restored_scheduler = torch.optim.lr_scheduler.LambdaLR(
                    restored_optimizer, lambda _: 1.0
                )
                restore_policy_checkpoint(
                    restored,
                    restored_optimizer,
                    restored_scheduler,
                    random.Random(0),
                    state,
                    checkpoint,
                    0,
                    torch.device("cpu"),
                )
                update(restored_policy, restored_optimizer, restored_scheduler)
                for name, value in trainable_values(restored).items():
                    torch.testing.assert_close(value, expected[name], rtol=1e-6, atol=1e-8)
                self.assertEqual(restored_scheduler.last_epoch, 4)
            finally:
                dist.destroy_process_group()


if __name__ == "__main__":
    unittest.main()
