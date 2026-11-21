"""Exact gradients, calibrated energy distance, and real tiny-model DDP updates."""

import contextlib
import hashlib
import io
import itertools
import json
import sys
import tempfile
import unittest
from pathlib import Path
from unittest.mock import patch

import numpy as np
import torch
import torch.distributed as dist
import torch.multiprocessing as mp
from torch.nn.parallel import DistributedDataParallel as DDP

ROOT = Path(__file__).resolve().parent
sys.path.insert(0, str(ROOT))
sys.path.insert(0, str(ROOT.parent / "helpsteer-rl-v2"))
from audit_kcgm_replay import calibration_scale, fixed_loss_divisor  # noqa: E402
from common import PassageDistribution  # noqa: E402, F401
from ddp_passage_rl import SequencePolicy, distributed_update, prompt_loo_baselines  # noqa: E402
from mmd import kernel, mmd2, passage_credit  # noqa: E402
from stage_runtime import stage  # noqa: E402
from test_ddp import tiny_actor  # noqa: E402

from stochastic_llm.rl import completion_log_probs  # noqa: E402


def changed_actor():
    actor = tiny_actor()
    torch.manual_seed(321)
    with torch.no_grad():
        for name, parameter in actor.named_parameters():
            if "lora_B" in name:
                parameter.copy_(torch.randn_like(parameter) * 0.1)
    return actor


def rows_for_centering():
    return [
        {"case_id": key, "prompt_ids": prompt, "completion_ids": completion}
        for key, prompt in (("q1", [1, 4]), ("q2", [1, 7]))
        for completion in ([5, 6, 2], [8, 2], [5, 9, 2], [2])
    ]


def update_worker(rank, folder):
    torch.set_num_threads(1)
    dist.init_process_group("gloo", init_method=f"file://{folder}/store", rank=rank, world_size=2)
    try:
        actor = changed_actor()
        policy = DDP(SequencePolicy(actor), broadcast_buffers=False)
        optimizer = torch.optim.SGD([p for p in actor.parameters() if p.requires_grad], lr=0.01)
        rows = rows_for_centering()[rank::2]
        credit = torch.linspace(-0.3, 0.4, 8)[rank::2]
        result = distributed_update(
            policy,
            optimizer,
            rows,
            credit,
            0,
            1,
            0.3,
            reference_kl_centering="prompt-loo",
            loss_divisor=1.502,
            record_log_probs=True,
        )
        torch.save(
            {
                "parameters": {
                    name: p.detach() for name, p in actor.named_parameters() if p.requires_grad
                },
                "metrics": result,
            },
            Path(folder) / f"rank-{rank}.pt",
        )
    finally:
        dist.destroy_process_group()


class KCGMTests(unittest.TestCase):
    def test_staging_preserves_frozen_calibration_without_authorizing_gpu(self):
        with tempfile.TemporaryDirectory() as folder:
            root = Path(folder)
            encoder = root / "encoder"
            encoder.mkdir()
            for name in (
                "config.json",
                "model.safetensors",
                "tokenizer.json",
                "tokenizer_config.json",
                "special_tokens_map.json",
                "vocab.txt",
                "sentence_bert_config.json",
            ):
                (encoder / name).write_text("test fixture")
            ready = root / "ready.json"
            ready.write_text("{}")
            for kernel_name in ("rbf", "energy"):
                output = root / kernel_name
                with patch("huggingface_hub.snapshot_download", return_value=str(encoder)):
                    with contextlib.redirect_stdout(io.StringIO()):
                        stage(output, ready, ROOT / f"kcgm-{kernel_name}-config.json")
                cfg = json.loads((output / "config.json").read_text())
                self.assertFalse(cfg["gpu_submission_authorized"])
                calibration = output / cfg["semantic_calibration_runtime_file"]
                self.assertEqual(
                    hashlib.sha256(calibration.read_bytes()).hexdigest(),
                    cfg["semantic_calibration_sha256"],
                )
                frozen = json.loads(calibration.read_text())
                self.assertEqual(cfg["semantic_kernel_scale"], frozen["kernel_scales"][kernel_name])

    def test_prompt_baseline_excludes_own_answer_and_other_questions(self):
        values = np.array([1.0, 3.0, 100.0, 104.0])
        ids = ["a", "a", "b", "b"]
        np.testing.assert_array_equal(prompt_loo_baselines(values, ids), [3, 1, 104, 100])
        changed = values.copy()
        changed[0] = 999
        self.assertEqual(prompt_loo_baselines(changed, ids)[0], 3)
        self.assertRaises(ValueError, prompt_loo_baselines, [1], ["single"])

    def test_centered_kl_has_exact_expected_gradient(self):
        logits = torch.tensor([0.2, -0.4, 0.5], dtype=torch.float64, requires_grad=True)
        logp = logits.log_softmax(0)
        ref = torch.tensor([0.5, 0.3, 0.2], dtype=torch.float64).log()
        target = (logp.exp() * (logp - ref)).sum()
        estimate = logits.sum() * 0
        for ids in itertools.product(range(3), repeat=3):
            ids = list(ids)
            probability = logp[ids].sum().exp().detach()
            delta = (logp[ids] - ref[ids]).detach()
            baseline = torch.tensor(prompt_loo_baselines(delta.numpy(), ["q"] * 3))
            estimate = estimate + probability * ((delta - baseline) * logp[ids]).mean()
        exact = torch.autograd.grad(target, logits, retain_graph=True)[0]
        actual = torch.autograd.grad(estimate, logits)[0]
        torch.testing.assert_close(actual, exact, rtol=1e-11, atol=1e-11)

    def test_energy_distance_and_its_exact_policy_gradient(self):
        values = np.array([[-1.0], [0.2], [1.0]])
        reference = np.array([[-1.0], [1.0]])
        np.testing.assert_array_equal(np.diag(kernel(values, values, 0, name="energy")), 0)
        self.assertAlmostEqual(
            mmd2(values[[0, 0, 2, 2]], reference, 0, unbiased=False, kernel_name="energy"), 0
        )
        logits = torch.tensor([0.2, -0.4, 0.5], dtype=torch.float64, requires_grad=True)
        logp = logits.log_softmax(0)
        expected_cost, gradient_surrogate = logits.sum() * 0, logits.sum() * 0
        for ids in itertools.product(range(3), repeat=3):
            ids = list(ids)
            probability = logp[ids].sum().exp()
            cost, advantage = passage_credit(
                values[ids], reference, 0, kernel_name="energy", kernel_scale=2.5
            )
            expected_cost = expected_cost + probability * cost
            gradient_surrogate = (
                gradient_surrogate
                - probability.detach() * (torch.tensor(advantage) * logp[ids]).mean()
            )
        exact = torch.autograd.grad(expected_cost, logits, retain_graph=True)[0]
        actual = torch.autograd.grad(gradient_surrogate, logits)[0]
        torch.testing.assert_close(actual, exact, rtol=1e-11, atol=1e-11)

    def test_frozen_kernel_scaling_and_objective_ratios(self):
        scale = calibration_scale([0.2, 0.4])
        self.assertAlmostEqual(scale * 0.3, 1)
        z = fixed_loss_divisor(1, 0.5, 0.002)
        self.assertAlmostEqual(z, 1.502)
        weights = np.array([1, 0.5, 0.002])
        np.testing.assert_allclose((weights / z) / (weights / z)[0], weights)
        self.assertRaises(ValueError, calibration_scale, [-1, 0])
        self.assertRaises(ValueError, fixed_loss_divisor, 1, -1, 0.002)

    def test_two_rank_centered_update_matches_global_gradient(self):
        torch.set_num_threads(1)
        actor, rows = changed_actor(), rows_for_centering()
        optimizer = torch.optim.SGD([p for p in actor.parameters() if p.requires_grad], lr=0.01)
        current = torch.cat([completion_log_probs(actor, [row], 0) for row in rows])
        with torch.no_grad(), actor.disable_adapter():
            reference = torch.cat([completion_log_probs(actor, [row], 0) for row in rows])
        delta = current.detach() - reference
        baseline = torch.tensor(prompt_loo_baselines(delta.numpy(), [r["case_id"] for r in rows]))
        credits = torch.linspace(-0.3, 0.4, 8)
        loss = -((credits - 0.3 * (delta - baseline)).detach() * current).mean() / 1.502
        loss.backward()
        parameters = [p for p in actor.parameters() if p.requires_grad]
        torch.nn.utils.clip_grad_norm_(parameters, 1)
        optimizer.step()
        expected = {
            name: p.detach().clone() for name, p in actor.named_parameters() if p.requires_grad
        }
        with tempfile.TemporaryDirectory() as folder:
            mp.spawn(update_worker, args=(folder,), nprocs=2, join=True)
            for rank in (0, 1):
                result = torch.load(Path(folder) / f"rank-{rank}.pt", weights_only=True)
                for name, value in result["parameters"].items():
                    torch.testing.assert_close(value, expected[name], rtol=2e-5, atol=2e-7)
                diagnostics = result["metrics"]["_row_metrics"]
                np.testing.assert_allclose(
                    [r["reference_loo_baseline"] for r in diagnostics],
                    baseline.numpy()[rank::2],
                    atol=2e-6,
                )
                self.assertEqual(len(diagnostics), 4)


if __name__ == "__main__":
    unittest.main()
