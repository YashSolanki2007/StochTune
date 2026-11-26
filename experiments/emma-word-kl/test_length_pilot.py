"""Pilot resource bounds, preserved schedule, and continuation compatibility."""

import json
import tempfile
import unittest
from pathlib import Path
from unittest.mock import patch

import preflight_gpu
from prepare_length_pilot import pilot_profile
from resume_policy import validate_resume_config
from stage_runtime import stage
from train_policy import training_step_limit


class LengthPilotTests(unittest.TestCase):
    def test_two_gpu_replacement_keeps_global_batch_and_reduces_allocation(self):
        cfg = pilot_profile(gpus=2)
        original = pilot_profile(gpus=4)
        for key in (
            "steps",
            "warmup_steps",
            "stop_after_steps",
            "global_group_size",
            "prompts_per_group",
            "samples_per_prompt",
            "length_penalty_weight",
        ):
            self.assertEqual(cfg.get(key), original.get(key))
        with tempfile.TemporaryDirectory() as folder:
            root = Path(folder)
            source = root / "config.json"
            source.write_text(json.dumps(cfg))
            runtime = root / "runtime"
            stage(runtime, source)
            script = (runtime / "policy.sbatch").read_text()
            for directive in (
                "--time=01:30:00",
                "--gres=gpu:2",
                "--cpus-per-task=16",
                "--mem=128G",
            ):
                self.assertIn(directive, script)
            with patch.object(preflight_gpu, "ROOT", runtime):
                report = preflight_gpu.preflight_gpu(False)
                self.assertEqual(report["gpus"], 2)
                self.assertEqual(report["gpu_wall_seconds"], 5400)
        self.assertRaises(ValueError, pilot_profile, 3)

    def test_stop_limit_keeps_full_schedule_and_rejects_invalid_limits(self):
        cfg = pilot_profile()
        self.assertEqual(training_step_limit(cfg), 100)
        self.assertEqual(cfg["steps"], 500)
        self.assertEqual(cfg["warmup_steps"], 50)
        self.assertEqual(cfg["global_group_size"] * training_step_limit(cfg), 1600)
        self.assertEqual(training_step_limit({"steps": 500}), 500)
        for limit in [0, -1, 501, True, 100.5]:
            self.assertRaises(ValueError, training_step_limit, {**cfg, "stop_after_steps": limit})

    def test_staged_budget_validation_only_and_approval_guard(self):
        cfg = pilot_profile()
        with tempfile.TemporaryDirectory() as folder:
            root = Path(folder)
            source = root / "config.json"
            source.write_text(json.dumps(cfg))
            runtime = root / "runtime"
            stage(runtime, source)
            script = (runtime / "policy.sbatch").read_text()
            self.assertIn("#SBATCH --time=01:30:00", script)
            self.assertIn("#SBATCH --gres=gpu:4", script)
            self.assertIn("#SBATCH --signal=B:USR1@1200", script)
            with patch.object(preflight_gpu, "ROOT", runtime):
                result = preflight_gpu.preflight_gpu(False)
                self.assertEqual(result["gpu_wall_seconds"], 5400)
                self.assertRaises(PermissionError, preflight_gpu.preflight_gpu)
                # Remove deployment hashes only for deliberate config-tampering checks.
                (runtime / "deployment-manifest.json").unlink()
                for change in [
                    {"stop_after_steps": 500},
                    {"requested_allocation_seconds": 10800},
                    {"final_evaluation_split": "test"},
                    {"warmup_steps": 10},
                ]:
                    (runtime / "run-config.json").write_text(json.dumps({**cfg, **change}))
                    self.assertRaises(AssertionError, preflight_gpu.preflight_gpu, False)

    def test_future_extension_can_change_stop_but_not_reward_or_schedule(self):
        source = pilot_profile()
        extension = {
            **source,
            "stop_after_steps": 500,
            "requested_allocation_seconds": 10800,
            "train_budget_seconds": 9000,
            "evaluation_reserve_seconds": 1800,
            "final_evaluation_split": "test",
        }
        validate_resume_config(extension, source)
        for change in [{"steps": 100}, {"length_penalty_weight": 0.4}, {"warmup_steps": 10}]:
            self.assertRaises(ValueError, validate_resume_config, {**extension, **change}, source)


if __name__ == "__main__":
    unittest.main()
