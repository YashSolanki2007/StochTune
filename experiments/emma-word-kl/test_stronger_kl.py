"""Check the exact single-run weights, full LR schedule and approval boundary."""

import json
import tempfile
import unittest
from pathlib import Path
from unittest.mock import patch

import preflight_gpu
import run
from prepare_stronger_kl import stronger_kl_profile
from stage_runtime import stage
from train_policy import training_step_limit


class StrongerKLTests(unittest.TestCase):
    def test_single_run_profile_and_staged_approval_guard(self):
        cfg = stronger_kl_profile()
        self.assertEqual(training_step_limit(cfg), 1000)
        self.assertEqual(cfg["steps"] * cfg["global_group_size"], 16000)
        self.assertEqual(cfg["warmup_steps"], 50)
        self.assertEqual(cfg["distribution_weight"], 0.5)
        self.assertEqual(cfg["length_penalty_weight"], 0.2)
        self.assertEqual(cfg["reference_kl_beta"], 0.002)
        self.assertEqual(cfg["truncation_penalty"], 0.1)
        self.assertEqual(cfg["validation_prompt_count"], 8)
        self.assertNotIn("resume_checkpoint", cfg)
        self.assertFalse(any(k.startswith("search_") for k in cfg))
        with tempfile.TemporaryDirectory() as folder:
            root = Path(folder)
            source = root / "config.json"
            source.write_text(json.dumps(cfg))
            runtime = root / "runtime"
            stage(runtime, source)
            script = (runtime / "policy.sbatch").read_text()
            self.assertIn("#SBATCH --time=05:00:00", script)
            self.assertIn("#SBATCH --gres=gpu:4", script)
            with patch.object(preflight_gpu, "ROOT", runtime):
                result = preflight_gpu.preflight_gpu(False)
                self.assertEqual(result["gpu_wall_seconds"], 18000)
                self.assertEqual(result["gpus"], 4)
                with patch.object(run.subprocess, "Popen") as launch:
                    self.assertRaises(PermissionError, run.run)
                    launch.assert_not_called()
                (runtime / "deployment-manifest.json").unlink()
                for change in [
                    {"steps": 500},
                    {"distribution_weight": 0.1},
                    {"distribution_weight": 0.3},
                    {"length_penalty_weight": 0.4},
                    {"reference_kl_beta": 0.02},
                    {"requested_allocation_seconds": 21600},
                ]:
                    (runtime / "run-config.json").write_text(json.dumps({**cfg, **change}))
                    self.assertRaises(AssertionError, preflight_gpu.preflight_gpu, False)


if __name__ == "__main__":
    unittest.main()
