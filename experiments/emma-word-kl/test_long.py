"""Keep the longer experiment's schedule, resource cap and approval guard aligned."""

import json
import tempfile
import unittest
from pathlib import Path
from unittest.mock import patch

import preflight_gpu
import run
from prepare_long import long_profile
from stage_runtime import stage


class LongRunTests(unittest.TestCase):
    def test_profile_uses_fresh_500_step_schedule_and_has_no_spending_approval(self):
        cfg = long_profile()
        self.assertEqual(cfg["policy_update"], "reinforce")
        self.assertEqual(cfg["steps"] * cfg["global_group_size"], 8000)
        self.assertEqual(cfg["warmup_steps"], 50)
        self.assertEqual(cfg["quality_baseline"], "prompt-loo")
        self.assertNotIn("resume_checkpoint", cfg)
        self.assertNotIn("grpo_passes", cfg)
        self.assertFalse(cfg["gpu_submission_authorized"])
        self.assertEqual(cfg["approved_maximum_gpu_hours"], 0)
        limits = preflight_gpu.allocation_limits(cfg)
        self.assertEqual(limits["allocation_seconds"], 14400)
        self.assertEqual(limits["evaluation_reserve_seconds"], 1800)

    def test_staged_slurm_controller_preflight_and_approval_guard_agree(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            cfg_path = root / "long.json"
            cfg_path.write_text(json.dumps(long_profile()))
            runtime = root / "runtime"
            stage(runtime, cfg_path)
            text = (runtime / "policy.sbatch").read_text()
            self.assertIn("#SBATCH --time=04:00:00", text)
            self.assertIn("#SBATCH --signal=B:USR1@1800", text)
            self.assertIn("#SBATCH --job-name=emma-reinforce-long-v1", text)
            with patch.object(preflight_gpu, "ROOT", runtime):
                report = preflight_gpu.preflight_gpu(require_authorization=False)
                self.assertEqual(report["gpu_wall_seconds"], 14400)
                self.assertFalse(report["gpu_submission_authorized"])
                with patch.object(run.subprocess, "Popen") as launch:
                    self.assertRaises(PermissionError, run.run)
                    launch.assert_not_called()
            cfg = long_profile()
            cfg["requested_allocation_seconds"] = 18000
            (runtime / "run-config.json").write_text(json.dumps(cfg))
            with patch.object(preflight_gpu, "ROOT", runtime):
                self.assertRaises(AssertionError, preflight_gpu.preflight_gpu, False)


if __name__ == "__main__":
    unittest.main()
