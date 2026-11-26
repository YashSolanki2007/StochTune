"""Match optimizer controls while preventing stale allocation authorization."""

import tempfile
import unittest
from pathlib import Path

from prepare_comparison import profiles
from stage_runtime import stage


class ComparisonTests(unittest.TestCase):
    def test_profiles_match_science_and_both_require_new_authorization(self):
        pair = profiles()
        for cfg in pair.values():
            self.assertFalse(cfg["gpu_submission_authorized"])
            self.assertEqual(cfg["approved_allocation_seconds"], 0)
            self.assertEqual(cfg["approved_maximum_gpu_hours"], 0)
            self.assertNotIn("previous_cancelled_job", cfg)
            self.assertNotIn("resume_checkpoint", cfg)
            self.assertEqual(cfg["quality_baseline"], "prompt-loo")
            self.assertEqual(cfg["advantage_normalization"], "none")
            self.assertEqual(cfg["steps"] * cfg["global_group_size"], 1280)
        differences = {
            key for key in pair["reinforce"] if pair["reinforce"][key] != pair["grpo"][key]
        }
        self.assertEqual(differences, {"policy_update", "experiment_profile"})

    def test_staging_rejects_inherited_approval_before_creating_runtime(self):
        import json

        with tempfile.TemporaryDirectory() as folder:
            root = Path(folder)
            path = root / "config.json"
            path.write_text(json.dumps({**profiles()["grpo"], "gpu_submission_authorized": True}))
            with self.assertRaises(ValueError):
                stage(root / "runtime", path)
            self.assertFalse((root / "runtime").exists())


if __name__ == "__main__":
    unittest.main()
