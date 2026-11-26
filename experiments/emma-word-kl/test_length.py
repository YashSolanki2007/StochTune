"""Fixed-band rewards, score separation, certificate binding and real global credits."""

import json
import tempfile
import unittest
from pathlib import Path
from unittest.mock import patch

import common
import numpy as np
import preflight_gpu
import torch
import torch.distributed as dist
import train_policy
from calibrate import write_certificate
from calibrate_length import calibrate
from common import FrozenPreferenceReward, config, make_distribution
from length_reward import apply_penalty, settings
from prepare_length import length_profile
from stage_runtime import stage
from train_policy import global_credit


def passages(lengths):
    return [
        {"id": str(i), "case_id": "q", "passage": "word " * length, "terminated": True}
        for i, length in enumerate(lengths)
    ]


class LengthTests(unittest.TestCase):
    def test_evaluation_reports_raw_quality_and_adjusted_reward_separately(self):
        cfg = {**config(), "length_penalty_weight": 0.2}
        rows = passages([200, 250, 350, 450])
        adjusted, details = apply_penalty(rows, [0.75] * 4, cfg)
        for row, quality, extra in zip(rows, adjusted, details):
            row.update(
                extra,
                quality_reward=float(quality),
                normalized_reward=0.75,
                rubric_scores={k: 4.0 for k in ["coherence", "readability", "prompt_fulfillment"]},
            )
        with tempfile.TemporaryDirectory() as folder:
            root = Path(folder)
            common.write_rows(root / "probe-samples.jsonl", rows)
            with (
                patch.object(
                    train_policy, "_evaluate", return_value={"quality_reward_mean": 0.675}
                ),
                patch.object(dist, "get_rank", return_value=0),
                patch("ddp_passage_rl.broadcast_object", side_effect=lambda value: value),
                patch.object(train_policy, "corpus_windows", return_value=set()),
            ):
                result = train_policy.evaluate(None, None, None, None, [], cfg, root, "probe", None)
            self.assertAlmostEqual(result["judge_quality_mean"], 0.75)
            self.assertAlmostEqual(result["quality_reward_mean"], 0.675)
            self.assertAlmostEqual(result["length_penalty_mean"], 0.075)
            self.assertEqual(result["length_in_range_rate"], 0.5)
            self.assertEqual(result["length_overflow_rate"], 0.25)
            self.assertEqual(result["length_underflow_rate"], 0.25)

    def test_band_boundaries_units_and_negative_rewards(self):
        rows = passages([200, 250, 300, 350, 400, 450, 1000])
        quality, details = apply_penalty(rows, np.full(7, 0.75), {"length_penalty_weight": 0.2})
        np.testing.assert_allclose(quality, [0.65, 0.75, 0.75, 0.75, 0.65, 0.55, -0.55])
        self.assertEqual(
            [r["length_in_range"] for r in details], [False, True, True, True, False, False, False]
        )
        self.assertEqual(details[4]["length_overflow_words"], 50)
        self.assertEqual(details[0]["length_underflow_words"], 50)
        json.dumps(details, allow_nan=False)

    def test_reward_wrapper_keeps_raw_judge_eos_and_length_separate(self):
        class Judge:
            def score(self, rows):
                return [
                    {
                        "quality": 0.75,
                        "scores": {
                            k: 4.0 for k in ["coherence", "readability", "prompt_fulfillment"]
                        },
                        "rating_token_mass": {
                            k: 1.0 for k in ["coherence", "readability", "prompt_fulfillment"]
                        },
                    }
                    for _ in rows
                ]

        reward = FrozenPreferenceReward.__new__(FrozenPreferenceReward)
        reward.cfg = {**config(), "length_penalty_weight": 0.2}
        reward.judge = Judge()
        rows = passages([450, 450])
        rows[1]["terminated"] = False
        records, values = reward.score(rows)
        np.testing.assert_allclose(values, [0.55, 0.45])
        self.assertEqual([r["normalized_reward"] for r in records], [0.75, 0.75])
        self.assertEqual(records[1]["truncation_penalty"], 0.1)
        self.assertAlmostEqual(records[1]["quality_reward_before_length_penalty"], 0.65)
        self.assertAlmostEqual(records[1]["length_penalty"], 0.2)
        json.dumps(records, allow_nan=False)

    def test_length_precedes_prompt_baseline_and_leaves_distribution_credit_unchanged(self):
        with tempfile.TemporaryDirectory() as folder:
            dist.init_process_group(
                "gloo", init_method=f"file://{folder}/store", rank=0, world_size=1
            )
            try:
                rows = passages([200, 250, 350, 450])
                quality, _ = apply_penalty(rows, [0.75] * 4, {"length_penalty_weight": 0.2})
                target = make_distribution({"word_distribution": "full-vocabulary"}, ["word word"])
                zero, _ = global_credit(
                    rows, quality, target, 0.0, torch.device("cpu"), quality_baseline="prompt-loo"
                )
                weighted, _ = global_credit(
                    rows, quality, target, 0.1, torch.device("cpu"), quality_baseline="prompt-loo"
                )
                np.testing.assert_allclose(zero.numpy(), [-1 / 30, 0.1, 0.1, -1 / 6], atol=1e-7)
                expected = 0.1 * target.credit([r["passage"] for r in rows])["advantages"]
                np.testing.assert_allclose((weighted - zero).numpy(), expected, atol=1e-7)
            finally:
                dist.destroy_process_group()

    def test_certificate_rejects_length_transform_changes(self):
        cfg = {**config(), "length_penalty_weight": 0.2}
        with tempfile.TemporaryDirectory() as folder:
            root = Path(folder)
            data = root / "data.jsonl"
            data.write_text("{}\n")
            checks = write_certificate(root, cfg, {}, data)
            ready = {"reward_kind": "frozen-fiction-rubric", "audit_path": str(root)}
            with patch.object(common, "config", return_value=cfg):
                common.verify_reward_identity(ready, checks)
                old = {k: v for k, v in checks.items() if k != "reward_transform_identity_sha256"}
                self.assertRaises(RuntimeError, common.verify_reward_identity, ready, old)
            for change in [
                {"length_penalty_weight": 0.4},
                {"length_max_words": 450},
                {"truncation_penalty": 0.2},
            ]:
                with patch.object(common, "config", return_value={**cfg, **change}):
                    self.assertRaises(RuntimeError, common.verify_reward_identity, ready, checks)

    def test_disabled_penalty_and_invalid_settings(self):
        q = np.array([0.1, 0.8])
        adjusted, _ = apply_penalty(passages([10, 1000]), q, {})
        np.testing.assert_array_equal(q, adjusted)
        for change in [
            {"length_penalty_weight": -1},
            {"length_penalty_weight": float("nan")},
            {"length_min_words": 350},
            {"length_penalty_unit_words": 0},
        ]:
            self.assertRaises(ValueError, settings, change)

    def test_training_replay_selection_rejects_test_data_and_late_updates(self):
        rows = [
            {
                "step": step,
                "case_id": str(case),
                "word_count": [250, 300, 400, 450][i],
                "quality_reward": [0.2, 0.4, 0.6, 0.8][i],
            }
            for step in range(1, 51)
            for case in range(4)
            for i in range(4)
        ]
        source = {
            "source_steps": [1, 50],
            "selection_uses_test_generations": False,
            "records": rows,
            "source_job_id": "fixture",
            "source_rollout_sha256": {},
        }
        result = calibrate(source)
        self.assertGreater(result["selected_weight"], 0)
        chosen = next(r for r in result["candidates"] if r["weight"] == result["selected_weight"])
        self.assertLessEqual(chosen["length_to_quality_credit_rms_ratio"], 1)
        self.assertRaises(
            ValueError, calibrate, {**source, "selection_uses_test_generations": True}
        )
        self.assertRaises(ValueError, calibrate, {**source, "source_steps": [1, 500]})

    def test_staged_length_profile_preflight_and_coefficient_binding(self):
        cfg = length_profile()
        self.assertFalse(cfg["gpu_submission_authorized"])
        self.assertEqual(cfg["length_penalty_weight"], 0.2)
        with tempfile.TemporaryDirectory() as folder:
            root = Path(folder)
            path = root / "cfg.json"
            path.write_text(json.dumps(cfg))
            runtime = root / "runtime"
            stage(runtime, path)
            with patch.object(preflight_gpu, "ROOT", runtime):
                report = preflight_gpu.preflight_gpu(require_authorization=False)
                self.assertEqual(report["gpus"], 4)
                self.assertEqual(report["gpu_wall_seconds"], 10800)
                self.assertRaises(PermissionError, preflight_gpu.preflight_gpu)
                cfg["length_penalty_weight"] = 0.4
                (runtime / "run-config.json").write_text(json.dumps(cfg))
                self.assertRaises(AssertionError, preflight_gpu.preflight_gpu, False)


if __name__ == "__main__":
    unittest.main()
