"""Validate the exact bounded Emma runtime before any GPU allocation is submitted."""

import hashlib
import json
from pathlib import Path

from preflight import preflight, rows

ROOT = Path(__file__).resolve().parent


def allocation_limits(cfg):
    """Fixed ceilings for the original pilot and the separately proposed long run."""
    profile = cfg.get("experiment_profile")
    pilot = profile == "emma-reinforce-length-pilot-4gpu-v1"
    four = pilot or profile in ("emma-reinforce-long-4gpu-v1", "emma-reinforce-length-4gpu-v1")
    longer = four or profile == "emma-reinforce-long-v1"
    return {
        "allocation_seconds": 5400 if pilot else 10800 if four else 14400 if longer else 6600,
        "gpus": 4 if four else 2,
        "calibration_seconds": 600,
        "evaluation_reserve_seconds": 1200 if pilot else 1800 if longer else 1200,
        "controller_reserve_seconds": 60,
    }


def preflight_gpu(require_authorization=True):
    report = preflight()
    cfg = json.loads((ROOT / "run-config.json").read_text())
    if require_authorization and not cfg["gpu_submission_authorized"]:
        raise PermissionError("This GPU run requires fresh user approval")
    comparison = cfg.get("experiment_profile", "").startswith("emma-optimizer-comparison-")
    pilot = cfg.get("experiment_profile") == "emma-reinforce-length-pilot-4gpu-v1"
    length_profile = pilot or cfg.get("experiment_profile") == "emma-reinforce-length-4gpu-v1"
    longer = length_profile or cfg.get("experiment_profile") in (
        "emma-reinforce-long-v1",
        "emma-reinforce-long-4gpu-v1",
    )
    limits = allocation_limits(cfg)
    if comparison or longer:
        from grpo import validate_update_config

        validate_update_config(cfg)
        assert cfg["steps"] == (500 if longer else 80)
        assert cfg["requested_allocation_seconds"] == limits["allocation_seconds"]
        from ddp_passage_rl import configured_world_size

        assert cfg["requested_gpus"] == limits["gpus"]
        assert configured_world_size(cfg) == limits["gpus"]
        assert cfg["quality_baseline"] == "prompt-loo"
        if longer:
            assert cfg["policy_update"] == "reinforce" and cfg["warmup_steps"] == 50
            assert cfg["train_budget_seconds"] == (
                limits["allocation_seconds"] - limits["evaluation_reserve_seconds"]
            )
            assert cfg["evaluation_reserve_seconds"] == limits["evaluation_reserve_seconds"]
            assert cfg["validation_every"] == 50 and cfg["checkpoint_every"] == 25
        if pilot:
            assert cfg["stop_after_steps"] == 100
            assert cfg["final_evaluation_split"] == "validation"
        else:
            assert cfg.get("stop_after_steps", cfg["steps"]) == cfg["steps"]
        if cfg["gpu_submission_authorized"]:
            assert cfg["approved_allocation_seconds"] == limits["allocation_seconds"]
            assert cfg["approved_maximum_gpus"] == limits["gpus"]
            assert cfg["approved_maximum_gpu_hours"] == (
                limits["gpus"] * limits["allocation_seconds"] / 3600
            )
        else:
            assert cfg["approved_allocation_seconds"] == 0 and cfg["approved_maximum_gpus"] == 0
            assert cfg["approved_maximum_gpu_hours"] == 0
    else:
        assert cfg["approved_allocation_seconds"] == 6600 and cfg["approved_maximum_gpus"] == 2
        assert 2 * (cfg["previous_allocation_elapsed_seconds"] + 6600) / 3600 <= 4
    assert cfg["distribution_objective"] == "word-kl"
    assert cfg["distribution_weight"] == 0.1 and cfg["reference_kl_beta"] == 0.002
    assert cfg["reward_validation_protocol"] == 4 and cfg["policy_training_requires_reward_audit"]
    assert not cfg["human_preference_labels"] and not cfg["independent_quality_evaluation"]
    assert cfg["judge_model"] == "Qwen/Qwen2.5-7B-Instruct"
    assert cfg["judge_revision"] == "a09a35458c702b33eeacc393d103063234e8bc28"
    assert cfg["train_budget_seconds"] <= (12600 if longer else 4800)
    assert cfg["max_new_tokens"] == 1024
    assert not cfg.get("resume_checkpoint")
    from length_reward import settings

    length = settings(cfg)
    if length_profile:
        assert length["min_words"] == 250 and length["max_words"] == 350
        assert length["unit_words"] == 100 and length["weight"] > 0
        assert cfg["length_penalty_calibration_file"] == "length-penalty-calibration.json"
        path = ROOT / cfg["length_penalty_calibration_file"]
        assert (
            hashlib.sha256(path.read_bytes()).hexdigest()
            == cfg["length_penalty_calibration_sha256"]
        )
        calibration_report = json.loads(path.read_text())
        assert calibration_report["selection_uses_test_generations"] is False
        assert calibration_report["selected_weight"] == length["weight"]
    else:
        assert length["weight"] == 0, "Unpenalized control must not inherit a length penalty"
    calibration = rows(ROOT / "data/reward_validation.jsonl")
    training = {r["id"]: r for r in rows(ROOT / "data/policy_train.jsonl")}
    assert len(calibration) == 24
    assert {r["kind"] for r in calibration} == {"source", "shuffled", "repetition"}
    assert len({r["group_id"] for r in calibration}) == 8
    from collections import Counter

    from common import words

    for r in calibration:
        original = training[r["case_id"]]
        assert r["group_id"] == original["group_id"] and r["prompt"] == original["prompt"]
        if r["kind"] == "source":
            assert r["passage"] == original["response"]
        if r["kind"] == "shuffled":
            assert Counter(words(r["passage"])) == Counter(words(original["response"]))
    # Import actual runtime entrypoints on CPU to catch namespace/API errors.
    from train_policy import evaluation_subset

    for name, count in (("validation", 8), ("test", 16)):
        available = rows(ROOT / f"data/policy_{name}.jsonl")
        chosen = evaluation_subset(available, count, 42)
        expected_groups = min(count, len({r["group_id"] for r in available}))
        assert len(chosen) == count and len({r["group_id"] for r in chosen}) == expected_groups
    deployment = ROOT / "deployment-manifest.json"
    if deployment.exists():
        for name, expected in json.loads(deployment.read_text()).items():
            assert hashlib.sha256((ROOT / name).read_bytes()).hexdigest() == expected, name
    report.update(
        gpu_submission_authorized=cfg["gpu_submission_authorized"],
        gpu_wall_seconds=limits["allocation_seconds"],
        gpus=limits["gpus"],
        maximum_cumulative_gpu_hours=(None if comparison or longer else 2 * (480 + 6600) / 3600),
        quality_source="frozen-fiction-rubric",
        calibration_rows=24,
    )
    return report


if __name__ == "__main__":
    print(json.dumps(preflight_gpu(), indent=2))
