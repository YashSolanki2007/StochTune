"""Validate the exact bounded Emma runtime before any GPU allocation is submitted."""

import hashlib
import json
from pathlib import Path

from preflight import preflight, rows

ROOT = Path(__file__).resolve().parent


def preflight_gpu():
    report = preflight()
    cfg = json.loads((ROOT / "run-config.json").read_text())
    assert cfg["gpu_submission_authorized"]
    assert cfg["approved_allocation_seconds"] == 6600 and cfg["approved_maximum_gpus"] == 2
    assert 2 * (cfg["previous_allocation_elapsed_seconds"] + 6600) / 3600 <= 4
    assert cfg["distribution_objective"] == "word-kl"
    assert cfg["distribution_weight"] == 0.1 and cfg["reference_kl_beta"] == 0.002
    assert cfg["reward_validation_protocol"] == 4 and cfg["policy_training_requires_reward_audit"]
    assert not cfg["human_preference_labels"] and not cfg["independent_quality_evaluation"]
    assert cfg["judge_model"] == "Qwen/Qwen2.5-7B-Instruct"
    assert cfg["judge_revision"] == "a09a35458c702b33eeacc393d103063234e8bc28"
    assert cfg["train_budget_seconds"] <= 4800 and cfg["max_new_tokens"] == 1024
    assert not cfg.get("resume_checkpoint")
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
        gpu_submission_authorized=True,
        gpu_wall_seconds=6600,
        gpus=2,
        maximum_cumulative_gpu_hours=2 * (480 + 6600) / 3600,
        quality_source="frozen-fiction-rubric",
        calibration_rows=24,
    )
    return report


if __name__ == "__main__":
    print(json.dumps(preflight_gpu(), indent=2))
