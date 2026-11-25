"""Validate the exact data, objective and authorized budget without using a GPU."""

import hashlib
import json
from pathlib import Path

ROOT = Path(__file__).resolve().parent


def preflight(root=ROOT):
    cfg = json.loads((root / "config.json").read_text())
    manifest = json.loads((root / "data/manifest.json").read_text())
    assert cfg["gpu_submission_authorized"]
    assert cfg["approved_maximum_gpus"] == 2
    assert cfg["approved_allocation_seconds"] == 7200
    assert cfg["approved_maximum_gpu_hours"] == 4
    assert cfg["distribution_objective"] == "word-kl"
    assert cfg["distribution_weight"] > 0 and cfg["reference_kl_beta"] > 0
    assert cfg["policy_training_requires_reward_audit"]
    assert cfg["reward_validation_protocol"] == 3
    assert not cfg.get("reward_pretrained_full") and not cfg.get("reward_model_name")
    assert not cfg.get("resume_checkpoint")
    assert not cfg["official_test_used"] and not manifest["official_test_used"]
    assert cfg["max_prompt_tokens"] + cfg["max_new_tokens"] < cfg["reward_max_sequence_length"]
    assert (
        manifest["configuration_sha256"]
        == hashlib.sha256((root / "config.json").read_bytes()).hexdigest()
    )
    for name, expected in manifest["prepared_sha256"].items():
        assert hashlib.sha256((root / "data" / name).read_bytes()).hexdigest() == expected, name
    rows = {
        name: [json.loads(line) for line in (root / f"data/{name}.jsonl").read_text().splitlines()]
        for name in manifest["counts"]
    }
    groups = {name: {r["group_id"] for r in values} for name, values in rows.items()}
    for i, name in enumerate(groups):
        assert len(rows[name]) == manifest["counts"][name]
        for other in list(groups)[i + 1 :]:
            assert groups[name].isdisjoint(groups[other]), (name, other)
    for name in ("reward_validation", "reward_audit"):
        assert len(groups[name]) == len(rows[name]) >= 128
        matched = sum(
            min(r["response_tokens_1"], r["response_tokens_2"])
            / max(r["response_tokens_1"], r["response_tokens_2"])
            >= 0.8
            for r in rows[name]
        )
        assert matched >= 50
    assert len(rows["reward_train"]) >= cfg["reward_pair_batch_size"] * 25
    for name, minimum in (("policy_train", 32), ("policy_validation", 8), ("policy_test", 16)):
        assert len(rows[name]) >= minimum
        assert all(len(r["prompt_ids"]) <= cfg["max_prompt_tokens"] for r in rows[name])
    for name in ("reward_train", "reward_validation", "reward_audit"):
        for r in rows[name]:
            assert max(len(r["ids_1"]), len(r["ids_2"])) <= cfg["reward_max_sequence_length"]
            assert r["preference_strength"] in (-1, 1)
            assert r["response_2" if r["preference_strength"] > 0 else "response_1"] == r["chosen"]
    references = [json.loads(s) for s in (root / "data/reference.jsonl").read_text().splitlines()]
    assert len({r["story_id"] for r in references}) == len(references)
    refgroups = {r["group_id"] for r in references}
    for name in ("reward_validation", "reward_audit", "policy_validation", "policy_test"):
        assert refgroups.isdisjoint(groups[name]), name
    vocabulary = json.loads((root / "data/vocabulary.json").read_text())
    categories = vocabulary["categories"]
    assert categories[-1]["label"] == "OTHER"
    assert abs(sum(r["probability"] for r in categories) - 1) < 1e-8
    assert len(categories) <= 10001
    return {
        "passed": True,
        "counts": manifest["counts"],
        "reference_stories": len(references),
        "word_categories_including_other": len(categories),
        "official_test_used": False,
        "policy_arm": "quality-unigram",
        "word_kl_weight": cfg["distribution_weight"],
        "base_policy_kl_weight": cfg["reference_kl_beta"],
        "max_gpu_hours": 4,
    }


if __name__ == "__main__":
    print(json.dumps(preflight(), indent=2))
