"""Check the exact programming target and existing scorer without loading a model."""

import hashlib
import json
from pathlib import Path

import numpy as np
from common import ROOT, config, make_distribution, read_rows, verify_reward_identity
from validation import require_reward_audit


def preflight():
    cfg = config()
    if not cfg.get("gpu_submission_authorized"):
        raise RuntimeError("This prepared profile has not been authorized for GPU submission")
    if cfg["approved_allocation_seconds"] != 7200 or cfg["approved_maximum_gpus"] != 2:
        raise ValueError("This run must remain within two H200s for two hours")
    if cfg["word_distribution"] != "frequency-capped-other" or cfg["distribution_weight"] != 0.5:
        raise ValueError("Expected the approved frequency-capped PMF with lambda=0.5")
    if cfg["max_prompt_tokens"] + cfg["max_new_tokens"] + 64 > cfg["reward_max_sequence_length"]:
        raise ValueError("The scorer context is too short for complete generated answers")
    if cfg["prompts_per_group"] != cfg["global_group_size"]:
        raise ValueError("Expected one complete answer per distinct training prompt")
    manifest = json.loads((ROOT / "data/manifest.json").read_text())
    if (
        hashlib.sha256((ROOT / "config.json").read_bytes()).hexdigest()
        != manifest["configuration_sha256"]
    ):
        raise ValueError("Runtime configuration differs from the deployed manifest")
    for name, expected in manifest["prepared_sha256"].items():
        if hashlib.sha256((ROOT / "data" / name).read_bytes()).hexdigest() != expected:
            raise ValueError(f"Prepared data changed: {name}")
    references = read_rows(ROOT / "data/reference.jsonl")
    cases = {
        name: read_rows(ROOT / f"data/policy_{name}.jsonl")
        for name in ("train", "validation", "test")
    }
    ids = {name: {r["id"] for r in rows} for name, rows in cases.items()}
    if {r["id"] for r in references} != ids["train"]:
        raise ValueError("PMF reference IDs must exactly match policy-training IDs")
    if any(
        ids[a] & ids[b]
        for a, b in (("train", "validation"), ("train", "test"), ("validation", "test"))
    ):
        raise ValueError("Policy splits overlap")
    if len(cases["train"]) < cfg["prompts_per_group"]:
        raise ValueError("Insufficient distinct training prompts")
    target = make_distribution(cfg, [r["response"] for r in references])
    vocabulary = json.loads((ROOT / "data/vocabulary.json").read_text())
    if list(target.index) != [r["word"] for r in vocabulary["categories"][:-1]]:
        raise ValueError("Runtime word vocabulary differs from the prepared vocabulary")
    np.testing.assert_array_equal(
        target.probabilities, [r["probability"] for r in vocabulary["categories"]]
    )
    ready = json.loads((ROOT / "outputs/reward-ready.json").read_text())
    if ready["reward_kind"] != "pretrained":
        raise ValueError("Expected the existing frozen pretrained scorer")
    for name in ("reward_model_name", "reward_model_revision", "reward_pretrained_full"):
        if ready["reward_model_config"][name] != cfg[name]:
            raise ValueError(f"Scorer certificate differs from approved configuration: {name}")
    audit = Path(ready["audit_path"])
    checks = json.loads((audit / "checks.json").read_text())
    require_reward_audit(
        checks, hashlib.sha256((ROOT / "data/reward_validation.jsonl").read_bytes()).hexdigest()
    )
    verify_reward_identity(ready, checks)
    print(
        json.dumps(
            {
                "preflight": "passed",
                "model_loading": False,
                "counts": {name: len(rows) for name, rows in cases.items()},
                "target_words": target.unknown,
                "other_mass": float(target.probabilities[-1]),
                "lambda": cfg["distribution_weight"],
                "scorer": cfg["reward_model_name"],
            }
        ),
        flush=True,
    )


if __name__ == "__main__":
    preflight()
