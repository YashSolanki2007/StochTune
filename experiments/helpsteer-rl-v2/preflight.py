"""Reject invalid data/config/reward certificates before any GPU allocation."""

import hashlib
import json
from pathlib import Path

from common import ROOT, config, verify_reward_identity
from validation import require_reward_audit


def preflight():
    cfg = config()
    if cfg["max_prompt_tokens"] + cfg["max_new_tokens"] + 64 > cfg["reward_max_sequence_length"]:
        raise ValueError("Reward context cannot accommodate generation plus chat formatting")
    if cfg["prompts_per_group"] != cfg["global_group_size"]:
        raise ValueError("This profile requires one response per distinct training prompt")
    manifest = json.loads((ROOT / "data/manifest.json").read_text())
    for name, expected in manifest["prepared_sha256"].items():
        if hashlib.sha256((ROOT / "data" / name).read_bytes()).hexdigest() != expected:
            raise ValueError(f"Prepared data has changed: {name}")
    ready_path = ROOT / "outputs/reward-ready.json"
    if not ready_path.exists():
        raise RuntimeError("No v2 reward audit has passed; policy submission is blocked")
    ready = json.loads(ready_path.read_text())
    audit_output = Path(ready["audit_path"])
    checks = json.loads((audit_output / "checks.json").read_text())
    require_reward_audit(
        checks, hashlib.sha256((ROOT / "data/reward_validation.jsonl").read_bytes()).hexdigest()
    )
    verify_reward_identity(ready, checks)
    print("V2 policy preflight passed. This command allocated no GPU.")


if __name__ == "__main__":
    preflight()
