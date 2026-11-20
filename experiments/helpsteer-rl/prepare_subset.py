"""Stage a bounded two-arm subset comparison, reusing the existing reward model."""

import argparse
import hashlib
import json
import random
import shutil
from pathlib import Path

from common import ROOT, SUPPORT, config, read_rows, write_json, write_rows


def prepare_subset(output):
    if output.exists() and any(output.iterdir()):
        raise FileExistsError(output)
    output.mkdir(parents=True, exist_ok=True)
    cfg = {**config(), **json.loads((ROOT / "subset-config.json").read_text())}
    write_json(output / "config.json", cfg)
    for name in ("common.py", "train_policy.py", "audit.py", "subset.sbatch"):
        shutil.copy2(ROOT / name, output / name)
    support = output / "support"
    support.mkdir()
    for name in (
        "ddp_passage_rl.py",
        "passage_reward.py",
        "pilot.py",
        "judge.py",
        "passage_rl.py",
        "quality.py",
    ):
        shutil.copy2(SUPPORT / name, support / name)
    rng = random.Random(cfg["seed"])
    selected = {}
    sizes = {
        "train": cfg["subset_train_prompt_count"],
        "validation": cfg["validation_prompt_count"],
        "test": cfg["test_prompt_count"],
    }
    data = output / "data"
    for split, n in sizes.items():
        rows = sorted(read_rows(ROOT / f"data/policy_{split}.jsonl"), key=lambda r: r["id"])
        rng.shuffle(rows)
        selected[split] = rows[:n]
        if len(selected[split]) != n:
            raise ValueError(f"Insufficient source prompts for {split}")
        write_rows(data / f"policy_{split}.jsonl", selected[split])
    ids = {r["id"] for r in selected["train"]}
    references = [r for r in read_rows(ROOT / "data/reference.jsonl") if r["id"] in ids]
    if len(references) != len(ids):
        raise ValueError("Subset PMF responses must match subset training prompts")
    write_rows(data / "reference.jsonl", references)
    source = ROOT / "data/manifest.json"
    write_json(
        data / "manifest.json",
        {
            "dataset": cfg["dataset"],
            "revision": cfg["dataset_revision"],
            "profile": cfg["experiment_profile"],
            "source_manifest_sha256": hashlib.sha256(source.read_bytes()).hexdigest(),
            "counts": {f"policy_{k}": len(v) for k, v in selected.items()},
            "reference_responses": len(references),
            "prompt_ids": {key: sorted(r["id"] for r in rows) for key, rows in selected.items()},
            "prepared_sha256": {
                p.name: hashlib.sha256(p.read_bytes()).hexdigest()
                for p in sorted(data.glob("*.jsonl"))
            },
            "reward_model": "Reuse the ongoing 5095-pair reward-model training; "
            "this subset applies to policy prompts and the reference PMF.",
            "full_run": (
                "Cancelled queued full policy jobs; no automatic promotion after this subset."
            ),
            "budget": (
                "Both policy arms sequentially in ONE two-H200 allocation capped at 105 minutes; "
                "40-minute training budget per arm reserves saving and independent evaluation time."
            ),
        },
    )
    print(json.dumps({"output": str(output), "counts": sizes, "updates_per_arm": cfg["steps"]}))


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", type=Path, required=True)
    prepare_subset(parser.parse_args().output.resolve())
