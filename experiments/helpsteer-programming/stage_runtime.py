"""Build an isolated runtime for the explicitly authorized two-H200 policy run."""

import argparse
import hashlib
import json
import shutil
from pathlib import Path

ROOT = Path(__file__).resolve().parent
REPO = ROOT.parents[1]


def stage(output, reward_ready):
    output.mkdir(parents=True, exist_ok=False)
    v2 = REPO / "experiments/helpsteer-rl-v2"
    for name in (
        "common.py",
        "train_policy.py",
        "audit.py",
        "validation.py",
        "word_distribution.py",
    ):
        shutil.copy2(v2 / name, output / name)
    for name in (
        "preflight.py",
        "run_policy.py",
        "policy.sbatch",
        "topic_filter.py",
        "topic-review.json",
        "README.md",
    ):
        shutil.copy2(ROOT / name, output / name)
    support = output / "support"
    support.mkdir()
    for name in (
        "ddp_passage_rl.py",
        "judge.py",
        "passage_reward.py",
        "passage_rl.py",
        "pilot.py",
        "quality.py",
    ):
        shutil.copy2(REPO / "experiments/shakespeare-quality-pilot" / name, support / name)
    shutil.copytree(ROOT / "data", output / "data")
    (output / "outputs").mkdir()
    shutil.copy2(reward_ready, output / "outputs/reward-ready.json")
    cfg = json.loads((ROOT / "config.json").read_text())
    cfg.update(
        gpu_submission_authorized=True,
        approved_allocation_seconds=7200,
        approved_maximum_gpus=2,
        approved_maximum_gpu_hours=4,
        authorization="User explicitly requested starting this two-H200, two-hour capped run.",
        experiment_profile="programming-unigrams-frequency-capped-v1",
        wandb_group="helpsteer-programming-unigram-v1",
    )
    (output / "config.json").write_text(json.dumps(cfg, indent=2) + "\n")
    manifest_path = output / "data/manifest.json"
    manifest = json.loads(manifest_path.read_text())
    manifest["prepared_configuration_sha256"] = manifest["configuration_sha256"]
    manifest["configuration_sha256"] = hashlib.sha256(
        (output / "config.json").read_bytes()
    ).hexdigest()
    manifest_path.write_text(json.dumps(manifest, indent=2) + "\n")
    source_hashes = {
        str(path.relative_to(output)): hashlib.sha256(path.read_bytes()).hexdigest()
        for path in sorted(output.rglob("*"))
        if path.is_file()
    }
    (output / "deployment-manifest.json").write_text(json.dumps(source_hashes, indent=2) + "\n")
    print(json.dumps({"staged": str(output), "files": len(source_hashes), "gpu_submission": False}))


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--reward-ready", type=Path, required=True)
    args = parser.parse_args()
    stage(args.output.resolve(), args.reward_ready.resolve())
