"""Build a fresh isolated runtime; never copy old submissions or model outputs."""

import argparse
import hashlib
import json
import shutil
from pathlib import Path

ROOT = Path(__file__).resolve().parent
REPO = ROOT.parents[1]


def stage(output):
    output.mkdir(parents=True, exist_ok=False)
    for p in ROOT.iterdir():
        if p.suffix in (".py", ".json", ".sbatch", ".md"):
            shutil.copy2(p, output / p.name)
    v2 = REPO / "experiments/helpsteer-rl-v2"
    for name, dest in (
        ("common.py", "_common_impl.py"),
        ("train_policy.py", "_policy_impl.py"),
        ("audit.py", "_audit_impl.py"),
        ("word_distribution.py", "word_distribution.py"),
    ):
        shutil.copy2(v2 / name, output / dest)
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
    hashes = {
        str(p.relative_to(output)): hashlib.sha256(p.read_bytes()).hexdigest()
        for p in sorted(output.rglob("*"))
        if p.is_file()
    }
    (output / "deployment-manifest.json").write_text(json.dumps(hashes, indent=2) + "\n")
    print(json.dumps({"staged": str(output), "files": len(hashes), "gpu_submission": False}))


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", type=Path, required=True)
    stage(parser.parse_args().output.resolve())
