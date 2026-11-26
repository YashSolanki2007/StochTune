"""Build a fresh isolated Emma runtime without old outputs or submission markers."""

import argparse
import hashlib
import json
import re
import shutil
from pathlib import Path

ROOT = Path(__file__).resolve().parent
REPO = ROOT.parents[1]


def stage(output, config_path=None):
    cfg = None
    if config_path is not None:
        cfg = json.loads(config_path.read_text())
        if cfg.get("gpu_submission_authorized"):
            raise ValueError("Stage comparison profiles without carrying GPU spending approval")
    output.mkdir(parents=True, exist_ok=False)
    for path in ROOT.iterdir():
        if path.name == "deployment-manifest.json":
            continue  # A new manifest cannot include a hash of the old manifest itself.
        if path.is_file() and path.suffix in (".py", ".json", ".md", ".sbatch"):
            shutil.copy2(path, output / path.name)
    v2 = REPO / "experiments/helpsteer-rl-v2"
    for name, dest in (
        ("common.py", "_common_impl.py"),
        ("train_policy.py", "_policy_impl.py"),
        ("word_distribution.py", "word_distribution.py"),
        ("grpo.py", "grpo.py"),
        ("resume_policy.py", "resume_policy.py"),
        ("test_grpo.py", "test_grpo.py"),
        ("test_resume.py", "test_resume.py"),
    ):
        source = ROOT / dest
        if not source.exists():
            source = v2 / name
        shutil.copy2(source, output / dest)
    if cfg is not None:
        (output / "run-config.json").write_text(json.dumps(cfg, indent=2) + "\n")
        from preflight_gpu import allocation_limits

        limits = allocation_limits(cfg)
        hours, remainder = divmod(limits["allocation_seconds"], 3600)
        minutes, seconds = divmod(remainder, 60)
        script = (output / "policy.sbatch").read_text()
        options = {
            "time": f"{hours:02}:{minutes:02}:{seconds:02}",
            "signal": f"B:USR1@{limits['evaluation_reserve_seconds']}",
            "job-name": cfg["experiment_profile"],
            "gres": f"gpu:{limits['gpus']}",
            "cpus-per-task": str(8 * limits["gpus"]),
            "mem": f"{64 * limits['gpus']}G",
        }
        for option, value in options.items():
            script, count = re.subn(
                rf"(?m)^#SBATCH --{option}=.*$", f"#SBATCH --{option}={value}", script
            )
            if count != 1:
                raise ValueError(f"Expected one Slurm {option} directive")
        (output / "policy.sbatch").write_text(script)
    support = output / "support"
    support.mkdir()
    for name in (
        "ddp_passage_rl.py",
        "judge.py",
        "passage_reward.py",
        "passage_rl.py",
        "pilot.py",
        "quality.py",
        "test_ddp.py",
    ):
        source = ROOT / "support" / name
        if not source.exists():
            source = REPO / "experiments/shakespeare-quality-pilot" / name
        shutil.copy2(source, support / name)
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
    parser.add_argument("--config", type=Path, help="Unapproved comparison run profile")
    args = parser.parse_args()
    stage(args.output.resolve(), args.config.resolve() if args.config else None)
