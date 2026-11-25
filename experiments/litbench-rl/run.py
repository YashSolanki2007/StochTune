"""Train a literary scorer then quality+word-KL RL in ONE two-hour allocation."""

import json
import os
import signal
import subprocess
import sys
import time
from pathlib import Path

from preflight import preflight

ROOT = Path(__file__).resolve().parent


def write(path, value):
    path.write_text(json.dumps(value, indent=2) + "\n")


def terminate(process):
    if process.poll() is None:
        os.killpg(process.pid, signal.SIGTERM)
        try:
            process.wait(timeout=10)
        except subprocess.TimeoutExpired:
            os.killpg(process.pid, signal.SIGKILL)
            process.wait()


def run():
    preflight()
    cfg = json.loads((ROOT / "config.json").read_text())
    if len(os.environ["CUDA_VISIBLE_DEVICES"].split(",")) != 2:
        raise RuntimeError("Exactly two allocated GPUs are required")
    if (ROOT / "outputs/reward-ready.json").exists():
        raise FileExistsError("Refusing to reuse a stale reward certificate")
    started = time.time()
    job = os.environ["SLURM_JOB_ID"]
    output = ROOT / f"outputs/budget-{job}"
    output.mkdir(exist_ok=False)
    stop = output / "stop-policy"
    signal.signal(signal.SIGUSR1, lambda *_: stop.touch())
    summary = {
        "job_id": job,
        "maximum_wall_seconds": 7200,
        "maximum_gpus": 2,
        "maximum_gpu_hours": 4,
        "policy_started": False,
        "quality_weight": 1,
        "word_kl_weight": cfg["distribution_weight"],
        "base_policy_kl_weight": cfg["reference_kl_beta"],
        "stages": {},
    }
    environment = {
        **os.environ,
        "HF_HUB_OFFLINE": "1",
        "TRANSFORMERS_OFFLINE": "1",
        "WANDB_MODE": "online",
    }
    process = None

    def launch(stage, arguments, deadline, extra):
        nonlocal process
        command = [
            sys.executable,
            "-m",
            "torch.distributed.run",
            "--standalone",
            "--nnodes=1",
            "--nproc-per-node=2",
            "--max-restarts=0",
            *arguments,
        ]
        begin = time.time()
        print(json.dumps({"phase": stage, "job_id": job}), flush=True)
        with (output / f"{stage}.log").open("w") as handle:
            process = subprocess.Popen(
                command,
                cwd=ROOT,
                env={**environment, **extra},
                stdout=handle,
                stderr=subprocess.STDOUT,
                start_new_session=True,
            )
            try:
                process.wait(timeout=max(1, deadline - time.time()))
            except subprocess.TimeoutExpired:
                terminate(process)
                summary["stop_reason"] = stage + " exceeded its stage deadline"
        summary["stages"][stage] = {
            "returncode": process.returncode,
            "elapsed_seconds": time.time() - begin,
        }
        write(output / "summary.json", summary)
        return process.returncode

    try:
        reward_rc = launch(
            "reward",
            ["train_reward.py"],
            started + 2700,
            {"REWARD_TRAINING_DEADLINE_UNIX": str(started + 1800)},
        )
        ready = ROOT / "outputs/reward-ready.json"
        if reward_rc or not ready.exists():
            summary["stop_reason"] = (
                "Reward training failed"
                if reward_rc
                else "New literary scorer failed held-out preference/length checks; no RL spending"
            )
            if reward_rc:
                raise RuntimeError("Literary reward training or logging failed; no policy started")
            return
        import hashlib

        from validation import require_reward_audit

        pointer = json.loads(ready.read_text())
        checks = json.loads((Path(pointer["audit_path"]) / "checks.json").read_text())
        require_reward_audit(
            checks, hashlib.sha256((ROOT / "data/reward_audit.jsonl").read_bytes()).hexdigest()
        )
        if stop.exists() or time.time() >= started + 5400:
            summary["stop_reason"] = "Insufficient remaining time for RL and final evaluation"
            return
        summary.update(policy_started=True, reward_wandb_url=pointer["wandb_url"])
        policy_rc = launch(
            "policy",
            [
                "train_policy.py",
                "--arm",
                "quality-unigram",
                "--output",
                f"outputs/quality-unigram-{job}",
            ],
            started + 7140,
            {
                "POLICY_STOP_FILE": str(stop),
                "POLICY_TRAINING_DEADLINE_UNIX": str(min(time.time() + 3600, started + 6000)),
            },
        )
        comparison = ROOT / f"outputs/quality-unigram-{job}/comparison.json"
        if comparison.exists():
            summary["policy_results"] = json.loads(comparison.read_text())
        if policy_rc:
            raise RuntimeError("The single authorized policy run failed; no automatic retry")
    finally:
        if process is not None:
            terminate(process)
        summary["elapsed_seconds"] = time.time() - started
        write(output / "summary.json", summary)
        print(json.dumps(summary), flush=True)


if __name__ == "__main__":
    run()
