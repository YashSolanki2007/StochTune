"""One bounded allocation: frozen-judge checks, policy training and final evaluation."""

import json
import os
import signal
import subprocess
import sys
import time
from pathlib import Path

from preflight_gpu import allocation_limits, preflight_gpu

ROOT = Path(__file__).resolve().parent


def write(path, value):
    path.write_text(json.dumps(value, indent=2, allow_nan=False) + "\n")


def terminate(process):
    if process.poll() is None:
        os.killpg(process.pid, signal.SIGTERM)
        try:
            process.wait(timeout=10)
        except subprocess.TimeoutExpired:
            os.killpg(process.pid, signal.SIGKILL)
            process.wait()


def run():
    preflight_gpu()
    cfg = json.loads((ROOT / "run-config.json").read_text())
    limits = allocation_limits(cfg)
    if len(os.environ["CUDA_VISIBLE_DEVICES"].split(",")) != limits["gpus"]:
        raise RuntimeError("Allocated H200 count differs from the configured GPU count")
    if (ROOT / "outputs/reward-ready.json").exists():
        raise FileExistsError("Refusing stale judge certificate")
    started = time.time()
    job = os.environ["SLURM_JOB_ID"]
    output = ROOT / f"outputs/budget-{job}"
    output.mkdir(exist_ok=False)
    stop = output / "stop-policy"
    signal.signal(signal.SIGUSR1, lambda *_: stop.touch())
    summary = {
        "job_id": job,
        "maximum_wall_seconds": limits["allocation_seconds"],
        "maximum_gpus": limits["gpus"],
        "previous_job": cfg.get("previous_cancelled_job"),
        "previous_wall_seconds": cfg.get("previous_allocation_elapsed_seconds", 0),
        "maximum_cumulative_gpu_hours": cfg.get("cumulative_maximum_gpu_hours"),
        "policy_update": cfg.get("policy_update", "reinforce"),
        "policy_started": False,
        "stages": {},
    }
    process = None
    environment = {
        **os.environ,
        "HF_HUB_OFFLINE": "1",
        "TRANSFORMERS_OFFLINE": "1",
        "WANDB_MODE": "online",
    }

    def launch(stage, arguments, deadline, extra=None):
        nonlocal process
        command = [
            sys.executable,
            "-m",
            "torch.distributed.run",
            "--standalone",
            "--nnodes=1",
            f"--nproc-per-node={limits['gpus']}",
            "--max-restarts=0",
            *arguments,
        ]
        begin = time.time()
        print(json.dumps({"phase": stage, "job_id": job}), flush=True)
        with (output / f"{stage}.log").open("w") as log:
            process = subprocess.Popen(
                command,
                cwd=ROOT,
                env={**environment, **(extra or {})},
                stdout=log,
                stderr=subprocess.STDOUT,
                start_new_session=True,
            )
            try:
                process.wait(timeout=max(1, deadline - time.time()))
            except subprocess.TimeoutExpired:
                terminate(process)
                summary["stop_reason"] = stage + " exceeded its deadline"
        summary["stages"][stage] = {
            "returncode": process.returncode,
            "elapsed_seconds": time.time() - begin,
        }
        write(output / "summary.json", summary)
        return process.returncode

    try:
        rc = launch("calibration", ["calibrate.py"], started + limits["calibration_seconds"])
        ready = ROOT / "outputs/reward-ready.json"
        if rc:
            raise RuntimeError("Calibration process failed; no automatic retry or RL")
        if not ready.exists():
            summary["stop_reason"] = "Frozen judge failed corruption checks; no RL"
            return
        if stop.exists():
            summary["stop_reason"] = "Stop requested before RL"
            return
        pointer = json.loads(ready.read_text())
        summary.update(policy_started=True, calibration_wandb_url=pointer["wandb_url"])
        rc = launch(
            "policy",
            [
                "train_policy.py",
                "--arm",
                "quality-unigram",
                "--output",
                f"outputs/quality-unigram-{job}",
            ],
            started + limits["allocation_seconds"] - limits["controller_reserve_seconds"],
            {
                "POLICY_STOP_FILE": str(stop),
                "POLICY_TRAINING_DEADLINE_UNIX": str(
                    min(
                        time.time() + cfg["train_budget_seconds"],
                        started
                        + limits["allocation_seconds"]
                        - limits["evaluation_reserve_seconds"],
                    )
                ),
            },
        )
        comparison = ROOT / f"outputs/quality-unigram-{job}/comparison.json"
        if comparison.exists():
            summary["results"] = json.loads(comparison.read_text())
        if rc:
            raise RuntimeError("The one authorized Emma policy run failed; no automatic retry")
    finally:
        if process is not None:
            terminate(process)
        summary["elapsed_seconds"] = time.time() - started
        write(output / "summary.json", summary)
        print(json.dumps(summary), flush=True)


if __name__ == "__main__":
    run()
