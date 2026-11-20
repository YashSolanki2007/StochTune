"""Run one bounded conditional semantic-MMD policy with the existing scorer."""

import json
import os
import signal
import subprocess
import sys
import time

from common import ROOT, config, write_json
from preflight import preflight


def run():
    preflight()
    cfg = config()
    allocation_seconds = cfg["approved_allocation_seconds"]
    if len(os.environ["CUDA_VISIBLE_DEVICES"].split(",")) != 2:
        raise RuntimeError("Expected exactly two allocated GPUs")
    started = time.time()
    job = os.environ["SLURM_JOB_ID"]
    output = ROOT / f"outputs/controller-{job}"
    output.mkdir(parents=True, exist_ok=False)
    stop_file = output / "stop-policy"
    signal.signal(signal.SIGUSR1, lambda *_: stop_file.touch())
    summary = {
        "job_id": job,
        "maximum_wall_seconds": allocation_seconds,
        "maximum_gpus": 2,
        "maximum_gpu_hours": allocation_seconds * cfg["approved_maximum_gpus"] / 3600,
        "training_budget_seconds": cfg["train_budget_seconds"],
        "reward_training_runs": 0,
        "new_scorer_audits": 0,
        "policy_runs": 1,
        "distribution_weight": cfg["distribution_weight"],
        "distribution_objective": cfg["distribution_objective"],
        "teacher_generation_runs": 0,
        "status": "starting",
    }
    write_json(output / "budget.json", summary)
    command = [
        sys.executable,
        "-m",
        "torch.distributed.run",
        "--standalone",
        "--nnodes=1",
        "--nproc-per-node=2",
        "--max-restarts=0",
        "train_policy.py",
        "--arm",
        "quality-semantic-mmd",
        "--output",
        f"outputs/quality-semantic-mmd-{job}",
    ]
    process = subprocess.Popen(
        command,
        cwd=ROOT,
        start_new_session=True,
        env={
            **os.environ,
            "POLICY_STOP_FILE": str(stop_file),
            "POLICY_TRAINING_DEADLINE_UNIX": str(started + cfg["train_budget_seconds"]),
        },
    )
    try:
        process.wait(timeout=max(1, started + allocation_seconds - 60 - time.time()))
    except subprocess.TimeoutExpired:
        summary["stop_reason"] = "Allocation wall-time budget"
        os.killpg(process.pid, signal.SIGTERM)
        try:
            process.wait(timeout=10)
        except subprocess.TimeoutExpired:
            os.killpg(process.pid, signal.SIGKILL)
            process.wait()
    finally:
        if process.poll() is None:
            os.killpg(process.pid, signal.SIGTERM)
    comparison = ROOT / f"outputs/quality-semantic-mmd-{job}/comparison.json"
    summary.update(returncode=process.returncode, elapsed_seconds=time.time() - started)
    summary["status"] = "complete" if process.returncode == 0 and comparison.exists() else "failed"
    if comparison.exists():
        summary["results"] = json.loads(comparison.read_text())
    write_json(output / "summary.json", summary)
    print(json.dumps(summary), flush=True)
    if summary["status"] != "complete":
        raise RuntimeError("The conditional semantic-MMD policy run did not finish successfully")


if __name__ == "__main__":
    run()
