"""Run exactly one authorized programming KL policy; reuse the existing scorer."""

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
        "maximum_wall_seconds": 7200,
        "maximum_gpus": 2,
        "maximum_gpu_hours": 4,
        "training_budget_seconds": cfg["train_budget_seconds"],
        "reward_training_runs": 0,
        "new_scorer_audits": 0,
        "policy_runs": 1,
        "distribution_weight": cfg["distribution_weight"],
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
        "quality-unigram",
        "--output",
        f"outputs/quality-unigram-{job}",
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
        process.wait(timeout=max(1, started + 7140 - time.time()))
    except subprocess.TimeoutExpired:
        summary["stop_reason"] = "Two-hour allocation budget"
        os.killpg(process.pid, signal.SIGTERM)
        try:
            process.wait(timeout=10)
        except subprocess.TimeoutExpired:
            os.killpg(process.pid, signal.SIGKILL)
            process.wait()
    finally:
        if process.poll() is None:
            os.killpg(process.pid, signal.SIGTERM)
    comparison = ROOT / f"outputs/quality-unigram-{job}/comparison.json"
    summary.update(returncode=process.returncode, elapsed_seconds=time.time() - started)
    summary["status"] = "complete" if process.returncode == 0 and comparison.exists() else "failed"
    if comparison.exists():
        summary["results"] = json.loads(comparison.read_text())
    write_json(output / "summary.json", summary)
    print(json.dumps(summary), flush=True)
    if summary["status"] != "complete":
        raise RuntimeError("The programming policy run did not finish successfully")


if __name__ == "__main__":
    run()
