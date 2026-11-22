"""One bounded two-H200 diagnostic; no retries or extra GPU allocations."""

import json
import os
import signal
import subprocess
import sys
import time
from pathlib import Path

from preflight import preflight

ROOT = Path(__file__).resolve().parent


def run():
    preflight()
    cfg = json.loads((ROOT / "config.json").read_text())
    if len(os.environ["CUDA_VISIBLE_DEVICES"].split(",")) != 2:
        raise ValueError("Expected two allocated GPUs")
    started = time.time()
    output = ROOT / f"outputs/controller-{os.environ['SLURM_JOB_ID']}"
    output.mkdir(parents=True, exist_ok=False)
    stop_file = output / "stop-policy"
    process = subprocess.Popen(
        [
            sys.executable,
            "-m",
            "torch.distributed.run",
            "--standalone",
            "--nnodes=1",
            "--nproc-per-node=2",
            "--max-restarts=0",
            "train_diagnostic.py",
        ],
        cwd=ROOT,
        start_new_session=True,
        env={
            **os.environ,
            "POLICY_TRAINING_DEADLINE_UNIX": str(started + cfg["train_budget_seconds"]),
            "POLICY_STOP_FILE": str(stop_file),
        },
    )
    signal.signal(signal.SIGUSR1, lambda *_: stop_file.touch())
    try:
        process.wait(
            timeout=max(1, started + cfg["approved_allocation_seconds"] - 60 - time.time())
        )
    except subprocess.TimeoutExpired:
        os.killpg(process.pid, signal.SIGTERM)
        try:
            process.wait(timeout=10)
        except subprocess.TimeoutExpired:
            os.killpg(process.pid, signal.SIGKILL)
            process.wait()
    result = ROOT / f"outputs/distribution-only-{os.environ['SLURM_JOB_ID']}/comparison.json"
    report = {
        "returncode": process.returncode,
        "elapsed_seconds": time.time() - started,
        "comparison_written": result.exists(),
        "maximum_gpu_hours": 3,
        "status": "complete" if process.returncode == 0 and result.exists() else "failed",
    }
    (output / "summary.json").write_text(json.dumps(report, indent=2) + "\n")
    print(json.dumps(report), flush=True)
    if report["status"] != "complete":
        raise RuntimeError("Distribution-only diagnostic failed")


if __name__ == "__main__":
    run()
