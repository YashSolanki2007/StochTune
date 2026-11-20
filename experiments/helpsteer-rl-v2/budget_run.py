"""Audit two scorers, then run ONE KL policy inside one fixed two-hour allocation."""

import argparse
import json
import os
import signal
import subprocess
import sys
import time
from pathlib import Path

from common import ROOT, config, write_json


def choose_scorer(reports):
    passing = [(name, report) for name, report in reports.items() if report.get("passed")]
    if not passing:
        return None
    # Predeclared selection: highest human preference accuracy after ALL gates pass;
    # ties prefer the smaller legacy model. These are validation, not test, results.
    return max(passing, key=lambda item: (item[1]["non_tie_accuracy"], item[0] == "legacy"))[0]


def terminate(process):
    if process.poll() is None:
        os.killpg(process.pid, signal.SIGTERM)
        try:
            process.wait(timeout=10)
        except subprocess.TimeoutExpired:
            os.killpg(process.pid, signal.SIGKILL)
            process.wait()


def run(reward_output):
    cfg = config()
    if cfg["approved_allocation_seconds"] != 7200 or cfg["approved_maximum_gpus"] != 2:
        raise ValueError(
            "This controller is limited to the explicitly approved two-GPU/two-hour budget"
        )
    started = time.time()
    job = os.environ["SLURM_JOB_ID"]
    output = ROOT / f"outputs/budget-{job}"
    output.mkdir(parents=True, exist_ok=False)
    stop_file = output / "stop-policy"
    signal.signal(signal.SIGUSR1, lambda *_: stop_file.touch())
    allocated = os.environ.get("CUDA_VISIBLE_DEVICES", "0,1").split(",")
    if len(allocated) != 2:
        raise RuntimeError("The budget allocation must expose exactly two GPUs")
    environment = {**os.environ, "HF_HUB_OFFLINE": "1", "TRANSFORMERS_OFFLINE": "1"}
    commands, processes, handles = {}, {}, []
    reports = {}
    summary = {
        "job_id": job,
        "maximum_wall_seconds": 7200,
        "maximum_gpus": 2,
        "maximum_gpu_hours": 4,
        "policy_control_runs": 0,
        "audit_candidates": ["legacy", "skywork"],
        "policy_started": False,
    }
    write_json(output / "budget.json", summary)
    try:
        for gpu, candidate in zip(allocated, ("legacy", "skywork")):
            candidate_output = ROOT / f"outputs/audit-{candidate}-{job}"
            command = [
                sys.executable,
                "audit_reward.py",
                "--candidate",
                candidate,
                "--reward-output",
                str(reward_output),
                "--output",
                str(candidate_output),
            ]
            commands[candidate] = command
            handle = (output / f"audit-{candidate}.log").open("w")
            handles.append(handle)
            processes[candidate] = subprocess.Popen(
                command,
                cwd=ROOT,
                env={**environment, "CUDA_VISIBLE_DEVICES": gpu},
                stdout=handle,
                stderr=subprocess.STDOUT,
                start_new_session=True,
            )
        print(json.dumps({"phase": "reward_audits", "job_id": job}), flush=True)
        audit_deadline = time.time() + 600
        while any(p.poll() is None for p in processes.values()) and time.time() < audit_deadline:
            time.sleep(1)
        for candidate, process in processes.items():
            terminate(process)
            path = ROOT / f"outputs/audit-{candidate}-{job}/checks.json"
            reports[candidate] = (
                json.loads(path.read_text())
                if path.exists()
                else {
                    "passed": False,
                    "error": "Audit failed or exceeded its ten-minute stage budget",
                    "returncode": process.returncode,
                }
            )
            reports[candidate]["audit_returncode"] = process.returncode
            if process.returncode:
                reports[candidate].update(
                    passed=False, error="Audit process or W&B finalization did not complete"
                )
        selected = choose_scorer(reports)
        summary.update(
            audit_reports=reports,
            selected_reward=selected,
            audit_elapsed_seconds=time.time() - started,
        )
        if selected is None:
            summary["stop_reason"] = "No scorer passed the predeclared quality checks"
            write_json(output / "summary.json", summary)
            print(json.dumps(summary), flush=True)
            return
        pointer = ROOT / f"outputs/audit-{selected}-{job}/reward-ready.json"
        ready = json.loads(pointer.read_text())
        write_json(ROOT / "outputs/reward-ready.json", ready)
        subprocess.run([sys.executable, "preflight.py"], cwd=ROOT, env=environment, check=True)
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
        summary.update(policy_started=True, selected_reward_wandb_url=ready["wandb_url"])
        write_json(output / "summary.json", summary)
        print(
            json.dumps(
                {
                    "phase": "quality_unigram_rl",
                    "selected_reward": selected,
                    "target_updates": config()["steps"],
                }
            ),
            flush=True,
        )
        handle = (output / "policy.log").open("w")
        handles.append(handle)
        process = subprocess.Popen(
            command,
            cwd=ROOT,
            env={
                **environment,
                "POLICY_STOP_FILE": str(stop_file),
                "POLICY_TRAINING_DEADLINE_UNIX": str(started + 6600),
            },
            stdout=handle,
            stderr=subprocess.STDOUT,
            start_new_session=True,
        )
        processes["policy"] = process
        try:
            process.wait(timeout=max(1, started + 7140 - time.time()))
        except subprocess.TimeoutExpired:
            terminate(process)
            summary["stop_reason"] = "Shared allocation budget; remaining checkpoints retained"
        summary.update(policy_returncode=process.returncode, elapsed_seconds=time.time() - started)
        comparison = ROOT / f"outputs/quality-unigram-{job}/comparison.json"
        if comparison.exists():
            summary["policy_results"] = json.loads(comparison.read_text())
        write_json(output / "summary.json", summary)
        print(json.dumps(summary), flush=True)
        if process.returncode:
            raise RuntimeError("The single authorized policy run did not complete successfully")
    finally:
        for process in processes.values():
            terminate(process)
        for handle in handles:
            handle.close()


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--reward-output", type=Path, required=True)
    run(parser.parse_args().reward_output.resolve())
