"""Frozen-judge checks across all ranks inside the same authorized allocation as RL."""

import datetime
import hashlib
import json
import os
import sys

import torch
import torch.distributed as dist
from common import ROOT, config, log_results, read_rows, start_tracking, write_json, write_rows
from ddp_passage_rl import configured_world_size, gather_objects
from fiction_reward import FrozenRubricJudge, scorer_identity
from validation import validate_scores


def write_certificate(output, cfg, checks, data_path):
    """Export exactly the JSON/certificate consumed by policy startup."""
    from length_reward import transform_identity

    calibration = {
        "scorer_identity_sha256": scorer_identity(cfg),
        "transform": "mean expected rating minus one, divided by four",
        "fitted_parameters": False,
        "human_validated": False,
        "reward_transform_identity_sha256": transform_identity(cfg),
    }
    write_json(output / "calibration.json", calibration)
    report = {
        **checks,
        "scorer_identity_sha256": scorer_identity(cfg),
        "reward_transform_identity_sha256": transform_identity(cfg),
        "validation_data_sha256": hashlib.sha256(data_path.read_bytes()).hexdigest(),
        "calibration_sha256": hashlib.sha256(
            (output / "calibration.json").read_bytes()
        ).hexdigest(),
    }
    write_json(output / "checks.json", report)
    return report


def calibration_assignment(rows, rank, world):
    if world not in (2, 4) or not 0 <= rank < world:
        raise ValueError("Invalid calibration rank or world size")
    return rows[rank::world]


def run():
    cfg = config()
    rank = int(os.environ["LOCAL_RANK"])
    device = torch.device("cuda", rank)
    torch.cuda.set_device(device)
    dist.init_process_group("nccl", device_id=device, timeout=datetime.timedelta(minutes=3))
    world = dist.get_world_size()
    if world != configured_world_size(cfg):
        raise ValueError("Calibration GPU count differs from the configured world size")
    output = ROOT / f"outputs/calibration-{os.environ['SLURM_JOB_ID']}"
    if rank == 0:
        output.mkdir(exist_ok=False)
    dist.barrier()
    run = None
    succeeded = False
    try:
        if rank == 0:
            run = start_tracking(cfg, output, "judge-calibration", human_preference_labels=False)
        judge = FrozenRubricJudge(cfg, rank)
        data_path = ROOT / "data/reward_validation.jsonl"
        rows = read_rows(data_path)
        local = judge.score(calibration_assignment(rows, rank, world))
        gathered = [s for group in gather_objects(local) for s in group]
        scores_by_id = {s["id"]: s for s in gathered}
        if len(gathered) != len(rows) or len(scores_by_id) != len(rows):
            raise ValueError("Every calibration example must be scored exactly once")
        scores = [scores_by_id[r["id"]] for r in rows]
        checks = validate_scores(rows, scores)
        # Exercise serialization on BOTH ranks before rank zero starts exporting.
        json.dumps(checks, allow_nan=False)
        if rank == 0:
            report = write_certificate(output, cfg, checks, data_path)
            write_rows(output / "scores.jsonl", [{**r, "judge": s} for r, s in zip(rows, scores)])
            run.summary.update(report)
            log_results(run, output, adapter=False)
            if report["passed"]:
                write_json(
                    ROOT / "outputs/reward-ready.json",
                    {
                        "path": str(output),
                        "audit_path": str(output),
                        "wandb_url": run.url,
                        "reward_kind": "frozen-fiction-rubric",
                        "reward_model_config": {},
                    },
                )
            print(json.dumps(report, allow_nan=False), flush=True)
        dist.barrier()
        succeeded = True
    finally:
        if run is not None:
            run.finish(exit_code=1 if sys.exc_info()[0] else 0)
        # On failure let torchrun terminate the other rank, rather than hanging
        # in NCCL destruction while that rank waits at a different collective.
        if succeeded:
            dist.destroy_process_group()


if __name__ == "__main__":
    run()
