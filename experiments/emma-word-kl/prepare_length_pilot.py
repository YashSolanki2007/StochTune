"""Prepare a 100-step length pilot on a resumable 500-step schedule."""

import argparse
import json
from pathlib import Path

from prepare_length import length_profile
from stage_runtime import stage


def pilot_profile(gpus=4):
    if gpus not in (2, 4):
        raise ValueError("Pilot requires two or four H200s")
    cfg = length_profile()
    cfg.update(
        experiment_profile=f"emma-reinforce-length-pilot-{gpus}gpu-v1",
        wandb_group=f"emma-reinforce-length-pilot-{gpus}gpu-v1",
        requested_gpus=gpus,
        distributed_world_size=gpus,
        stop_after_steps=100,
        final_evaluation_split="validation",
        requested_allocation_seconds=5400,
        train_budget_seconds=4200,
        evaluation_reserve_seconds=1200,
        authorization="Unapproved staged draft; deployment records the user's pilot approval.",
        experiment_purpose="100-step proof of concept for the fixed 250–350-word penalty; "
        "preserve the 500-step cosine schedule and full checkpoint state. "
        "Use validation, not test prompts, for all pilot evaluation.",
    )
    return cfg


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--stage", action="store_true")
    parser.add_argument("--gpus", type=int, choices=(2, 4), default=4)
    args = parser.parse_args()
    cfg = pilot_profile(args.gpus)
    output = args.output.resolve()
    config_path = output.with_suffix(".json") if args.stage else output
    if config_path.exists():
        raise FileExistsError(config_path)
    config_path.parent.mkdir(parents=True, exist_ok=True)
    config_path.write_text(json.dumps(cfg, indent=2) + "\n")
    if args.stage:
        stage(output, config_path)
    print(
        json.dumps(
            {
                "profile": cfg["experiment_profile"],
                "stop_after_steps": 100,
                "schedule_steps": 500,
                "jobs_submitted": 0,
            }
        )
    )
