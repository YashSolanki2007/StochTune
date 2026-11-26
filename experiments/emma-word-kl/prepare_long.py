"""Prepare a fresh 500-batch REINFORCE run; no submission or spending approval."""

import argparse
import json
from pathlib import Path

from prepare_comparison import profiles
from stage_runtime import stage


def long_profile(gpus=2):
    if gpus not in (2, 4):
        raise ValueError("Long-run GPU count must be two or four")
    cfg = profiles()["reinforce"]
    for key in [
        "grpo_passes",
        "grpo_clip_epsilon",
        "grpo_reference_penalty",
        "grpo_loss_normalization",
        "comparison_basis",
    ]:
        cfg.pop(key, None)
    cfg.update(
        experiment_profile="emma-reinforce-long-v1",
        wandb_group="emma-reinforce-long-v1",
        steps=500,
        warmup_steps=50,
        train_budget_seconds=12600,
        requested_allocation_seconds=14400,
        evaluation_reserve_seconds=1800,
        checkpoint_every=25,
        validation_every=50,
        eval_samples_per_prompt=4,
        authorization="Proposed longer REINFORCE run only; final GPU-budget approval pending.",
        policy_initialization="Fresh pinned instruction actor and new 500-step cosine schedule; "
        "not a continuation of the completed 80-step schedule",
        experiment_purpose="Test longer REINFORCE with unchanged reward and prompt-LOO quality "
        "credit; one seed, no extra optimizer comparison or retries.",
    )
    if gpus == 4:
        cfg.update(
            experiment_profile="emma-reinforce-long-4gpu-v1",
            wandb_group="emma-reinforce-long-4gpu-v1",
            distributed_world_size=4,
            requested_gpus=4,
            requested_allocation_seconds=10800,
            train_budget_seconds=9000,
            authorization="User selected four H200s with a three-hour cap; "
            "authorization is recorded only in the final deployed configuration.",
        )
    return cfg


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--stage", action="store_true")
    parser.add_argument("--gpus", type=int, choices=(2, 4), default=2)
    args = parser.parse_args()
    cfg = long_profile(args.gpus)
    output = args.output.resolve()
    if args.stage:
        config_path = output.with_suffix(".json")
        if config_path.exists():
            raise FileExistsError(config_path)
        config_path.parent.mkdir(parents=True, exist_ok=True)
        config_path.write_text(json.dumps(cfg, indent=2) + "\n")
        stage(output, config_path)
    else:
        if output.exists():
            raise FileExistsError(output)
        output.parent.mkdir(parents=True, exist_ok=True)
        output.write_text(json.dumps(cfg, indent=2) + "\n")
    print(
        json.dumps(
            {
                "prepared": str(output),
                "steps": 500,
                "training_answers": 8000,
                "requested_maximum_gpu_hours": (
                    cfg["requested_gpus"] * cfg["requested_allocation_seconds"] / 3600
                ),
                "gpu_submission_authorized": False,
                "jobs_submitted": 0,
            }
        )
    )
