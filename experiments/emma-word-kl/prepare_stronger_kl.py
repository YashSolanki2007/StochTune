"""Prepare one fresh 1,000-step run with stronger word KL; never submit jobs."""

import argparse
import json
from pathlib import Path

from prepare_length import length_profile
from stage_runtime import stage


def stronger_kl_profile():
    cfg = length_profile()
    cfg.update(
        experiment_profile="emma-stronger-word-kl-4gpu-v1",
        wandb_group="emma-stronger-word-kl-4gpu-v1",
        steps=1000,
        distribution_weight=0.3,
        requested_allocation_seconds=18000,
        train_budget_seconds=16200,
        evaluation_reserve_seconds=1800,
        final_evaluation_split="test",
        authorization="Prepared single-run draft only. User must approve the stated weights "
        "and five-hour/four-H200 allocation before submission. No grid jobs are authorized.",
        policy_initialization="Fresh pinned instruction actor; new 1000-step cosine schedule "
        "with 50 warmup steps, not a continuation of the completed 500-step schedule",
        experiment_purpose="Single stronger-word-KL experiment: quality weight 1, word-KL "
        "weight 0.3, length weight 0.2, base-policy KL 0.002, EOS penalty 0.1. "
        "No grid, promotions, automatic retries or seed-confirmation runs.",
    )
    return cfg


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--stage", action="store_true")
    args = parser.parse_args()
    output = args.output.resolve()
    config_path = output.with_suffix(".json") if args.stage else output
    if config_path.exists():
        raise FileExistsError(config_path)
    config_path.parent.mkdir(parents=True, exist_ok=True)
    cfg = stronger_kl_profile()
    config_path.write_text(json.dumps(cfg, indent=2) + "\n")
    if args.stage:
        stage(output, config_path)
    print(
        json.dumps(
            {
                "prepared": str(output),
                "steps": 1000,
                "word_kl_weight": 0.3,
                "length_weight": 0.2,
                "gpu_submission_authorized": False,
                "maximum_gpu_hours_if_approved": 20,
                "jobs_submitted": 0,
            }
        )
    )
