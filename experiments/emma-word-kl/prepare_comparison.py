"""Prepare two CPU-validated optimizer profiles without GPU spending approval."""

import argparse
import json
from pathlib import Path

from common import config
from stage_runtime import stage


def profiles():
    shared = config()
    for key in list(shared):
        if key.startswith(("search_", "resume_")):
            shared.pop(key)
    # Historical allocation authorization does not transfer to a new experiment.
    for key in (
        "previous_cancelled_job",
        "previous_allocation_elapsed_seconds",
        "cumulative_maximum_gpu_hours",
        "length_penalty_weight",
        "length_min_words",
        "length_max_words",
        "length_penalty_unit_words",
        "length_penalty_calibration_file",
        "length_penalty_calibration_sha256",
        "stop_after_steps",
        "final_evaluation_split",
    ):
        shared.pop(key, None)
    shared.update(
        steps=80,
        warmup_steps=8,
        train_budget_seconds=4800,
        evaluation_reserve_seconds=1200,
        validation_every=20,
        checkpoint_every=10,
        quality_baseline="prompt-loo",
        advantage_normalization="none",
        distribution_weight=0.1,
        reference_kl_beta=0.002,
        validation_prompt_count=8,
        test_prompt_count=16,
        grpo_loss_normalization="sequence-sum",
        grpo_reference_penalty="frozen-sequence-credit",
        grpo_passes=2,
        grpo_clip_epsilon=0.2,
        eval_samples_per_prompt=4,
        requested_allocation_seconds=6600,
        requested_gpus=2,
        distributed_world_size=2,
        gpu_submission_authorized=False,
        approved_allocation_seconds=0,
        approved_maximum_gpus=0,
        approved_maximum_gpu_hours=0,
        authorization="Implementation only; fresh approval is required for each new GPU run.",
        policy_initialization="Fresh pinned instruction actor; no checkpoint-70 continuation",
        comparison_basis="Same seed, prompts, 1280 new training answers, rewards and baselines; "
        "GRPO uses twice as many optimizer steps. GPU time must also be reported.",
    )
    return {
        method: {
            **shared,
            "policy_update": method,
            "experiment_profile": f"emma-optimizer-comparison-{method}-v1",
            "wandb_group": "emma-optimizer-comparison-v1",
        }
        for method in ("reinforce", "grpo")
    }


def prepare(output, stage_runtimes=False):
    output.mkdir(parents=True, exist_ok=False)
    paths = []
    for method, cfg in profiles().items():
        path = output / f"{method}-config.json"
        path.write_text(json.dumps(cfg, indent=2) + "\n")
        paths.append(str(path))
        if stage_runtimes:
            stage(output / f"emma-{method}-v1", path)
    return {"configs": paths, "gpu_submission_authorized": False, "jobs_submitted": 0}


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--stage", action="store_true")
    args = parser.parse_args()
    print(json.dumps(prepare(args.output.resolve(), args.stage), indent=2))
