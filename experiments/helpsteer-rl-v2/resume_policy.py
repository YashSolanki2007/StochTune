"""Continue a policy checkpoint while keeping its scientific configuration fixed."""

import hashlib
import json
from pathlib import Path

# These fields affect bookkeeping/allocation, not the policy or its LR schedule.
RESUME_METADATA_FIELDS = {
    "experiment_profile",
    "wandb_group",
    "train_budget_seconds",
    "gpu_submission_authorized",
    "approved_allocation_seconds",
    "approved_maximum_gpus",
    "approved_maximum_gpu_hours",
    "authorization",
    "shared_allocation_wall_seconds",
    "shared_allocation_maximum_gpu_hours",
    "policy_initialization",
    "resume_checkpoint",
    "resume_source_config",
    "resume_source_config_sha256",
    "resume_checkpoint_sha256",
    "resume_step",
    "resume_source_job_id",
}


def validate_resume_config(cfg, source):
    changed = {
        key
        for key in set(cfg) | set(source)
        if key not in RESUME_METADATA_FIELDS and cfg.get(key) != source.get(key)
    }
    if changed:
        raise ValueError(f"Continuation changes scientific configuration: {sorted(changed)}")


def validate_resume_checkpoint(cfg, arm, world=2):
    import torch

    checkpoint = Path(cfg["resume_checkpoint"])
    source_path = Path(cfg["resume_source_config"])
    if hashlib.sha256(source_path.read_bytes()).hexdigest() != cfg["resume_source_config_sha256"]:
        raise ValueError("Resume source configuration changed")
    source = json.loads(source_path.read_text())
    validate_resume_config(cfg, source)
    complete = json.loads((checkpoint / "complete.json").read_text())
    if not complete.get("complete") or complete["world_size"] != world:
        raise ValueError("Need a complete checkpoint with the same DDP world size")
    expected_files = cfg["resume_checkpoint_sha256"]
    if (
        "state.pt" not in expected_files
        or "adapter/adapter_model.safetensors" not in expected_files
    ):
        raise ValueError("Resume must bind optimizer state and adapter weights")
    for name, expected in expected_files.items():
        if hashlib.sha256((checkpoint / name).read_bytes()).hexdigest() != expected:
            raise ValueError(f"Resume checkpoint changed: {name}")
    state = torch.load(checkpoint / "state.pt", map_location="cpu", weights_only=False)
    source_hash = hashlib.sha256(json.dumps(source, sort_keys=True).encode()).hexdigest()
    if state["config_hash"] != source_hash or state["manifest"] != {"arm": arm}:
        raise ValueError("Checkpoint belongs to a different source configuration or objective")
    step = state["step"]
    if step != cfg["resume_step"] or step != complete["step"] or not 0 < step < cfg["steps"]:
        raise ValueError("Invalid continuation step or completed training schedule")
    if len(state["rank_rng"]) != world or state["scheduler"]["last_epoch"] != step:
        raise ValueError("Checkpoint RNG/scheduler does not match its training step")
    return state


def restore_policy_checkpoint(
    actor, optimizer, scheduler, prompt_rng, state, checkpoint, rank, device
):
    from ddp_passage_rl import restore_rank_rng
    from peft.utils.save_and_load import load_peft_weights, set_peft_model_state_dict

    weights = load_peft_weights(str(Path(checkpoint) / "adapter"), device=str(device))
    set_peft_model_state_dict(actor, weights, adapter_name="default")
    optimizer.load_state_dict(state["optimizer"])
    scheduler.load_state_dict(state["scheduler"])
    prompt_rng.setstate(state["prompt_rng"])
    restore_rank_rng(state["rank_rng"][rank], device)
    return state["step"]
