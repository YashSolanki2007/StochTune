"""Verify RL launch inputs on the login node without allocating a GPU or creating a run."""

import json
from pathlib import Path

from peft import PeftConfig
from transformers import AutoTokenizer

from stochastic_llm.io import file_sha256, read_config, read_jsonl, write_json
from stochastic_llm.kl_reward import HistogramTarget, group_kl_reward
from stochastic_llm.tracking import tracking_check


def preflight():
    config = read_config("configs/batch-kl-rl-h200.yaml")
    initial = Path(config["initial_adapter"])
    adapter = PeftConfig.from_pretrained(initial)
    if adapter.base_model_name_or_path != config["model"] or adapter.revision != config["revision"]:
        raise ValueError("Initial SFT adapter does not match the pinned base model")
    if config["method"] != "batch_kl_rl" or config["group_size"] < 2:
        raise ValueError("Invalid KL-reward RL configuration")
    for field in (
        "max_steps",
        "prompts_per_update",
        "rollout_batch_size",
        "microbatch_size",
        "validation_cases",
        "validation_steps",
        "save_steps",
    ):
        if config[field] < 1:
            raise ValueError(f"Invalid {field}")
    if config["validation_samples_per_case"] < 2 or not 0 <= config["warmup_ratio"] < 1:
        raise ValueError("Invalid validation count or warmup")
    if config["reference_kl_beta"] < 0 or config["pseudocount"] <= 0:
        raise ValueError("Invalid reference penalty or smoothing")
    for folder in ("outputs/kl-rl-benchmark", config["output_dir"], "outputs/eval-batch-kl-rl"):
        if Path(folder).exists() and any(Path(folder).iterdir()):
            raise ValueError(f"Output already exists: {folder}")
    data = Path(config["data_dir"])
    manifest = json.loads((data / "manifest.json").read_text())
    for name, digest in manifest["files"].items():
        if file_sha256(data / name) != digest:
            raise ValueError(f"Dataset checksum mismatch: {name}")
    training = read_jsonl(data / "train_configs.jsonl")
    validation = read_jsonl(data / "validation_configs.jsonl")[: config["validation_cases"]]
    tokenizer = AutoTokenizer.from_pretrained(
        config["model"], revision=config["revision"], local_files_only=True, trust_remote_code=False
    )
    lengths = [
        len(x)
        for x in tokenizer([c["prompt"] for c in training + validation], add_special_tokens=False)[
            "input_ids"
        ]
    ]
    if max(lengths) + config["max_new_tokens"] > config["max_sequence_length"]:
        raise ValueError("Prompt plus rollout length exceeds the configured context limit")
    for case in training + validation:
        target = HistogramTarget.from_case(case, config["reward_bins"], config["decimals"])
        group_kl_reward(
            [target.invalid_bin] * config["group_size"], target.probabilities, config["pseudocount"]
        )
    cloud = tracking_check(config)
    report = {
        "model": config["model"],
        "revision": config["revision"],
        "initial_adapter_sha256": file_sha256(initial / "adapter_model.safetensors"),
        "initial_adapter_config_sha256": file_sha256(initial / "adapter_config.json"),
        "training_configs_checked": len(training),
        "validation_configs_checked": len(validation),
        "maximum_prompt_tokens": max(lengths),
        "rollouts_per_update": config["group_size"] * config["prompts_per_update"],
        "planned_training_rollouts": config["max_steps"]
        * config["group_size"]
        * config["prompts_per_update"],
        "wandb": cloud,
        "gpu_jobs_submitted": 0,
        "approval_status": "awaiting_user_final_approval",
    }
    write_json("logs/kl-rl-preflight.json", report)
    print(json.dumps(report, indent=2))


if __name__ == "__main__":
    preflight()
