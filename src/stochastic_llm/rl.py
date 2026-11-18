"""On-policy RL using generated groups and a negative empirical-KL reward.

No target answer labels or teacher-forced synthetic answers are used in this
stage. The actor is differentiated only through its generated-action log probs.
"""

import json
import signal
import time
from pathlib import Path

import numpy as np
import torch
from peft import PeftConfig, PeftModel
from transformers import GenerationConfig, get_cosine_schedule_with_warmup, set_seed

from .io import (
    file_sha256,
    read_jsonl,
    require_empty_directory,
    write_json,
    write_jsonl,
)
from .kl_reward import HistogramTarget, group_kl_reward
from .metrics import parse_number, score_samples
from .modeling import device_and_dtype, environment_info, load_model_and_tokenizer
from .tracking import tracked


def completion_log_probs(model, rows, pad_token_id):
    """Sum log pi over actual completion tokens, including EOS and excluding padding."""
    maximum = max(len(row["prompt_ids"]) + len(row["completion_ids"]) for row in rows)
    ids, masks, positions = [], [], []
    for row in rows:
        prompt, completion = row["prompt_ids"], row["completion_ids"]
        if not prompt or not completion:
            raise ValueError("Rollouts must have a prompt and at least one generated token")
        sequence = prompt + completion
        ids.append(sequence + [pad_token_id] * (maximum - len(sequence)))
        masks.append([1] * len(sequence) + [0] * (maximum - len(sequence)))
        positions.append(
            [False] * (len(prompt) - 1)
            + [True] * len(completion)
            + [False] * (maximum - len(sequence))
        )
    inputs = torch.tensor(ids, device=model.device)
    attention = torch.tensor(masks, device=model.device)
    selected = torch.tensor(positions, device=model.device)
    outputs = model(input_ids=inputs, attention_mask=attention, use_cache=False)
    logits = outputs.logits[:, :-1][selected].float()
    targets = inputs[:, 1:][selected]
    token_log_probs = logits.gather(1, targets.unsqueeze(1)).squeeze(1) - torch.logsumexp(logits, 1)
    batch_indices = torch.arange(len(rows), device=model.device)[:, None].expand_as(selected)[
        selected
    ]
    return torch.zeros(len(rows), device=model.device).scatter_add(
        0, batch_indices, token_log_probs
    )


def rollout_group(model, tokenizer, case, config, count=None):
    count = int(count or config["group_size"])
    prompt_ids = tokenizer.encode(case["prompt"], add_special_tokens=False)
    if len(prompt_ids) + config["max_new_tokens"] > config["max_sequence_length"]:
        raise ValueError("Prompt plus rollout token budget exceeds max_sequence_length")
    generation = GenerationConfig(
        do_sample=True,
        temperature=1.0,
        top_p=1.0,
        top_k=0,
        typical_p=1.0,
        repetition_penalty=1.0,
        num_beams=1,
        max_new_tokens=config["max_new_tokens"],
        eos_token_id=tokenizer.eos_token_id,
        pad_token_id=tokenizer.pad_token_id,
        bos_token_id=tokenizer.bos_token_id,
        use_cache=True,
    )
    model.set_adapter("default")
    model.eval()  # Keep dropout OFF for both sampling and log-prob recomputation.
    rows = []
    for start in range(0, count, config["rollout_batch_size"]):
        size = min(config["rollout_batch_size"], count - start)
        inputs = torch.tensor([prompt_ids] * size, device=model.device)
        with torch.inference_mode():
            sequences = model.generate(
                input_ids=inputs,
                attention_mask=torch.ones_like(inputs),
                generation_config=generation,
            )
        for generated in sequences[:, len(prompt_ids) :].cpu().tolist():
            terminated = tokenizer.eos_token_id in generated
            if terminated:
                generated = generated[: generated.index(tokenizer.eos_token_id) + 1]
            text = tokenizer.decode(
                generated[:-1] if terminated else generated, skip_special_tokens=False
            )
            value = parse_number(text) if terminated else None
            rows.append(
                {
                    "case_id": case["id"],
                    "prompt_ids": prompt_ids,
                    "completion_ids": generated,
                    "text": text,
                    "terminated": terminated,
                    "valid": value is not None,
                    "value": value,
                }
            )
    return rows


def policy_update(model, optimizer, groups, pad_token_id, microbatch_size, reference_kl_beta):
    """One on-policy update; never reuse these rollouts after optimizer.step()."""
    rows = [row for group in groups for row in group["rows"]]
    advantages = np.concatenate([group["advantages"] for group in groups])
    optimizer.zero_grad(set_to_none=True)
    model.eval()
    loss_sum, reference_sum = 0.0, 0.0
    try:
        for start in range(0, len(rows), microbatch_size):
            chunk = rows[start : start + microbatch_size]
            reference = None
            if reference_kl_beta:
                model.set_adapter("reference")
                with torch.no_grad():
                    reference = completion_log_probs(model, chunk, pad_token_id)
            model.set_adapter("default")
            log_probs = completion_log_probs(model, chunk, pad_token_id)
            credit = torch.tensor(
                advantages[start : start + len(chunk)], dtype=log_probs.dtype, device=model.device
            )
            if reference is not None:
                log_ratio = log_probs.detach() - reference
                # Score-function gradient of KL(pi || frozen initial policy).
                # The extra +1 term has zero expected score and can be omitted.
                credit = credit - reference_kl_beta * log_ratio
                reference_sum += log_ratio.sum().item()
            loss = -(credit.detach() * log_probs).sum() / len(rows)
            if not torch.isfinite(loss):
                raise FloatingPointError("Nonfinite policy-gradient loss")
            loss.backward()
            loss_sum += loss.detach().item()
    finally:
        model.set_adapter("default")
    parameters = [p for p in model.parameters() if p.requires_grad]
    grad_norm = torch.nn.utils.clip_grad_norm_(parameters, 1.0)
    if not torch.isfinite(grad_norm):
        raise FloatingPointError("Nonfinite policy gradient")
    optimizer.step()
    return {
        "policy_gradient_loss": loss_sum,
        "gradient_norm": float(grad_norm),
        "reference_sequence_kl_estimate": reference_sum / len(rows),
    }


def validation_probe(model, tokenizer, cases, config):
    metrics = []
    devices = [model.device.index or 0] if model.device.type == "cuda" else []
    with torch.random.fork_rng(devices=devices):
        torch.manual_seed(config.get("validation_seed", 1731))
        for case in cases:
            rows = rollout_group(
                model, tokenizer, case, config, count=config["validation_samples_per_case"]
            )
            target = HistogramTarget.from_case(case, config["reward_bins"], config["decimals"])
            bins = [target.assign(row["value"], row["terminated"]) for row in rows]
            reward = group_kl_reward(bins, target.probabilities, config["pseudocount"])
            metrics.append(
                {
                    "id": case["id"],
                    "histogram_kl": reward["kl"],
                    "reward_valid_rate": 1 - reward["counts"][-1] / len(rows),
                    **score_samples(case, rows, config["decimals"]),
                }
            )
    ks = [r["ks_statistic"] for r in metrics if r["ks_statistic"] is not None]
    distances = [
        r["normalized_wasserstein"] for r in metrics if r["normalized_wasserstein"] is not None
    ]
    return {
        "mean_histogram_kl": float(np.mean([r["histogram_kl"] for r in metrics])),
        "numeric_valid_rate": sum(r["valid"] for r in metrics)
        / sum(r["requested"] for r in metrics),
        "reward_valid_rate": float(np.mean([r["reward_valid_rate"] for r in metrics])),
        "median_ks": float(np.median(ks)) if ks else None,
        "median_normalized_wasserstein": float(np.median(distances)) if distances else None,
        "cases": metrics,
    }


def save_rl_checkpoint(model, tokenizer, optimizer, scheduler, rng, step, output, best_cost):
    checkpoint = Path(output) / f"checkpoint-{step}"
    checkpoint.mkdir(parents=True, exist_ok=True)
    (checkpoint / "complete.json").unlink(missing_ok=True)
    model.set_adapter("default")
    model.save_pretrained(checkpoint / "adapter", selected_adapters=["default"])
    tokenizer.save_pretrained(checkpoint / "adapter")
    temporary = checkpoint / "state.pt.tmp"
    torch.save(
        {
            "step": step,
            "optimizer": optimizer.state_dict(),
            "scheduler": scheduler.state_dict(),
            "torch_rng": torch.get_rng_state(),
            "cuda_rng": torch.cuda.get_rng_state_all() if torch.cuda.is_available() else [],
            "numpy_rng": rng.bit_generator.state,
            "best_validation_cost": best_cost,
        },
        temporary,
    )
    temporary.replace(checkpoint / "state.pt")
    write_json(checkpoint / "complete.json", {"step": step, "complete": True})
    return checkpoint


@tracked("rl-training")
def train_rl(config, resume_from_checkpoint=None):
    config = dict(config)
    device_and_dtype(config)
    if config.get("method") != "batch_kl_rl":
        raise ValueError("RL method must be batch_kl_rl")
    if (
        config["group_size"] < 2
        or config["prompts_per_update"] < 1
        or config["max_steps"] < 1
        or config["microbatch_size"] < 1
        or config["rollout_batch_size"] < 1
        or config["reference_kl_beta"] < 0
    ):
        raise ValueError("Invalid RL group size, batch size, steps, or regularizer")
    if config["pseudocount"] <= 0:
        raise ValueError("pseudocount must be positive")
    initial = Path(config["initial_adapter"])
    initial_hash = file_sha256(initial / "adapter_model.safetensors")
    initial_config_hash = file_sha256(initial / "adapter_config.json")
    initial_config = PeftConfig.from_pretrained(initial)
    if initial_config.base_model_name_or_path != config["model"]:
        raise ValueError("Initial adapter and configured base model differ")
    output = Path(config["output_dir"])
    data_dir = Path(config["data_dir"])
    manifest = json.loads((data_dir / "manifest.json").read_text())
    for name, digest in manifest["files"].items():
        if file_sha256(data_dir / name) != digest:
            raise ValueError(f"Dataset changed: {name}")
    if resume_from_checkpoint:
        if json.loads((output / "run_config.json").read_text()) != config:
            raise ValueError("Resume config differs from original RL config")
        old_metadata = json.loads((output / "run_metadata.json").read_text())
        if (
            old_metadata["initial_adapter_sha256"] != initial_hash
            or old_metadata["initial_adapter_config_sha256"] != initial_config_hash
            or old_metadata["data_manifest"] != manifest
        ):
            raise ValueError("RL resume inputs differ from original run")
        checkpoint = Path(resume_from_checkpoint)
        if not (checkpoint / "complete.json").exists():
            raise ValueError("Resume requires a complete RL checkpoint")
    else:
        require_empty_directory(output)
    set_seed(config["seed"])
    loading = dict(config)
    if initial_config.revision:
        if config.get("revision", "main") not in ("main", initial_config.revision):
            raise ValueError("Base revision differs from initial adapter")
        loading["revision"] = initial_config.revision
    base, tokenizer = load_model_and_tokenizer(loading, training=True)
    actor_path = Path(resume_from_checkpoint) / "adapter" if resume_from_checkpoint else initial
    model = PeftModel.from_pretrained(base, actor_path, is_trainable=True)
    model.load_adapter(initial, adapter_name="reference", is_trainable=False)
    model.set_adapter("default")
    model.eval()
    optimizer = torch.optim.AdamW(
        [p for p in model.parameters() if p.requires_grad],
        lr=config["learning_rate"],
        weight_decay=config["weight_decay"],
    )
    scheduler = get_cosine_schedule_with_warmup(
        optimizer,
        num_warmup_steps=int(config["max_steps"] * config["warmup_ratio"]),
        num_training_steps=config["max_steps"],
    )
    rng = np.random.default_rng(config["seed"])
    start, best_cost = 0, float("inf")
    if resume_from_checkpoint:
        state = torch.load(
            Path(resume_from_checkpoint) / "state.pt", map_location="cpu", weights_only=True
        )
        optimizer.load_state_dict(state["optimizer"])
        scheduler.load_state_dict(state["scheduler"])
        rng.bit_generator.state = state["numpy_rng"]
        torch.set_rng_state(state["torch_rng"])
        if state["cuda_rng"]:
            torch.cuda.set_rng_state_all(state["cuda_rng"])
        start, best_cost = state["step"], state["best_validation_cost"]
    training = read_jsonl(data_dir / "train_configs.jsonl")
    validation = read_jsonl(data_dir / "validation_configs.jsonl")[: config["validation_cases"]]
    if not training or not validation:
        raise ValueError("Training and validation configurations must be nonempty")
    families = sorted({case["family"] for case in training})
    grouped = {family: [c for c in training if c["family"] == family] for family in families}
    targets = {
        case["id"]: HistogramTarget.from_case(case, config["reward_bins"], config["decimals"])
        for case in training
    }
    metadata = {
        "objective": "on-policy REINFORCE with negative batch forward-KL reward",
        "credit_assignment": "leave-one-out group reward baseline, no GRPO normalization",
        "initial_adapter_sha256": initial_hash,
        "initial_adapter_config_sha256": initial_config_hash,
        "data_manifest": manifest,
        "resolved_revision": getattr(base.config, "_commit_hash", None),
        "environment": environment_info(),
        "supervised_answer_labels_used": False,
        "reference_regularizer": "reverse sequence KL to frozen initial SFT policy",
    }
    write_json(output / "run_config.json", config)
    write_json(output / "run_metadata.json", metadata)
    write_json(
        output / "reward_targets.json",
        [
            {
                "case_id": case["id"],
                "family": case["family"],
                "parameters": case["parameters"],
                "edges": targets[case["id"]].edges.tolist(),
                "probabilities": targets[case["id"]].probabilities.tolist(),
            }
            for case in training
        ],
    )
    stopped = {"requested": False}

    def stop(signum, frame):
        stopped["requested"] = True

    old_handler = signal.signal(signal.SIGUSR1, stop)
    started = time.perf_counter()
    history_path = output / "history.jsonl"
    last_probe = {"step": None, "result": None}

    def log(values):
        with history_path.open("a") as handle:
            handle.write(json.dumps(values, allow_nan=False) + "\n")
        print(json.dumps(values, allow_nan=False), flush=True)
        if config.get("wandb", {}).get("enabled"):
            import wandb

            wandb.log(values)

    def probe(step):
        nonlocal best_cost
        if last_probe["step"] == step:
            return last_probe["result"]
        values = validation_probe(model, tokenizer, validation, config)
        last_probe.update({"step": step, "result": values})
        write_json(output / f"validation-{step}.json", values)
        log(
            {"rl/update": step, **{f"validation/{k}": v for k, v in values.items() if k != "cases"}}
        )
        if values["mean_histogram_kl"] < best_cost:
            best_cost = values["mean_histogram_kl"]
            model.save_pretrained(output / "best_adapter", selected_adapters=["default"])
            tokenizer.save_pretrained(output / "best_adapter")
            write_json(output / "best.json", {"step": step, "validation_kl": best_cost})
        return values

    if config.get("wandb", {}).get("enabled"):
        import wandb

        wandb.run.config.update({"provenance": metadata}, allow_val_change=True)
        wandb.run.define_metric("rl/update")
        wandb.run.define_metric("rl/*", step_metric="rl/update")
        wandb.run.define_metric("validation/*", step_metric="rl/update")
    step = start
    initial_validation = None
    try:
        if not resume_from_checkpoint:
            initial_validation = probe(0)
        elif (output / "validation-0.json").exists():
            initial_validation = json.loads((output / "validation-0.json").read_text())
        for step in range(start + 1, config["max_steps"] + 1):
            update_started = time.perf_counter()
            groups, raw = [], []
            for i in range(config["prompts_per_update"]):
                family = families[((step - 1) * config["prompts_per_update"] + i) % len(families)]
                case = grouped[family][int(rng.integers(len(grouped[family])))]
                rows = rollout_group(model, tokenizer, case, config)
                target = targets[case["id"]]
                indices = [target.assign(r["value"], r["terminated"]) for r in rows]
                credit = group_kl_reward(indices, target.probabilities, config["pseudocount"])
                groups.append(
                    {
                        "rows": rows,
                        "advantages": credit["advantages"],
                        "kl": credit["kl"],
                        "invalid_count": int(credit["counts"][-1]),
                    }
                )
                for row, bin_id, advantage in zip(rows, indices, credit["advantages"]):
                    raw.append(
                        {
                            "update": step,
                            "group": i,
                            **row,
                            "reward_bin": bin_id,
                            "group_reward": credit["reward"],
                            "scaled_advantage": float(advantage),
                        }
                    )
            losses = policy_update(
                model,
                optimizer,
                groups,
                tokenizer.pad_token_id,
                config["microbatch_size"],
                config["reference_kl_beta"],
            )
            scheduler.step()
            rows = [row for group in groups for row in group["rows"]]
            write_jsonl(output / f"rollouts-{step:05d}.jsonl", raw)
            log(
                {
                    "rl/update": step,
                    "rl/target_histogram_kl": float(np.mean([g["kl"] for g in groups])),
                    "rl/group_reward": -float(np.mean([g["kl"] for g in groups])),
                    "rl/numeric_valid_rate": sum(r["valid"] for r in rows) / len(rows),
                    "rl/reward_valid_rate": 1 - sum(g["invalid_count"] for g in groups) / len(rows),
                    "rl/learning_rate": scheduler.get_last_lr()[0],
                    "rl/seconds_per_update": time.perf_counter() - update_started,
                    **{f"rl/{k}": v for k, v in losses.items()},
                }
            )
            if stopped["requested"]:
                break
            if step % config["validation_steps"] == 0:
                probe(step)
            if step % config["save_steps"] == 0:
                save_rl_checkpoint(
                    model, tokenizer, optimizer, scheduler, rng, step, output, best_cost
                )
        save_rl_checkpoint(model, tokenizer, optimizer, scheduler, rng, step, output, best_cost)
        if stopped["requested"]:
            raise InterruptedError("RL checkpoint saved after SIGUSR1; resume before evaluation")
        final_validation = probe(step)
        model.set_adapter("default")
        model.save_pretrained(output / "adapter", selected_adapters=["default"])
        tokenizer.save_pretrained(output / "adapter")
        metrics = {
            "completed_updates": step,
            "wall_seconds": time.perf_counter() - started,
            "generated_training_responses": step
            * config["prompts_per_update"]
            * config["group_size"],
            "initial_validation_kl": initial_validation["mean_histogram_kl"]
            if initial_validation
            else None,
            "final_validation_kl": final_validation["mean_histogram_kl"],
            "best_validation_kl": best_cost,
            "final_numeric_valid_rate": final_validation["numeric_valid_rate"],
        }
        write_json(output / "metrics.json", metrics)
        return metrics
    finally:
        signal.signal(signal.SIGUSR1, old_handler)
