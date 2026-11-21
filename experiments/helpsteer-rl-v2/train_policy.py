"""Two-H200 controlled RL: frozen human-preference reward, with/without word KL."""

import argparse
import contextlib
import datetime
import hashlib
import json
import os
import random
import signal
import statistics
import time
from pathlib import Path

import numpy as np
import torch
import torch.distributed as dist
from audit import IndependentAudit
from common import (
    ROOT,
    FrozenPreferenceReward,
    config,
    distribution_diagnostics,
    log_results,
    make_distribution,
    quality_advantages,
    read_rows,
    start_tracking,
    verify_reward_identity,
    words,
    write_json,
    write_rows,
)
from ddp_passage_rl import (
    SequencePolicy,
    broadcast_object,
    distributed_update,
    gather_objects,
    save_checkpoint,
)
from judge import load_model
from peft import LoraConfig, get_peft_model
from torch.nn.parallel import DistributedDataParallel as DDP
from transformers import GenerationConfig, get_cosine_schedule_with_warmup
from validation import require_reward_audit


def training_assignment(selected, cfg, rank, world):
    n = cfg["prompts_per_group"]
    if len(selected) != n or len({r["id"] for r in selected}) != n:
        raise ValueError("Training group must contain the requested distinct prompts")
    if n % world or cfg["global_group_size"] % n:
        raise ValueError("Prompt/batch counts must divide evenly across ranks")
    return selected[rank::world], cfg["global_group_size"] // n


def terminal_ids(actor, tokenizer):
    configured = actor.generation_config.eos_token_id
    configured = configured if isinstance(configured, list) else [configured]
    ids = sorted({token for token in configured + [tokenizer.eos_token_id] if token is not None})
    if not ids:
        raise ValueError("The actor needs at least one end-of-sequence token")
    return ids


def trim_completion(completion, stops):
    stop = next((i for i, token in enumerate(completion) if token in stops), None)
    return (completion[: stop + 1], True) if stop is not None else (completion, False)


def rollouts(actor, tokenizer, cases, count, cfg):
    jobs = [(case, sample) for case in cases for sample in range(count)]
    result = []
    stops = terminal_ids(actor, tokenizer)
    generation = GenerationConfig(
        do_sample=True,
        temperature=1.0,
        top_p=1.0,
        top_k=0,
        repetition_penalty=1.0,
        max_new_tokens=cfg["max_new_tokens"],
        pad_token_id=tokenizer.pad_token_id,
        eos_token_id=stops,
    )
    actor.eval()
    for start in range(0, len(jobs), cfg["rollout_batch_size"]):
        chunk = jobs[start : start + cfg["rollout_batch_size"]]
        width = max(len(case["prompt_ids"]) for case, _ in chunk)
        if width > cfg["max_prompt_tokens"]:
            raise ValueError("Policy prompt exceeds the prepared token budget")
        ids = [
            [tokenizer.pad_token_id] * (width - len(c["prompt_ids"])) + c["prompt_ids"]
            for c, _ in chunk
        ]
        masks = [
            [0] * (width - len(c["prompt_ids"])) + [1] * len(c["prompt_ids"]) for c, _ in chunk
        ]
        with torch.inference_mode():
            output = actor.generate(
                input_ids=torch.tensor(ids, device=actor.device),
                attention_mask=torch.tensor(masks, device=actor.device),
                generation_config=generation,
            )
        for (case, sample), completion in zip(chunk, output[:, width:].cpu().tolist()):
            completion, terminated = trim_completion(completion, stops)
            result.append(
                {
                    "id": f"{case['id']}-{sample}",
                    "case_id": case["id"],
                    "prompt": case["prompt"],
                    "prompt_ids": case["prompt_ids"],
                    "completion_ids": completion,
                    "terminated": terminated,
                    "passage": tokenizer.decode(completion, skip_special_tokens=True).strip(),
                }
            )
    return result


def global_credit(rows, quality, target, weight, device):
    parts = gather_objects(
        [
            {
                "passage": row["passage"],
                "quality": float(q),
                **(
                    {"case_id": row["case_id"], "semantic_embedding": row["semantic_embedding"]}
                    if "semantic_embedding" in row
                    else {}
                ),
            }
            for row, q in zip(rows, quality)
        ]
    )
    flat = [r for part in parts for r in part]
    if len({len(part) for part in parts}) != 1:
        raise ValueError("Ranks must have equal local batch sizes")
    metrics = None
    if dist.get_rank() == 0:
        quality = np.array([r["quality"] for r in flat])
        if hasattr(target, "training_credit"):
            qa, raw_da, fidelity_metrics = target.training_credit(flat, quality)
            da = weight * raw_da
            cost = fidelity_metrics["conditional_mmd2_u"]
        else:
            fidelity = target.credit([r["passage"] for r in flat])
            qa = quality_advantages(quality)
            da = weight * fidelity["advantages"]
            fidelity_metrics = target.metrics(fidelity)
            cost = fidelity["kl"]
        credit = torch.tensor(qa + da, dtype=torch.float32, device=device)
        metrics = {
            "quality_reward": float(quality.mean()),
            **fidelity_metrics,
            "quality_advantage_rms": float(np.sqrt(np.mean(qa**2))),
            "distribution_advantage_rms": float(np.sqrt(np.mean(da**2))),
            "joint_reward": float(quality.mean()) - weight * cost,
        }
    else:
        credit = torch.empty(len(flat), device=device)
    dist.broadcast(credit, src=0)
    offset = sum(len(p) for p in parts[: dist.get_rank()])
    return credit[offset : offset + len(rows)], broadcast_object(metrics)


def evaluate(
    policy, tokenizer, reward, target, cases, cfg, output, label, device, audit=None, initial=False
):
    actor = policy.module.actor
    disable = actor.disable_adapter() if initial else contextlib.nullcontext()
    with torch.random.fork_rng(devices=[device.index]), disable:
        torch.manual_seed(cfg["seed"] + 9000 + dist.get_rank() * 100003)
        rows = rollouts(
            actor,
            tokenizer,
            cases[dist.get_rank() :: dist.get_world_size()],
            cfg["eval_samples_per_prompt"],
            cfg,
        )
        scores, quality = reward.score(rows)
        if hasattr(target, "embed_records"):
            rows = target.embed_records(rows)
        judgments = audit.score(rows) if audit is not None else [None] * len(rows)
    local = [
        {**row, **score, "independent_audit": judgment}
        for row, score, judgment in zip(rows, scores, judgments)
    ]
    records = [r for part in gather_objects(local) for r in part]
    metrics = None
    if dist.get_rank() == 0:
        metrics = {
            "quality_reward_mean": statistics.mean(r["quality_reward"] for r in records),
            "normalized_quality_before_penalty": statistics.mean(
                r["normalized_reward"] for r in records
            ),
            "truncation_penalty_mean": statistics.mean(r["truncation_penalty"] for r in records),
            "raw_preference_reward_mean": statistics.mean(r["raw_reward"] for r in records),
            **(
                target.diagnostics(records)
                if hasattr(target, "diagnostics")
                else distribution_diagnostics(target, [r["passage"] for r in records])
            ),
            "passages": len(records),
            "terminated_rate": statistics.mean(r["terminated"] for r in records),
            "mean_word_count": statistics.mean(len(words(r["passage"])) for r in records),
        }
        if audit is not None:
            for key in ("coherence", "readability", "prompt_fulfillment"):
                metrics[f"independent_{key}"] = statistics.mean(
                    r["independent_audit"]["scores"][key] for r in records
                )
        write_rows(output / f"{label}-samples.jsonl", records)
        write_json(output / f"{label}-metrics.json", metrics)
        if hasattr(target, "per_prompt_metrics"):
            write_json(output / f"{label}-conditional-mmd.json", target.per_prompt_metrics(records))
    return broadcast_object(metrics)


def train(arm, output):
    cfg = config()
    weight = 0.0 if arm == "quality-only" else cfg["distribution_weight"]
    local_rank = int(os.environ["LOCAL_RANK"])
    torch.cuda.set_device(local_rank)
    device = torch.device("cuda", local_rank)
    dist.init_process_group("nccl", timeout=datetime.timedelta(minutes=10))
    rank, world = dist.get_rank(), dist.get_world_size()
    if world != 2:
        raise ValueError("This experiment requires two GPUs")
    torch.set_num_threads(4)
    started = time.monotonic()
    ready = json.loads((ROOT / "outputs/reward-ready.json").read_text())
    reward_output = Path(ready["path"])
    audit_output = Path(ready.get("audit_path", reward_output))
    checks = json.loads((audit_output / "checks.json").read_text())
    data_hash = hashlib.sha256((ROOT / "data/reward_validation.jsonl").read_bytes()).hexdigest()
    require_reward_audit(checks, data_hash)
    verify_reward_identity(ready, checks)
    if rank == 0:
        if output.exists() and any(output.iterdir()):
            raise FileExistsError(output)
        output.mkdir(parents=True, exist_ok=True)
    dist.barrier()
    run = None
    stop = [False]
    signal.signal(signal.SIGUSR1, lambda *_: stop.__setitem__(0, True))
    try:
        if rank == 0:
            run = start_tracking(
                cfg,
                output,
                arm,
                distribution_weight_actual=weight,
                reward_model_wandb_url=ready["wandb_url"],
                human_validated=False,
            )
        torch.manual_seed(cfg["seed"])
        tokenizer, base = load_model(cfg["model"], cfg["revision"], device=local_rank)
        actor = get_peft_model(
            base,
            LoraConfig(
                task_type="CAUSAL_LM",
                r=cfg["lora_rank"],
                lora_alpha=cfg["lora_alpha"],
                lora_dropout=0.0,
                target_modules=["q_proj", "k_proj", "v_proj", "o_proj"],
                bias="none",
            ),
        ).eval()
        policy = DDP(
            SequencePolicy(actor),
            device_ids=[local_rank],
            output_device=local_rank,
            broadcast_buffers=False,
            gradient_as_bucket_view=True,
        )
        reward_cfg = {**cfg, **ready.get("reward_model_config", {})}
        reward = FrozenPreferenceReward(
            reward_cfg,
            reward_output,
            local_rank,
            calibration_path=audit_output / "calibration.json",
        )
        cases = {
            name: read_rows(ROOT / f"data/policy_{name}.jsonl")
            for name in ("train", "validation", "test")
        }
        references = [r["response"] for r in read_rows(ROOT / "data/reference.jsonl")]
        target = make_distribution(cfg, references)
        if cfg.get("distribution_objective") == "conditional-semantic-mmd":
            from semantic_runtime import ConditionalSemanticTarget

            target = ConditionalSemanticTarget(cfg, target, device)
        optimizer = torch.optim.AdamW(
            [p for p in actor.parameters() if p.requires_grad],
            lr=cfg["learning_rate"],
            weight_decay=cfg["weight_decay"],
        )
        scheduler = get_cosine_schedule_with_warmup(optimizer, cfg["warmup_steps"], cfg["steps"])
        prompt_rng = random.Random(cfg["seed"])
        torch.manual_seed(cfg["seed"] + rank * 100003)
        random.seed(cfg["seed"] + rank * 100003)
        initial = evaluate(
            policy,
            tokenizer,
            reward,
            target,
            cases["validation"],
            cfg,
            output,
            "validation-0",
            device,
        )
        if rank == 0:
            run.log({f"validation/{k}": v for k, v in initial.items()}, step=0)
        step, reason = 0, "step_limit"
        while step < cfg["steps"]:
            flag = torch.tensor(int(stop[0]), device=device)
            dist.all_reduce(flag, op=dist.ReduceOp.MAX)
            finished = broadcast_object(
                (
                    time.monotonic() - started > cfg["train_budget_seconds"]
                    or bool(flag.item())
                    or time.time() > float(os.environ.get("POLICY_TRAINING_DEADLINE_UNIX", "inf"))
                    or (
                        bool(os.environ.get("POLICY_STOP_FILE"))
                        and Path(os.environ["POLICY_STOP_FILE"]).exists()
                    )
                )
                if rank == 0
                else None
            )
            if finished:
                reason = "time_budget_or_signal"
                break
            iteration = time.monotonic()
            selected = broadcast_object(
                prompt_rng.sample(cases["train"], cfg["prompts_per_group"]) if rank == 0 else None
            )
            local_cases, samples_per_case = training_assignment(selected, cfg, rank, world)
            rows = rollouts(actor, tokenizer, local_cases, samples_per_case, cfg)
            scores, quality = reward.score(rows)
            if hasattr(target, "embed_records"):
                rows = target.embed_records(rows)
            credit, metrics = global_credit(rows, quality, target, weight, device)
            update_metrics = distributed_update(
                policy,
                optimizer,
                rows,
                credit,
                tokenizer.pad_token_id,
                cfg["gradient_microbatch_size"],
                cfg["reference_kl_beta"],
                reference_kl_centering=cfg.get("reference_kl_centering", "none"),
                loss_divisor=cfg.get("objective_loss_divisor", 1.0),
                record_log_probs=cfg.get("record_policy_log_probs", False),
            )
            per_answer = update_metrics.pop("_row_metrics", None)
            if per_answer is not None:
                rows = [{**row, **extra} for row, extra in zip(rows, per_answer)]
            metrics.update(update_metrics)
            divisor = cfg.get("objective_loss_divisor", 1.0)
            metrics.update(
                normalized_joint_reward=metrics["joint_reward"] / divisor,
                effective_quality_credit_rms=metrics["quality_advantage_rms"] / divisor,
                effective_distribution_credit_rms=metrics["distribution_advantage_rms"] / divisor,
                effective_reference_kl_credit_rms=cfg["reference_kl_beta"]
                * update_metrics["reference_kl_credit_rms_centered"]
                / divisor,
            )
            scheduler.step()
            step += 1
            summary = gather_objects(
                {
                    "words": sum(len(words(r["passage"])) for r in rows),
                    "terminated": sum(r["terminated"] for r in rows),
                }
            )
            metrics.update(
                step=step,
                training_passages=step * cfg["global_group_size"],
                elapsed_seconds=time.monotonic() - started,
                seconds_per_update=time.monotonic() - iteration,
                terminated_rate=sum(r["terminated"] for r in summary) / cfg["global_group_size"],
                mean_word_count=sum(r["words"] for r in summary) / cfg["global_group_size"],
                distinct_prompts=cfg["prompts_per_group"],
            )
            if rank == 0:
                with (output / "history.jsonl").open("a") as handle:
                    handle.write(json.dumps(metrics) + "\n")
                run.log({f"rl/{k}": v for k, v in metrics.items()}, step=step)
                print(json.dumps(metrics), flush=True)
            with (output / f"rollouts-rank-{rank}.jsonl").open("a") as handle:
                for row, score, advantage in zip(rows, scores, credit.tolist()):
                    handle.write(
                        json.dumps({"step": step, **row, **score, "advantage": advantage}) + "\n"
                    )
            if step % cfg["validation_every"] == 0:
                result = evaluate(
                    policy,
                    tokenizer,
                    reward,
                    target,
                    cases["validation"],
                    cfg,
                    output,
                    f"validation-{step}",
                    device,
                )
                if rank == 0:
                    run.log({f"validation/{k}": v for k, v in result.items()}, step=step)
            if step % cfg["checkpoint_every"] == 0:
                save_checkpoint(
                    policy,
                    tokenizer,
                    optimizer,
                    scheduler,
                    prompt_rng,
                    step,
                    output,
                    hashlib.sha256(json.dumps(cfg, sort_keys=True).encode()).hexdigest(),
                    {"arm": arm},
                    device,
                )
        save_checkpoint(
            policy,
            tokenizer,
            optimizer,
            scheduler,
            prompt_rng,
            step,
            output,
            hashlib.sha256(json.dumps(cfg, sort_keys=True).encode()).hexdigest(),
            {"arm": arm},
            device,
        )
        # This auditor is separate from the learned reward and is never optimized.
        audit = IndependentAudit(local_rank, cfg["audit_max_sequence_length"])
        baseline = evaluate(
            policy,
            tokenizer,
            reward,
            target,
            cases["test"],
            cfg,
            output,
            "baseline",
            device,
            audit=audit,
            initial=True,
        )
        final = evaluate(
            policy,
            tokenizer,
            reward,
            target,
            cases["test"],
            cfg,
            output,
            "final",
            device,
            audit=audit,
        )
        if rank == 0:
            actor.save_pretrained(output / "adapter")
            tokenizer.save_pretrained(output / "adapter")
            report = {
                "arm": arm,
                "distribution_weight": weight,
                "distribution_objective": cfg.get("distribution_objective", "word-kl"),
                "baseline": baseline,
                "final": final,
                "training_updates": step,
                "training_passages": step * cfg["global_group_size"],
                "stop_reason": reason,
                "elapsed_seconds": time.monotonic() - started,
                "human_validated": False,
                "quality_source": ready.get("reward_kind", "adapter"),
                "reward_model_config": ready.get("reward_model_config", {}),
                "reward_trained_on_human_preferences": True,
                "limitations": [
                    "One training seed; generated outputs lack human evaluation.",
                    "Auditor is a separate same-family 7B model, not human ground truth.",
                    (
                        "MMD matches two equal-weight human-rated prototypes per question; "
                        "it does not establish the population distribution of valid answers."
                        if cfg.get("distribution_objective") == "conditional-semantic-mmd"
                        else "Finite-batch KL is biased and sample-count dependent."
                    ),
                ],
            }
            write_json(output / "comparison.json", report)
            run.summary.update(report)
            log_results(run, output)
            print(json.dumps(report), flush=True)
        dist.barrier()
    finally:
        if rank == 0 and run is not None:
            import sys

            run.finish(exit_code=1 if sys.exc_info()[0] else 0)
        dist.destroy_process_group()


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--arm", choices=["quality-only", "quality-unigram", "quality-semantic-mmd"], required=True
    )
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    train(args.arm, args.output.resolve())
