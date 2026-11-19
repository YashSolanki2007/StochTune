"""Two-rank passage RL with global rewards, synchronized LoRA, and a time budget."""

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
from judge import load_model
from passage_reward import PassageDistribution, quality_advantages
from passage_rl import rollouts, score_rows
from peft import LoraConfig, get_peft_model
from pilot import ROOT, duplicate_window_rate, words, write_json
from quality import FrozenQualityJudge
from torch.nn.parallel import DistributedDataParallel as DDP
from transformers import get_cosine_schedule_with_warmup

from stochastic_llm.rl import completion_log_probs


class SequencePolicy(torch.nn.Module):
    """Expose completion likelihood through DDP's forward and gradient hooks."""

    def __init__(self, actor):
        super().__init__()
        self.actor = actor

    def forward(self, rows, pad_token_id):
        return completion_log_probs(self.actor, rows, pad_token_id)


def gather_objects(value):
    values = [None] * dist.get_world_size()
    dist.all_gather_object(values, value)
    return values


def broadcast_object(value):
    box = [value]
    dist.broadcast_object_list(box, src=0)
    return box[0]


def global_credit(local_rows, local_quality, target, weight, device):
    """Both ranks contribute to ONE histogram and ONE quality baseline."""
    gathered = gather_objects(
        [
            {"passage": row["passage"], "quality": float(quality)}
            for row, quality in zip(local_rows, local_quality)
        ]
    )
    sizes = [len(part) for part in gathered]
    if len(set(sizes)) != 1 or min(sizes) < 1:
        raise ValueError("Training ranks must have equal, nonempty local batches")
    n = sum(sizes)
    if dist.get_rank() == 0:
        rows = [row for part in gathered for row in part]
        quality = np.array([r["quality"] for r in rows])
        reward = target.credit([r["passage"] for r in rows])
        credit = quality_advantages(quality) + weight * reward["advantages"]
        tensor = torch.tensor(credit, dtype=torch.float32, device=device)
        metrics = {
            "quality_reward": float(quality.mean()),
            "bigram_kl": reward["kl"],
            "unknown_bigram_rate": reward["unknown_bigram_rate"],
            "joint_reward": float(quality.mean()) - weight * reward["kl"],
            "global_group_size": n,
        }
    else:
        tensor = torch.empty(n, dtype=torch.float32, device=device)
        metrics = None
    dist.broadcast(tensor, src=0)
    metrics = broadcast_object(metrics)
    offset = sum(sizes[: dist.get_rank()])
    return tensor[offset : offset + sizes[dist.get_rank()]], metrics


def distributed_update(policy, optimizer, rows, credit, pad_id, microbatch, reference_beta):
    """Average local sequence losses; DDP averaging yields the global batch mean."""
    actor = policy.module.actor
    actor.eval()
    optimizer.zero_grad(set_to_none=True)
    loss_sum, reference_sum = 0.0, 0.0
    for start in range(0, len(rows), microbatch):
        chunk = rows[start : start + microbatch]
        last = start + microbatch >= len(rows)
        sync = contextlib.nullcontext() if last else policy.no_sync()
        with torch.no_grad(), actor.disable_adapter():
            reference = completion_log_probs(actor, chunk, pad_id)
        # no_sync must wrap both forward AND backward for accumulation to work.
        with sync:
            current = policy(chunk, pad_id)
            delta = current.detach() - reference
            advantage = (
                credit[start : start + len(chunk)].to(current.dtype) - reference_beta * delta
            )
            loss = -(advantage.detach() * current).sum() / len(rows)
            if not torch.isfinite(loss):
                raise FloatingPointError("Nonfinite distributed policy loss")
            loss.backward()
        loss_sum += loss.detach().item()
        reference_sum += delta.sum().item()
    parameters = [p for p in actor.parameters() if p.requires_grad]
    norm = torch.nn.utils.clip_grad_norm_(parameters, 1.0)
    if not torch.isfinite(norm):
        raise FloatingPointError("Nonfinite distributed gradient")
    optimizer.step()
    metrics = torch.tensor(
        [loss_sum, reference_sum / len(rows)], device=credit.device, dtype=torch.float64
    )
    dist.all_reduce(metrics)
    metrics /= dist.get_world_size()
    return {
        "policy_gradient_loss": metrics[0].item(),
        "reference_sequence_kl_estimate": metrics[1].item(),
        "gradient_norm": float(norm),
    }


def rank_rng_state(device):
    return {
        "torch": torch.get_rng_state(),
        "python": random.getstate(),
        "cuda": torch.cuda.get_rng_state(device) if device.type == "cuda" else None,
    }


def restore_rank_rng(state, device):
    torch.set_rng_state(state["torch"])
    random.setstate(state["python"])
    if state["cuda"] is not None:
        torch.cuda.set_rng_state(state["cuda"], device)


def save_checkpoint(
    policy, tokenizer, optimizer, scheduler, prompt_rng, step, output, config_hash, manifest, device
):
    states = gather_objects(rank_rng_state(device))
    path = output / f"checkpoint-{step}"
    if dist.get_rank() == 0:
        if not (path / "complete.json").exists():
            path.mkdir(exist_ok=True)
            policy.module.actor.save_pretrained(path / "adapter")
            tokenizer.save_pretrained(path / "adapter")
            temporary = path / "state.pt.tmp"
            torch.save(
                {
                    "step": step,
                    "optimizer": optimizer.state_dict(),
                    "scheduler": scheduler.state_dict(),
                    "prompt_rng": prompt_rng.getstate(),
                    "rank_rng": states,
                    "config_hash": config_hash,
                    "manifest": manifest,
                },
                temporary,
            )
            temporary.replace(path / "state.pt")
            write_json(
                path / "complete.json", {"step": step, "world_size": len(states), "complete": True}
            )
        write_json(output / "latest-checkpoint.json", {"step": step, "path": str(path)})
    dist.barrier()
    return path


def distributed_evaluate(
    policy,
    tokenizer,
    judge,
    target,
    cases,
    count,
    cfg,
    output,
    label,
    corpus_windows,
    device,
    initial=False,
):
    rank = dist.get_rank()
    local_cases = cases[rank :: dist.get_world_size()]
    if not local_cases:
        raise ValueError("Every rank must have evaluation cases")
    actor = policy.module.actor
    reference = actor.disable_adapter() if initial else contextlib.nullcontext()
    with torch.random.fork_rng(devices=[device.index]), reference:
        torch.manual_seed(cfg["seed"] + 9000 + rank * 100003)
        rows = rollouts(actor, tokenizer, local_cases, count, cfg)
        scores, quality = score_rows(rows, judge)
    local = [
        {**row, "judge": score, "quality_reward": float(q)}
        for row, score, q in zip(rows, scores, quality)
    ]
    parts = gather_objects(local)
    metrics = None
    if rank == 0:
        records = [row for part in parts for row in part]
        fidelity = target.credit([r["passage"] for r in records])
        metrics = {
            "quality_reward_mean": statistics.mean(r["quality_reward"] for r in records),
            "bigram_kl": fidelity["kl"],
            "unknown_bigram_rate": fidelity["unknown_bigram_rate"],
            "passages": len(records),
            "terminated_rate": statistics.mean(r["terminated"] for r in records),
            "mean_word_count": statistics.mean(len(words(r["passage"])) for r in records),
            "corpus_12word_overlap": statistics.mean(
                duplicate_window_rate(r["passage"], corpus_windows) or 0.0 for r in records
            ),
        }
        for dimension in ("coherence", "fluency", "prompt_fulfillment"):
            metrics[dimension] = statistics.mean(r["judge"]["scores"][dimension] for r in records)
        metrics["joint_reward"] = (
            metrics["quality_reward_mean"] - cfg["distribution_weight"] * fidelity["kl"]
        )
        write_json(output / f"{label}-metrics.json", metrics)
        with (output / f"{label}-samples.jsonl").open("w") as handle:
            for record in records:
                handle.write(json.dumps(record, ensure_ascii=False) + "\n")
    return broadcast_object(metrics)


def source_manifest():
    names = [
        "ddp-config.json",
        "ddp-prompts.json",
        "ddp_passage_rl.py",
        "passage_reward.py",
        "passage_rl.py",
        "quality.py",
        "judge.py",
        "pilot.py",
        "data/reference.txt",
    ]
    return {name: hashlib.sha256((ROOT / name).read_bytes()).hexdigest() for name in names}


def train(output, resume=None):
    launched = time.monotonic()
    local_rank = int(os.environ["LOCAL_RANK"])
    torch.cuda.set_device(local_rank)
    device = torch.device("cuda", local_rank)
    dist.init_process_group("nccl", timeout=datetime.timedelta(minutes=10))
    rank, world = dist.get_rank(), dist.get_world_size()
    torch.set_num_threads(4)
    cfg = json.loads((ROOT / "ddp-config.json").read_text())
    if world != cfg["world_size"] or world != 2:
        raise ValueError("This allocation is configured for exactly two GPUs")
    if cfg["global_group_size"] % world or cfg["prompts_per_group"] != world:
        raise ValueError("Group/prompt partition is incompatible with the world size")
    if os.environ.get("WANDB_MODE", "online") != "online" or os.environ.get(
        "WANDB_DISABLED", "false"
    ).lower() in ("true", "1", "yes"):
        raise RuntimeError("Online W&B logging is required")
    calibration = ROOT / "outputs/calibration-498176/checks.json"
    if not json.loads(calibration.read_text())["passed"]:
        raise RuntimeError("Existing judge calibration did not pass")
    manifest = source_manifest()
    config_hash = hashlib.sha256(json.dumps(cfg, sort_keys=True).encode()).hexdigest()
    prompts = json.loads((ROOT / "ddp-prompts.json").read_text())
    all_prompts = [r["prompt"] for rows in prompts.values() for r in rows]
    if len(all_prompts) != len(set(all_prompts)):
        raise ValueError("Prompt splits overlap")
    if rank == 0:
        if resume is None and output.exists() and any(output.iterdir()):
            raise FileExistsError(output)
        output.mkdir(parents=True, exist_ok=True)
    dist.barrier()
    stop_requested = [False]
    signal.signal(signal.SIGUSR1, lambda *_: stop_requested.__setitem__(0, True))
    run = None
    if rank == 0:
        import wandb

        previous = json.loads((output / "wandb_run.json").read_text()) if resume else None
        run = wandb.init(
            entity=cfg["wandb_entity"],
            project=cfg["wandb_project"],
            name=f"shakespeare-ddp-{os.environ.get('SLURM_JOB_ID')}",
            group="shakespeare-passage-ddp-v1",
            job_type="rl-ddp",
            mode="online",
            id=previous["id"] if previous else None,
            resume="must" if previous else "never",
            config={
                **cfg,
                "source_hashes": manifest,
                "slurm_job_id": os.environ.get("SLURM_JOB_ID"),
                "train_prompts": len(prompts["train"]),
                "global_batch": cfg["global_group_size"],
                "local_batch": cfg["global_group_size"] // world,
                "reward_unchanged_from_pilot": True,
                "human_validated": False,
            },
            dir=str(output),
            settings=wandb.Settings(base_url=cfg["wandb_base_url"]),
        )
        write_json(output / "wandb_run.json", {"id": run.id, "url": run.url})
        write_json(output / "run_config.json", cfg)
        write_json(output / "source-manifest.json", manifest)
        source = wandb.Artifact(f"passage-ddp-source-{run.id}", type="source")
        for name in manifest:
            source.add_file(str(ROOT / name), name=name)
        source.add_file(str(calibration), name="calibration.json")
        run.log_artifact(source)
        print(json.dumps({"status": "loading", "wandb_url": run.url, "gpus": world}), flush=True)
    try:
        torch.manual_seed(cfg["seed"])
        tokenizer, base = load_model(cfg["actor_model"], cfg["actor_revision"], device=local_rank)
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
        judge = FrozenQualityJudge(
            cfg["judge_model"], cfg["judge_revision"], cfg["judge_batch_size"], device=local_rank
        )
        reference = (ROOT / "data/reference.txt").read_text()
        target = PassageDistribution(
            reference.split("\n\n"), cfg["histogram_total_pseudocount_mass"]
        )
        corpus = words((ROOT / "data/tinyshakespeare.txt").read_text())
        corpus_windows = {tuple(corpus[i : i + 12]) for i in range(len(corpus) - 11)}
        optimizer = torch.optim.AdamW(
            [p for p in actor.parameters() if p.requires_grad],
            lr=cfg["learning_rate"],
            weight_decay=cfg["weight_decay"],
        )
        scheduler = get_cosine_schedule_with_warmup(optimizer, cfg["warmup_steps"], cfg["steps"])
        prompt_rng = random.Random(cfg["seed"])
        torch.manual_seed(cfg["seed"] + rank * 100003)
        random.seed(cfg["seed"] + rank * 100003)
        step = 0
        if resume:
            if not (resume / "complete.json").exists():
                raise ValueError("Cannot resume an incomplete checkpoint")
            state = torch.load(resume / "state.pt", map_location="cpu", weights_only=False)
            if state["config_hash"] != config_hash or state["manifest"] != manifest:
                raise ValueError("Resume source/configuration changed")
            from peft.utils.save_and_load import load_peft_weights, set_peft_model_state_dict

            weights = load_peft_weights(str(resume / "adapter"), device=str(device))
            set_peft_model_state_dict(actor, weights, adapter_name="default")
            optimizer.load_state_dict(state["optimizer"])
            scheduler.load_state_dict(state["scheduler"])
            prompt_rng.setstate(state["prompt_rng"])
            restore_rank_rng(state["rank_rng"][rank], device)
            step = state["step"]
            if step >= cfg["steps"]:
                raise ValueError("The checkpoint already reached the training step limit")
        if not resume:
            initial = distributed_evaluate(
                policy,
                tokenizer,
                judge,
                target,
                prompts["validation"],
                cfg["validation_samples_per_prompt"],
                cfg,
                output,
                "validation-0",
                corpus_windows,
                device,
            )
            if rank == 0:
                run.log({f"validation/{k}": v for k, v in initial.items()}, step=0)
        initial_step = step
        reason = "step_limit"
        while step < cfg["steps"]:
            signal_flag = torch.tensor(int(stop_requested[0]), device=device)
            dist.all_reduce(signal_flag, op=dist.ReduceOp.MAX)
            should_stop = broadcast_object(
                time.monotonic() - launched >= cfg["train_budget_seconds"]
                or bool(signal_flag.item())
                if rank == 0
                else None
            )
            if should_stop:
                reason = "time_budget" if not signal_flag.item() else "signal"
                break
            iteration = time.monotonic()
            cases = broadcast_object(
                prompt_rng.sample(prompts["train"], world) if rank == 0 else None
            )
            local_rows = rollouts(
                actor, tokenizer, [cases[rank]], cfg["global_group_size"] // world, cfg
            )
            scores, quality = score_rows(local_rows, judge)
            credit, metrics = global_credit(
                local_rows, quality, target, cfg["distribution_weight"], device
            )
            metrics.update(
                distributed_update(
                    policy,
                    optimizer,
                    local_rows,
                    credit,
                    tokenizer.pad_token_id,
                    cfg["gradient_microbatch_size"],
                    cfg["reference_kl_beta"],
                )
            )
            scheduler.step()
            step += 1
            elapsed = torch.tensor(time.monotonic() - iteration, device=device)
            dist.all_reduce(elapsed, op=dist.ReduceOp.MAX)
            more = gather_objects(
                {
                    "terminated": sum(r["terminated"] for r in local_rows),
                    "words": sum(len(words(r["passage"])) for r in local_rows),
                    "judge_dimensions": {
                        key: sum(s["scores"][key] for s in scores)
                        for key in ("coherence", "fluency", "prompt_fulfillment")
                    },
                }
            )
            if rank == 0:
                metrics.update(
                    step=step,
                    training_passages=step * cfg["global_group_size"],
                    seconds_per_update=elapsed.item(),
                    elapsed_seconds=time.monotonic() - launched,
                    learning_rate=scheduler.get_last_lr()[0],
                    terminated_rate=sum(r["terminated"] for r in more) / cfg["global_group_size"],
                    mean_word_count=sum(r["words"] for r in more) / cfg["global_group_size"],
                )
                for dimension in more[0]["judge_dimensions"]:
                    metrics[dimension] = (
                        sum(r["judge_dimensions"][dimension] for r in more)
                        / cfg["global_group_size"]
                    )
                with (output / "history.jsonl").open("a") as handle:
                    handle.write(json.dumps(metrics) + "\n")
                run.log({f"rl/{k}": v for k, v in metrics.items()}, step=step)
                print(json.dumps(metrics), flush=True)
            with (output / f"rollouts-rank-{rank}.jsonl").open("a") as handle:
                for row, score, advantage in zip(local_rows, scores, credit.tolist()):
                    handle.write(
                        json.dumps(
                            {"step": step, **row, "judge": score, "advantage": advantage},
                            ensure_ascii=False,
                        )
                        + "\n"
                    )
            if step % cfg["validation_every"] == 0:
                validation = distributed_evaluate(
                    policy,
                    tokenizer,
                    judge,
                    target,
                    prompts["validation"],
                    cfg["validation_samples_per_prompt"],
                    cfg,
                    output,
                    f"validation-{step}",
                    corpus_windows,
                    device,
                )
                if rank == 0:
                    run.log({f"validation/{k}": v for k, v in validation.items()}, step=step)
            if step % cfg["checkpoint_every"] == 0:
                save_checkpoint(
                    policy,
                    tokenizer,
                    optimizer,
                    scheduler,
                    prompt_rng,
                    step,
                    output,
                    config_hash,
                    manifest,
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
            config_hash,
            manifest,
            device,
        )
        baseline = distributed_evaluate(
            policy,
            tokenizer,
            judge,
            target,
            prompts["test"],
            cfg["test_samples_per_prompt"],
            cfg,
            output,
            "baseline",
            corpus_windows,
            device,
            initial=True,
        )
        final = distributed_evaluate(
            policy,
            tokenizer,
            judge,
            target,
            prompts["test"],
            cfg["test_samples_per_prompt"],
            cfg,
            output,
            "final",
            corpus_windows,
            device,
        )
        if rank == 0:
            actor.save_pretrained(output / "adapter")
            tokenizer.save_pretrained(output / "adapter")
            comparison = {
                "baseline": baseline,
                "final": final,
                "training_updates": step,
                "training_passages": step * cfg["global_group_size"],
                "updates_this_allocation": step - initial_step,
                "stop_reason": reason,
                "elapsed_seconds": time.monotonic() - launched,
                "world_size": world,
                "gpu_hours_upper_bound": world * cfg["allocation_seconds"] / 3600,
                "human_validated": False,
                "limitations": [
                    "One seed; same reward/evaluation judge; no human validation.",
                    "Broader prompts differ from the earlier 100-update pilot.",
                    "Sparse empirical KL remains biased and batch-size dependent.",
                ],
            }
            write_json(output / "comparison.json", comparison)
            run.summary.update({f"baseline/{k}": v for k, v in baseline.items()})
            run.summary.update({f"final/{k}": v for k, v in final.items()})
            run.summary.update(
                {k: v for k, v in comparison.items() if k not in ("baseline", "final")}
            )
            import wandb

            artifact = wandb.Artifact(f"passage-ddp-results-{run.id}", type="rl-results")
            for path in output.iterdir():
                if path.is_file():
                    artifact.add_file(str(path), name=path.name)
            artifact.add_dir(str(output / "adapter"), name="adapter")
            run.log_artifact(artifact)
            print(json.dumps(comparison), flush=True)
        dist.barrier()
    finally:
        if rank == 0 and run is not None:
            import sys

            run.finish(exit_code=1 if sys.exc_info()[0] else 0)
        dist.destroy_process_group()


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--resume", type=Path)
    args = parser.parse_args()
    train(args.output.resolve(), args.resume.resolve() if args.resume else None)
