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
    PassageDistribution,
    config,
    log_results,
    quality_advantages,
    read_rows,
    start_tracking,
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


def rollouts(actor, tokenizer, cases, count, cfg):
    jobs = [(case, sample) for case in cases for sample in range(count)]
    result = []
    generation = GenerationConfig(
        do_sample=True,
        temperature=1.0,
        top_p=1.0,
        top_k=0,
        repetition_penalty=1.0,
        max_new_tokens=cfg["max_new_tokens"],
        pad_token_id=tokenizer.pad_token_id,
        eos_token_id=tokenizer.eos_token_id,
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
            terminated = tokenizer.eos_token_id in completion
            if terminated:
                completion = completion[: completion.index(tokenizer.eos_token_id) + 1]
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
        [{"passage": row["passage"], "quality": float(q)} for row, q in zip(rows, quality)]
    )
    flat = [r for part in parts for r in part]
    if len({len(part) for part in parts}) != 1:
        raise ValueError("Ranks must have equal local batch sizes")
    metrics = None
    if dist.get_rank() == 0:
        quality = np.array([r["quality"] for r in flat])
        fidelity = target.credit([r["passage"] for r in flat])
        qa = quality_advantages(quality)
        da = weight * fidelity["advantages"]
        credit = torch.tensor(qa + da, dtype=torch.float32, device=device)
        metrics = {
            "quality_reward": float(quality.mean()),
            **target.metrics(fidelity),
            "quality_advantage_rms": float(np.sqrt(np.mean(qa**2))),
            "distribution_advantage_rms": float(np.sqrt(np.mean(da**2))),
            "joint_reward": float(quality.mean()) - weight * fidelity["kl"],
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
        judgments = audit.score(rows) if audit is not None else [None] * len(rows)
    local = [
        {**row, **score, "independent_audit": judgment}
        for row, score, judgment in zip(rows, scores, judgments)
    ]
    records = [r for part in gather_objects(local) for r in part]
    metrics = None
    if dist.get_rank() == 0:
        fidelity = target.credit([r["passage"] for r in records])
        metrics = {
            "quality_reward_mean": statistics.mean(r["quality_reward"] for r in records),
            "raw_preference_reward_mean": statistics.mean(r["raw_reward"] for r in records),
            **target.metrics(fidelity),
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
    if not json.loads((reward_output / "checks.json").read_text())["passed"]:
        raise RuntimeError("Human-pair reward validation gate did not pass")
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
        reward = FrozenPreferenceReward(cfg, reward_output, local_rank)
        cases = {
            name: read_rows(ROOT / f"data/policy_{name}.jsonl")
            for name in ("train", "validation", "test")
        }
        references = [r["response"] for r in read_rows(ROOT / "data/reference.jsonl")]
        target = PassageDistribution(references, order=1)
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
                (time.monotonic() - started > cfg["train_budget_seconds"] or bool(flag.item()))
                if rank == 0
                else None
            )
            if finished:
                reason = "time_budget_or_signal"
                break
            iteration = time.monotonic()
            selected = broadcast_object(
                prompt_rng.sample(cases["train"], world) if rank == 0 else None
            )
            rows = rollouts(
                actor, tokenizer, [selected[rank]], cfg["global_group_size"] // world, cfg
            )
            scores, quality = reward.score(rows)
            credit, metrics = global_credit(rows, quality, target, weight, device)
            metrics.update(
                distributed_update(
                    policy,
                    optimizer,
                    rows,
                    credit,
                    tokenizer.pad_token_id,
                    cfg["gradient_microbatch_size"],
                    cfg["reference_kl_beta"],
                )
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
        audit = IndependentAudit(local_rank)
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
                "baseline": baseline,
                "final": final,
                "training_updates": step,
                "training_passages": step * cfg["global_group_size"],
                "stop_reason": reason,
                "elapsed_seconds": time.monotonic() - started,
                "human_validated": False,
                "reward_trained_on_human_preferences": True,
                "limitations": [
                    "One training seed; generated outputs lack human evaluation.",
                    "Auditor is a separate same-family 7B model, not human ground truth.",
                    "Finite-batch KL is biased and sample-count dependent.",
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
    parser.add_argument("--arm", choices=["quality-only", "quality-unigram"], required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    train(args.arm, args.output.resolve())
