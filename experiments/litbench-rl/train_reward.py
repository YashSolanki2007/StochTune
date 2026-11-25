"""Two-rank Bradley-Terry literary reward training inside the shared GPU budget."""

import contextlib
import hashlib
import json
import os
import random
import time

import numpy as np
import torch
import torch.distributed as dist
from common import (
    ROOT,
    bt_loss,
    config,
    log_results,
    read_rows,
    reward_model,
    sequence_scores,
    start_tracking,
    write_json,
    write_rows,
)
from ddp_passage_rl import broadcast_object, gather_objects
from peft.utils.save_and_load import load_peft_weights, set_peft_model_state_dict
from torch.nn.parallel import DistributedDataParallel as DDP
from transformers import get_cosine_schedule_with_warmup
from validation import validate_reward


class RewardPolicy(torch.nn.Module):
    def __init__(self, model):
        super().__init__()
        self.model = model

    def forward(self, sequences, tokenizer, maximum):
        return sequence_scores(self.model, tokenizer, sequences, maximum)


def evaluate(model, tokenizer, rows, cfg):
    rank = dist.get_rank()
    local = rows[rank::2]
    predictions = []
    with torch.inference_mode():
        for offset in range(0, len(local), 2):
            chunk = local[offset : offset + 2]
            seqs = [r["ids_1"] for r in chunk] + [r["ids_2"] for r in chunk]
            scores = sequence_scores(
                model, tokenizer, seqs, cfg["reward_max_sequence_length"]
            ).tolist()
            predictions.extend(
                {"pair_id": r["pair_id"], "one": a, "two": b}
                for r, a, b in zip(chunk, scores[: len(chunk)], scores[len(chunk) :])
            )
    lookup = {r["pair_id"]: r for part in gather_objects(predictions) for r in part}
    one = [lookup[r["pair_id"]]["one"] for r in rows]
    two = [lookup[r["pair_id"]]["two"] for r in rows]
    values = torch.tensor(one)
    other = torch.tensor(two)
    report = validate_reward(rows, one, two)
    report["pairwise_loss"] = float(
        bt_loss(values, other, torch.tensor([r["preference_strength"] for r in rows]))
    )
    return (
        report,
        np.array(one + two),
        [
            {
                "pair_id": r["pair_id"],
                "score_1": a,
                "score_2": b,
                "preference_strength": r["preference_strength"],
                "response_tokens_1": r["response_tokens_1"],
                "response_tokens_2": r["response_tokens_2"],
            }
            for r, a, b in zip(rows, one, two)
        ],
    )


def update(policy, optimizer, tokenizer, rows, cfg):
    optimizer.zero_grad(set_to_none=True)
    value = 0
    for offset in range(0, len(rows), cfg["reward_pair_microbatch_size"]):
        chunk = rows[offset : offset + cfg["reward_pair_microbatch_size"]]
        sync = contextlib.nullcontext() if offset + len(chunk) == len(rows) else policy.no_sync()
        with sync:
            seqs = [r["ids_1"] for r in chunk] + [r["ids_2"] for r in chunk]
            scores = policy(seqs, tokenizer, cfg["reward_max_sequence_length"])
            strengths = torch.tensor(
                [r["preference_strength"] for r in chunk], device=scores.device
            )
            loss = (
                bt_loss(scores[: len(chunk)], scores[len(chunk) :], strengths)
                * len(chunk)
                / len(rows)
            )
            if not torch.isfinite(loss):
                raise FloatingPointError("Nonfinite literary preference loss")
            loss.backward()
        value += float(loss.detach())
    norm = torch.nn.utils.clip_grad_norm_([p for p in policy.parameters() if p.requires_grad], 1)
    if not torch.isfinite(norm):
        raise FloatingPointError("Nonfinite literary reward gradient")
    optimizer.step()
    return value, float(norm)


def train():
    cfg = config()
    rank = int(os.environ["LOCAL_RANK"])
    torch.cuda.set_device(rank)
    dist.init_process_group("nccl")
    torch.set_num_threads(4)
    torch.manual_seed(cfg["seed"])
    if dist.get_world_size() != 2:
        raise ValueError("Need two ranks for literary reward training")
    output = ROOT / f"outputs/reward-{os.environ['SLURM_JOB_ID']}"
    if rank == 0:
        output.mkdir(exist_ok=False)
    dist.barrier()
    run = None
    started = time.time()
    try:
        if rank == 0:
            run = start_tracking(
                cfg, output, "litbench-bradley-terry", human_preference_labels=True
            )
        tokenizer, model = reward_model(cfg, rank)
        policy = DDP(
            RewardPolicy(model), device_ids=[rank], output_device=rank, broadcast_buffers=False
        )
        rows = read_rows(ROOT / "data/reward_train.jsonl")
        validation = read_rows(ROOT / "data/reward_validation.jsonl")
        initial, _, _ = evaluate(model, tokenizer, validation, cfg)
        if rank == 0:
            write_json(output / "initial-validation.json", initial)
            run.log(
                {f"validation/{k}": v for k, v in initial.items() if isinstance(v, (float, int))},
                step=0,
            )
        optimizer = torch.optim.AdamW(
            [p for p in model.parameters() if p.requires_grad],
            lr=cfg["reward_learning_rate"],
            weight_decay=cfg["weight_decay"],
        )
        scheduler = get_cosine_schedule_with_warmup(optimizer, 10, cfg["reward_steps"])
        rng = random.Random(cfg["seed"])
        order = list(rows)
        rng.shuffle(order)
        cursor = 0
        best = float("inf")
        best_step = 0
        step = 0
        while step < cfg["reward_steps"]:
            finished = broadcast_object(
                time.time() >= float(os.environ["REWARD_TRAINING_DEADLINE_UNIX"])
                if rank == 0
                else None
            )
            if finished:
                break
            if cursor + cfg["reward_pair_batch_size"] > len(order):
                rng.shuffle(order)
                cursor = 0
            batch = order[cursor : cursor + cfg["reward_pair_batch_size"]]
            cursor += len(batch)
            if len(batch) != cfg["reward_pair_batch_size"]:
                raise ValueError("Incomplete global preference batch")
            loss, norm = update(policy, optimizer, tokenizer, batch[rank::2], cfg)
            scheduler.step()
            step += 1
            tensor = torch.tensor([loss], device=torch.device("cuda", rank))
            dist.all_reduce(tensor)
            loss = float(tensor.item() / 2)
            if rank == 0:
                metrics = {
                    "step": step,
                    "training_loss": loss,
                    "gradient_norm": norm,
                    "training_pairs": step * len(batch),
                    "elapsed_seconds": time.time() - started,
                }
                with (output / "history.jsonl").open("a") as handle:
                    handle.write(json.dumps(metrics) + "\n")
                run.log({f"reward/{k}": v for k, v in metrics.items()}, step=step)
                print(json.dumps(metrics), flush=True)
            if step % cfg["reward_validation_every"] == 0:
                report, _, _ = evaluate(model, tokenizer, validation, cfg)
                if report["pairwise_loss"] < best:
                    best = report["pairwise_loss"]
                    best_step = step
                    if rank == 0:
                        model.save_pretrained(output / "adapter")
                        tokenizer.save_pretrained(output / "adapter")
                dist.barrier()
                if rank == 0:
                    write_json(output / f"validation-{step}.json", report)
                    run.log(
                        {
                            f"validation/{k}": v
                            for k, v in report.items()
                            if isinstance(v, (int, float))
                        },
                        step=step,
                    )
        report, _, _ = evaluate(model, tokenizer, validation, cfg)
        if report["pairwise_loss"] < best:
            best_step = step
            if rank == 0:
                model.save_pretrained(output / "adapter")
                tokenizer.save_pretrained(output / "adapter")
        dist.barrier()
        set_peft_model_state_dict(
            model, load_peft_weights(str(output / "adapter"), device=f"cuda:{rank}")
        )
        selection, values, _ = evaluate(model, tokenizer, validation, cfg)
        audit, _, predictions = evaluate(
            model, tokenizer, read_rows(ROOT / "data/reward_audit.jsonl"), cfg
        )
        if rank == 0:
            audit["checks"]["at_least_25_training_updates"] = step >= 25
            audit["passed"] = all(audit["checks"].values())
            write_json(
                output / "calibration.json",
                {
                    "center": float(values.mean()),
                    "scale": max(float(values.std()), 1e-6),
                    "transform": "sigmoid z-score; not calibrated human preference probabilities",
                },
            )
            audit.update(
                training_updates=step,
                training_pairs=step * cfg["reward_pair_batch_size"],
                selected_step=best_step,
                initial_validation=initial,
                selected_validation=selection,
                elapsed_seconds=time.time() - started,
                validation_data_sha256=hashlib.sha256(
                    (ROOT / "data/reward_audit.jsonl").read_bytes()
                ).hexdigest(),
                adapter_sha256=hashlib.sha256(
                    (output / "adapter/adapter_model.safetensors").read_bytes()
                ).hexdigest(),
                calibration_sha256=hashlib.sha256(
                    (output / "calibration.json").read_bytes()
                ).hexdigest(),
            )
            write_rows(output / "pair-predictions.jsonl", predictions)
            write_json(output / "checks.json", audit)
            run.summary.update(audit)
            log_results(run, output)
            if audit["passed"]:
                write_json(
                    ROOT / "outputs/reward-ready.json",
                    {
                        "path": str(output),
                        "audit_path": str(output),
                        "wandb_url": run.url,
                        "reward_kind": "adapter",
                        "reward_model_config": {},
                    },
                )
            print(json.dumps(audit), flush=True)
        dist.barrier()
    finally:
        if run is not None:
            import sys

            run.finish(exit_code=1 if sys.exc_info()[0] else 0)
        dist.destroy_process_group()


if __name__ == "__main__":
    train()
