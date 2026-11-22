"""Train only energy-distance fidelity plus initial-policy KL on four fixed questions."""

import datetime
import hashlib
import json
import os
import random
import signal
import statistics
import sys
import time
from pathlib import Path

import torch
import torch.distributed as dist
from peft import LoraConfig, get_peft_model
from torch.nn.parallel import DistributedDataParallel as DDP
from transformers import get_cosine_schedule_with_warmup

ROOT = Path(__file__).resolve().parent
if not (ROOT / "common.py").exists():
    sys.path.insert(0, str(ROOT.parent / "helpsteer-rl-v2"))
    sys.path.insert(0, str(ROOT.parent / "helpsteer-semantic-mmd"))
from common import (  # noqa: E402
    make_distribution,
    read_rows,
    start_tracking,
    words,
    write_json,
    write_rows,
)
from ddp_passage_rl import (  # noqa: E402
    SequencePolicy,
    broadcast_object,
    distributed_update,
    gather_objects,
    save_checkpoint,
)
from diagnostic import comparison, select_cases, zero_quality  # noqa: E402
from judge import load_model  # noqa: E402
from semantic_runtime import ConditionalSemanticTarget  # noqa: E402
from train_policy import global_credit, rollouts  # noqa: E402


def evaluate(policy, tokenizer, target, scopes, cfg, output, label, device):
    result = {}
    for scope, cases in scopes.items():
        count = cfg["eval_seen_samples"] if scope == "seen" else cfg["eval_unseen_samples"]
        with torch.random.fork_rng(devices=[device.index]):
            torch.manual_seed(cfg["seed"] + 9000 + dist.get_rank() * 100003)
            rows = rollouts(policy.module.actor, tokenizer, cases[dist.get_rank() :: 2], count, cfg)
            rows = target.embed_records(rows)
        records = [r for part in gather_objects(rows) for r in part]
        metrics = target.diagnostics(records)
        metrics.update(
            passages=len(records),
            terminated_rate=statistics.mean(r["terminated"] for r in records),
            mean_word_count=statistics.mean(len(words(r["passage"])) for r in records),
            per_question=target.per_prompt_metrics(records),
        )
        if dist.get_rank() == 0:
            write_rows(output / f"{label}-{scope}-samples.jsonl", records)
            write_json(output / f"{label}-{scope}-metrics.json", metrics)
        result[scope] = metrics
    return result


def train():
    cfg = json.loads((ROOT / "config.json").read_text())
    rank = int(os.environ["LOCAL_RANK"])
    torch.cuda.set_device(rank)
    device = torch.device("cuda", rank)
    dist.init_process_group("nccl", timeout=datetime.timedelta(minutes=10))
    if dist.get_world_size() != 2:
        raise ValueError("Diagnostic requires exactly two GPUs")
    torch.set_num_threads(4)
    started = time.time()
    output = ROOT / f"outputs/distribution-only-{os.environ['SLURM_JOB_ID']}"
    if rank == 0:
        output.mkdir(exist_ok=False)
    dist.barrier()
    run = None
    try:
        if rank == 0:
            run = start_tracking(
                cfg,
                output,
                "distribution-only-diagnostic",
                quality_weight_actual=0,
                reward_model_loaded=False,
                distribution_weight_actual=cfg["distribution_weight"],
            )
        torch.manual_seed(cfg["seed"])
        tokenizer, base = load_model(cfg["model"], cfg["revision"], device=rank)
        actor = get_peft_model(
            base,
            LoraConfig(
                task_type="CAUSAL_LM",
                r=cfg["lora_rank"],
                lora_alpha=cfg["lora_alpha"],
                lora_dropout=0,
                target_modules=["q_proj", "k_proj", "v_proj", "o_proj"],
                bias="none",
            ),
        ).eval()
        policy = DDP(
            SequencePolicy(actor),
            device_ids=[rank],
            output_device=rank,
            broadcast_buffers=False,
            gradient_as_bucket_view=True,
        )
        selection = json.loads((ROOT / "data/diagnostic-selection.json").read_text())
        scopes = {
            "seen": select_cases(
                read_rows(ROOT / "data/policy_train.jsonl"), selection["seen_ids"]
            ),
            "unseen": select_cases(
                read_rows(ROOT / "data/policy_validation.jsonl"), selection["unseen_ids"]
            ),
        }
        references = [r["response"] for r in read_rows(ROOT / "data/reference.jsonl")]
        target = ConditionalSemanticTarget(cfg, make_distribution(cfg, references), device)
        optimizer = torch.optim.AdamW(
            [p for p in actor.parameters() if p.requires_grad],
            lr=cfg["learning_rate"],
            weight_decay=cfg["weight_decay"],
        )
        scheduler = get_cosine_schedule_with_warmup(optimizer, cfg["warmup_steps"], cfg["steps"])
        prompt_rng = random.Random(cfg["seed"])
        torch.manual_seed(cfg["seed"] + rank * 100003)
        random.seed(cfg["seed"] + rank * 100003)
        stop = [False]
        signal.signal(signal.SIGUSR1, lambda *_: stop.__setitem__(0, True))
        baseline = evaluate(policy, tokenizer, target, scopes, cfg, output, "baseline", device)
        if rank == 0:
            run.log(
                {
                    f"baseline/{scope}/{k}": v
                    for scope, m in baseline.items()
                    for k, v in m.items()
                    if isinstance(v, (int, float))
                },
                step=0,
            )
        step, reason = 0, "step_limit"
        while step < cfg["steps"]:
            flag = torch.tensor(int(stop[0]), device=device)
            dist.all_reduce(flag, op=dist.ReduceOp.MAX)
            finished = broadcast_object(
                (
                    bool(flag.item())
                    or time.time() >= float(os.environ["POLICY_TRAINING_DEADLINE_UNIX"])
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
            # Every update sees exactly the same four training questions.
            rows = rollouts(
                actor, tokenizer, scopes["seen"][rank::2], cfg["samples_per_question"], cfg
            )
            rows = target.embed_records(rows)
            credit, metrics = global_credit(
                rows, zero_quality(rows), target, cfg["distribution_weight"], device
            )
            if metrics["quality_advantage_rms"] != 0:
                raise RuntimeError("Quality leaked into the distribution-only reward")
            update = distributed_update(
                policy,
                optimizer,
                rows,
                credit,
                tokenizer.pad_token_id,
                cfg["gradient_microbatch_size"],
                cfg["reference_kl_beta"],
                reference_kl_centering="prompt-loo",
                loss_divisor=cfg["objective_loss_divisor"],
                record_log_probs=True,
            )
            per_answer = update.pop("_row_metrics")
            scheduler.step()
            step += 1
            metrics.update(update)
            metrics.update(
                step=step,
                training_passages=step * cfg["global_group_size"],
                seconds_per_update=time.monotonic() - iteration,
                elapsed_seconds=time.time() - started,
                quality_weight_actual=0,
            )
            with (output / f"rollouts-rank-{rank}.jsonl").open("a") as handle:
                for row, extra, advantage in zip(rows, per_answer, credit.tolist()):
                    handle.write(
                        json.dumps({"step": step, **row, **extra, "advantage": advantage}) + "\n"
                    )
            if rank == 0:
                with (output / "history.jsonl").open("a") as handle:
                    handle.write(json.dumps(metrics) + "\n")
                run.log({f"rl/{k}": v for k, v in metrics.items()}, step=step)
                print(json.dumps(metrics), flush=True)
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
                    {"arm": "distribution-only-diagnostic"},
                    device,
                )
            if step % cfg["validation_every"] == 0:
                progress = evaluate(
                    policy,
                    tokenizer,
                    target,
                    {"seen": scopes["seen"]},
                    cfg,
                    output,
                    f"progress-{step}",
                    device,
                )
                if rank == 0:
                    run.log(
                        {
                            f"progress/seen/{k}": v
                            for k, v in progress["seen"].items()
                            if isinstance(v, (int, float))
                        },
                        step=step,
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
            {"arm": "distribution-only-diagnostic"},
            device,
        )
        final = evaluate(policy, tokenizer, target, scopes, cfg, output, "final", device)
        if rank == 0:
            report = {
                **comparison(baseline, final),
                "training_updates": step,
                "training_passages": step * cfg["global_group_size"],
                "stop_reason": reason,
                "elapsed_seconds": time.time() - started,
                "quality_weight_actual": 0,
                "initial_policy_kl_beta": cfg["reference_kl_beta"],
            }
            write_json(output / "comparison.json", report)
            run.summary.update(
                {
                    f"final/{scope}/{k}": v
                    for scope, m in final.items()
                    for k, v in m.items()
                    if isinstance(v, (int, float))
                }
            )
            import wandb

            artifact = wandb.Artifact(f"distribution-diagnostic-{run.id}", type="results")
            for path in output.iterdir():
                if path.is_file() and path.suffix in (".json", ".jsonl"):
                    artifact.add_file(str(path), name=path.name)
            artifact.add_dir(str(output / f"checkpoint-{step}/adapter"), name="adapter")
            run.log_artifact(artifact)
            print(json.dumps(report), flush=True)
    finally:
        if run is not None:
            run.finish()
        dist.destroy_process_group()


if __name__ == "__main__":
    train()
