"""Fit a scalar human-preference reward using the Bradley-Terry likelihood."""

import argparse
import json
import random
import signal
import time
from pathlib import Path

import torch
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
)
from peft.utils.save_and_load import load_peft_weights, set_peft_model_state_dict
from transformers import get_cosine_schedule_with_warmup


def batch_scores(model, tokenizer, rows):
    sequences = [r["ids_1"] for r in rows] + [r["ids_2"] for r in rows]
    scores = sequence_scores(model, tokenizer, sequences)
    return scores[: len(rows)], scores[len(rows) :]


def evaluate(model, tokenizer, rows):
    all_one, all_two = [], []
    with torch.inference_mode():
        for start in range(0, len(rows), 8):
            one, two = batch_scores(model, tokenizer, rows[start : start + 8])
            all_one.extend(one.tolist())
            all_two.extend(two.tolist())
    one, two = torch.tensor(all_one), torch.tensor(all_two)
    strengths = torch.tensor([r["preference_strength"] for r in rows])
    selected = strengths != 0
    correct = ((two[selected] > one[selected]) == (strengths[selected] > 0)).float()
    return {
        "pairwise_loss": float(bt_loss(one, two, strengths)),
        "non_tie_accuracy": float(correct.mean()),
        "non_tie_pairs": int(selected.sum()),
        "pairs": len(rows),
        "mean_score": float(torch.cat((one, two)).mean()),
        "std_score": float(torch.cat((one, two)).std(unbiased=False)),
    }


def train(output):
    cfg = config()
    if not torch.cuda.is_available():
        raise RuntimeError("Train only in a scheduled GPU job")
    if output.exists() and any(output.iterdir()):
        raise FileExistsError(output)
    output.mkdir(parents=True, exist_ok=True)
    torch.set_num_threads(8)
    torch.manual_seed(cfg["seed"])
    rng = random.Random(cfg["seed"])
    train_rows = read_rows(ROOT / "data/reward_train.jsonl")
    validation = read_rows(ROOT / "data/reward_validation.jsonl")
    started = time.monotonic()
    stop = [False]
    signal.signal(signal.SIGUSR1, lambda *_: stop.__setitem__(0, True))
    with start_tracking(cfg, output, "bradley-terry-reward", human_preference_labels=True) as run:
        tokenizer, model = reward_model(cfg, 0)
        initial = evaluate(model, tokenizer, validation)
        write_json(output / "initial-validation.json", initial)
        run.log({f"validation/{k}": v for k, v in initial.items()}, step=0)
        optimizer = torch.optim.AdamW(
            [p for p in model.parameters() if p.requires_grad],
            lr=cfg["reward_learning_rate"],
            weight_decay=cfg["weight_decay"],
        )
        batch_size = cfg["reward_pair_batch_size"]
        total_steps = cfg["reward_epochs"] * ((len(train_rows) + batch_size - 1) // batch_size)
        scheduler = get_cosine_schedule_with_warmup(
            optimizer, max(1, total_steps // 20), total_steps
        )
        best, best_step, step = float("inf"), 0, 0
        for epoch in range(cfg["reward_epochs"]):
            order = list(train_rows)
            rng.shuffle(order)
            for start in range(0, len(order), batch_size):
                if stop[0] or time.monotonic() - started > cfg["reward_train_budget_seconds"]:
                    break
                batch = order[start : start + batch_size]
                optimizer.zero_grad(set_to_none=True)
                value = 0.0
                for offset in range(0, len(batch), cfg["reward_pair_microbatch_size"]):
                    chunk = batch[offset : offset + cfg["reward_pair_microbatch_size"]]
                    one, two = batch_scores(model, tokenizer, chunk)
                    strengths = torch.tensor(
                        [r["preference_strength"] for r in chunk], device=one.device
                    )
                    loss = bt_loss(one, two, strengths) * len(chunk) / len(batch)
                    if not torch.isfinite(loss):
                        raise FloatingPointError("Nonfinite Bradley-Terry loss")
                    loss.backward()
                    value += loss.detach().item()
                norm = torch.nn.utils.clip_grad_norm_(
                    [p for p in model.parameters() if p.requires_grad], 1.0
                )
                if not torch.isfinite(norm):
                    raise FloatingPointError("Nonfinite reward-model gradient")
                optimizer.step()
                scheduler.step()
                step += 1
                metrics = {
                    "step": step,
                    "epoch": epoch,
                    "training_loss": value,
                    "gradient_norm": float(norm),
                    "elapsed_seconds": time.monotonic() - started,
                }
                with (output / "history.jsonl").open("a") as handle:
                    handle.write(json.dumps(metrics) + "\n")
                run.log({f"reward/{k}": v for k, v in metrics.items()}, step=step)
                print(json.dumps(metrics), flush=True)
                if step % cfg["reward_validation_every"] == 0 or step == total_steps:
                    results = evaluate(model, tokenizer, validation)
                    run.log({f"validation/{k}": v for k, v in results.items()}, step=step)
                    write_json(output / f"validation-{step}.json", results)
                    if results["pairwise_loss"] < best:
                        best, best_step = results["pairwise_loss"], step
                        model.save_pretrained(output / "adapter")
                        tokenizer.save_pretrained(output / "adapter")
            if stop[0] or time.monotonic() - started > cfg["reward_train_budget_seconds"]:
                break
        final_validation = evaluate(model, tokenizer, validation)
        if final_validation["pairwise_loss"] < best:
            best, best_step = final_validation["pairwise_loss"], step
            model.save_pretrained(output / "adapter")
            tokenizer.save_pretrained(output / "adapter")
        torch.save(
            {
                "step": step,
                "optimizer": optimizer.state_dict(),
                "scheduler": scheduler.state_dict(),
                "torch_rng": torch.get_rng_state(),
                "cuda_rng": torch.cuda.get_rng_state_all(),
                "prompt_rng": rng.getstate(),
            },
            output / "final-training-state.pt",
        )
        set_peft_model_state_dict(
            model, load_peft_weights(str(output / "adapter"), device="cuda:0")
        )
        selected = evaluate(model, tokenizer, validation)
        passed = selected["non_tie_accuracy"] > 0.5 and selected["std_score"] > 1e-6 and step > 0
        calibration = {
            "center": selected["mean_score"],
            "scale": max(selected["std_score"], 1e-6),
            "transform": "sigmoid((raw_score-center)/scale); not calibrated human win probability",
        }
        write_json(output / "calibration.json", calibration)
        report = {
            "passed": passed,
            "training_updates": step,
            "selected_step": best_step,
            "initial_validation": initial,
            "selected_validation": selected,
            "elapsed_seconds": time.monotonic() - started,
            "criterion": (
                "Non-tied held-out human-pair accuracy >50%; technical gate, "
                "not human validation of policy outputs"
            ),
        }
        write_json(output / "checks.json", report)
        run.summary.update(report)
        log_results(run, output)
        if not passed:
            raise RuntimeError("Reward validation gate failed; dependent RL must not run")
        write_json(ROOT / "outputs/reward-ready.json", {"path": str(output), "wandb_url": run.url})
        print(json.dumps(report), flush=True)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", type=Path, required=True)
    train(parser.parse_args().output.resolve())
