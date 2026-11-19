"""Exploratory passage-level REINFORCE: frozen quality judge + aggregate bigram KL."""

import argparse
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
import wandb
from judge import load_model
from passage_reward import PassageDistribution, quality_advantages
from peft import LoraConfig, get_peft_model
from pilot import ROOT, duplicate_window_rate, words, write_json
from quality import FrozenQualityJudge
from transformers import GenerationConfig, get_cosine_schedule_with_warmup

from stochastic_llm.rl import completion_log_probs


def task_prompt(case):
    return (
        "Write an original dramatic monologue of 80 to 120 words in an intelligible "
        "Shakespearean register. Develop the situation coherently and make the decision "
        "clear. Do not quote an existing play. Output only the passage.\nSituation: "
        + case["prompt"]
    )


def rollouts(actor, tokenizer, cases, count_per_case, cfg):
    jobs = []
    for case in cases:
        prompt = task_prompt(case)
        prompt_ids = tokenizer.apply_chat_template(
            [{"role": "user", "content": prompt}], tokenize=True, add_generation_prompt=True
        )
        if len(prompt_ids) + cfg["max_new_tokens"] > 1024:
            raise ValueError("Prompt exceeds passage token budget")
        for sample_index in range(count_per_case):
            jobs.append((case["id"], prompt, prompt_ids, sample_index))
    actor.eval()
    rows = []
    generation = GenerationConfig(
        do_sample=True,
        temperature=1.0,
        top_p=1.0,
        top_k=0,
        repetition_penalty=1.0,
        max_new_tokens=cfg["max_new_tokens"],
        eos_token_id=tokenizer.eos_token_id,
        pad_token_id=tokenizer.pad_token_id,
    )
    for start in range(0, len(jobs), cfg["rollout_batch_size"]):
        chunk = jobs[start : start + cfg["rollout_batch_size"]]
        width = max(len(r[2]) for r in chunk)
        ids = [[tokenizer.pad_token_id] * (width - len(r[2])) + r[2] for r in chunk]
        attention = [[0] * (width - len(r[2])) + [1] * len(r[2]) for r in chunk]
        with torch.inference_mode():
            output = actor.generate(
                input_ids=torch.tensor(ids, device=actor.device),
                attention_mask=torch.tensor(attention, device=actor.device),
                generation_config=generation,
            )
        for job, completion in zip(chunk, output[:, width:].cpu().tolist()):
            terminated = tokenizer.eos_token_id in completion
            if terminated:
                completion = completion[: completion.index(tokenizer.eos_token_id) + 1]
            passage = tokenizer.decode(completion, skip_special_tokens=True).strip()
            rows.append(
                {
                    "id": f"{job[0]}-{job[3]}",
                    "case_id": job[0],
                    "prompt": job[1],
                    "prompt_ids": job[2],
                    "completion_ids": completion,
                    "passage": passage,
                    "terminated": terminated,
                }
            )
    return rows


def score_rows(rows, judge):
    results = judge.score(rows)
    quality = np.array([r["quality"] for r in results], dtype=np.float64)
    # A truncated answer still contributes every observed bigram; it receives no
    # positive quality reward, rather than being silently removed from the group.
    quality *= np.array([r["terminated"] for r in rows], dtype=np.float64)
    return results, quality


def update(actor, optimizer, rows, credit, tokenizer, cfg):
    optimizer.zero_grad(set_to_none=True)
    actor.eval()
    loss_sum, reference_sum = 0.0, 0.0
    for start in range(0, len(rows), cfg["gradient_microbatch_size"]):
        chunk = rows[start : start + cfg["gradient_microbatch_size"]]
        with torch.no_grad(), actor.disable_adapter():
            reference = completion_log_probs(actor, chunk, tokenizer.pad_token_id)
        current = completion_log_probs(actor, chunk, tokenizer.pad_token_id)
        delta = current.detach() - reference
        advantage = (
            torch.tensor(
                credit[start : start + len(chunk)], device=actor.device, dtype=current.dtype
            )
            - cfg["reference_kl_beta"] * delta
        )
        loss = -(advantage.detach() * current).sum() / len(rows)
        if not torch.isfinite(loss):
            raise FloatingPointError("Nonfinite passage policy loss")
        loss.backward()
        loss_sum += loss.detach().item()
        reference_sum += delta.sum().item()
    norm = torch.nn.utils.clip_grad_norm_(
        [p for p in actor.parameters() if p.requires_grad], max_norm=1.0
    )
    if not torch.isfinite(norm):
        raise FloatingPointError("Nonfinite passage policy gradient")
    optimizer.step()
    return {
        "policy_gradient_loss": loss_sum,
        "gradient_norm": float(norm),
        "reference_sequence_kl_estimate": reference_sum / len(rows),
    }


def checkpoint(actor, tokenizer, optimizer, scheduler, rng, step, output):
    path = output / f"checkpoint-{step}"
    path.mkdir()
    actor.save_pretrained(path / "adapter")
    tokenizer.save_pretrained(path / "adapter")
    torch.save(
        {
            "step": step,
            "optimizer": optimizer.state_dict(),
            "scheduler": scheduler.state_dict(),
            "rng": rng.getstate(),
            "torch_rng": torch.get_rng_state(),
            "cuda_rng": torch.cuda.get_rng_state_all(),
        },
        path / "state.pt",
    )
    write_json(path / "complete.json", {"step": step, "complete": True})


def evaluate(actor, tokenizer, judge, target, cases, count, cfg, output, label, corpus_windows):
    # Isolate fixed evaluation sampling from training; baseline/final use the same seed.
    with torch.random.fork_rng(devices=[actor.device.index or 0]):
        torch.manual_seed(cfg["seed"] + 9000)
        rows = rollouts(actor, tokenizer, cases, count, cfg)
        scores, quality = score_rows(rows, judge)
    fidelity = target.credit([r["passage"] for r in rows])
    metrics = {
        "quality_reward_mean": float(quality.mean()),
        "bigram_kl": fidelity["kl"],
        "unknown_bigram_rate": fidelity["unknown_bigram_rate"],
        "passages": len(rows),
        "terminated_rate": statistics.mean(r["terminated"] for r in rows),
        "mean_word_count": statistics.mean(len(words(r["passage"])) for r in rows),
        "corpus_12word_overlap": statistics.mean(
            duplicate_window_rate(r["passage"], corpus_windows) or 0.0 for r in rows
        ),
    }
    for dimension in ("coherence", "fluency", "prompt_fulfillment"):
        metrics[dimension] = statistics.mean(s["scores"][dimension] for s in scores)
    write_json(output / f"{label}-metrics.json", metrics)
    with (output / f"{label}-samples.jsonl").open("w") as handle:
        for row, score in zip(rows, scores):
            handle.write(json.dumps({**row, "judge": score}, ensure_ascii=False) + "\n")
    return metrics


def run(output, calibration, benchmark=False, benchmark_timing=None):
    if not torch.cuda.is_available():
        raise RuntimeError("A scheduled CUDA GPU is required")
    if os.environ.get("WANDB_MODE", "online") != "online":
        raise RuntimeError("Online W&B logging is required")
    if os.environ.get("WANDB_DISABLED", "false").lower() in ("true", "1", "yes"):
        raise RuntimeError("W&B cannot be disabled")
    if not json.loads((calibration / "checks.json").read_text())["passed"]:
        raise RuntimeError("Quality-judge calibration must pass before RL")
    cfg = json.loads((ROOT / "passage-config.json").read_text())
    if not benchmark:
        if benchmark_timing is None:
            raise ValueError("A successful timing benchmark is required")
        timing = json.loads(benchmark_timing.read_text())
        if timing["projected_total_seconds"] > cfg["benchmark_max_projected_seconds"]:
            raise RuntimeError("Benchmark projects more than 90 minutes; pilot not started")
    if output.exists() and any(output.iterdir()):
        raise FileExistsError(output)
    output.mkdir(parents=True, exist_ok=True)
    prompts = json.loads((ROOT / "passage-prompts.json").read_text())
    if cfg["group_size"] % cfg["prompts_per_group"]:
        raise ValueError("Group size must divide evenly across prompts")
    manifest = {
        str(p.relative_to(ROOT)): hashlib.sha256(p.read_bytes()).hexdigest()
        for p in [
            ROOT / "passage-config.json",
            ROOT / "passage-prompts.json",
            ROOT / "passage_rl.py",
            ROOT / "passage_reward.py",
            ROOT / "quality.py",
            ROOT / "data/reference.txt",
        ]
    }
    if not benchmark and timing["manifest"] != manifest:
        raise ValueError("Source or configuration differs from the successful benchmark")
    cfg.update(
        {
            "benchmark": benchmark,
            "slurm_job_id": os.environ.get("SLURM_JOB_ID"),
            "calibration": str(calibration),
            "source_hashes": manifest,
            "objective": "mean quality - 0.1 * full-support batch bigram KL",
            "reference_regularizer": "0.002 * sequence KL to initial instruct model",
            "human_validated": False,
        }
    )
    torch.manual_seed(cfg["seed"])
    rng = random.Random(cfg["seed"])
    stop_requested = [False]
    signal.signal(signal.SIGUSR1, lambda *_: stop_requested.__setitem__(0, True))
    started = time.monotonic()
    with wandb.init(
        entity=cfg["wandb_entity"],
        project=cfg["wandb_project"],
        name=f"shakespeare-{'benchmark' if benchmark else 'rl'}-{cfg['slurm_job_id']}",
        group="shakespeare-passage-rl-v1",
        job_type="rl-benchmark" if benchmark else "rl",
        mode="online",
        config=cfg,
        dir=str(output),
        settings=wandb.Settings(base_url=cfg["wandb_base_url"]),
    ) as run:
        write_json(output / "wandb_run.json", {"id": run.id, "url": run.url})
        write_json(output / "run_config.json", cfg)
        artifact = wandb.Artifact(f"passage-rl-source-{run.id}", type="source")
        for path in ROOT.iterdir():
            if path.is_file() and path.suffix in (".py", ".json", ".md", ".sbatch"):
                artifact.add_file(str(path), name=path.name)
        artifact.add_file(str(ROOT / "data/reference.txt"), name="reference.txt")
        artifact.add_file(str(calibration / "checks.json"), name="calibration-checks.json")
        run.log_artifact(artifact)
        tokenizer, base = load_model(cfg["actor_model"], cfg["actor_revision"])
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
        )
        actor.eval()
        judge = FrozenQualityJudge(
            cfg["judge_model"], cfg["judge_revision"], cfg["judge_batch_size"]
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
            weight_decay=0.01,
        )
        steps = 5 if benchmark else cfg["steps"]
        scheduler = get_cosine_schedule_with_warmup(optimizer, max(1, int(0.03 * steps)), steps)
        baseline = None
        if not benchmark:
            baseline = evaluate(
                actor,
                tokenizer,
                judge,
                target,
                prompts["test"],
                cfg["eval_samples_per_prompt"],
                cfg,
                output,
                "baseline",
                corpus_windows,
            )
            run.log({f"baseline/{k}": v for k, v in baseline.items()})
        durations = []
        for step in range(1, steps + 1):
            update_start = time.monotonic()
            cases = rng.sample(prompts["train"], cfg["prompts_per_group"])
            rows = rollouts(actor, tokenizer, cases, cfg["group_size"] // len(cases), cfg)
            scores, quality = score_rows(rows, judge)
            reward = target.credit([r["passage"] for r in rows])
            credits = (
                quality_advantages(quality) + cfg["distribution_weight"] * reward["advantages"]
            )
            loss_metrics = update(actor, optimizer, rows, credits, tokenizer, cfg)
            scheduler.step()
            duration = time.monotonic() - update_start
            durations.append(duration)
            metrics = {
                "step": step,
                "training_passages": step * cfg["group_size"],
                "quality_reward": float(quality.mean()),
                "bigram_kl": reward["kl"],
                "unknown_bigram_rate": reward["unknown_bigram_rate"],
                "joint_reward": float(quality.mean()) - cfg["distribution_weight"] * reward["kl"],
                "terminated_rate": statistics.mean(r["terminated"] for r in rows),
                "seconds_per_update": duration,
                "learning_rate": scheduler.get_last_lr()[0],
                **loss_metrics,
            }
            with (output / "history.jsonl").open("a") as handle:
                handle.write(json.dumps(metrics) + "\n")
            with (output / "rollouts.jsonl").open("a") as handle:
                for row, score, credit in zip(rows, scores, credits):
                    handle.write(
                        json.dumps(
                            {"step": step, **row, "judge": score, "advantage": float(credit)},
                            ensure_ascii=False,
                        )
                        + "\n"
                    )
            run.log({f"rl/{k}": v for k, v in metrics.items()}, step=step)
            print(json.dumps(metrics), flush=True)
            if not benchmark and step % cfg["validation_every"] == 0:
                probe = evaluate(
                    actor,
                    tokenizer,
                    judge,
                    target,
                    prompts["validation"],
                    cfg["validation_samples_per_prompt"],
                    cfg,
                    output,
                    f"validation-{step}",
                    corpus_windows,
                )
                run.log({f"validation/{k}": v for k, v in probe.items()}, step=step)
            if not benchmark and (step % cfg["checkpoint_every"] == 0 or stop_requested[0]):
                checkpoint(actor, tokenizer, optimizer, scheduler, rng, step, output)
            if stop_requested[0]:
                raise RuntimeError("Interrupted after saving the current update")
        if benchmark:
            average = statistics.mean(durations[1:])
            timing = {
                "seconds_per_update": average,
                "timed_updates": len(durations) - 1,
                "projected_update_seconds": average * cfg["steps"],
                "projected_total_seconds": average * cfg["steps"] * 1.3 + 120,
                "projection_note": "30% evaluation/checkpoint margin + 120s loading",
                "elapsed_seconds": time.monotonic() - started,
                "manifest": manifest,
            }
            write_json(output / "timing.json", timing)
            run.summary.update(timing)
            print(json.dumps(timing), flush=True)
        else:
            final = evaluate(
                actor,
                tokenizer,
                judge,
                target,
                prompts["test"],
                cfg["eval_samples_per_prompt"],
                cfg,
                output,
                "final",
                corpus_windows,
            )
            actor.save_pretrained(output / "adapter")
            tokenizer.save_pretrained(output / "adapter")
            write_json(
                output / "comparison.json",
                {
                    "baseline": baseline,
                    "final": final,
                    "training_updates": steps,
                    "training_passages": steps * cfg["group_size"],
                    "elapsed_seconds": time.monotonic() - started,
                    "human_validated": False,
                    "limitations": [
                        "One seed; four held-out prompts; same training/eval judge.",
                        "Batch KL is biased and sample-size dependent.",
                        "No compute-matched SFT or MaD-RL control.",
                    ],
                },
            )
            run.summary.update({f"final/{k}": v for k, v in final.items()})
        results = wandb.Artifact(f"passage-rl-results-{run.id}", type="rl-results")
        for path in output.iterdir():
            if path.is_file():
                results.add_file(str(path), name=path.name)
        if (output / "adapter").exists():
            results.add_dir(str(output / "adapter"), name="adapter")
        run.log_artifact(results)
        print(f"Completed: {run.url}", flush=True)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--calibration", type=Path, required=True)
    parser.add_argument("--benchmark", action="store_true")
    parser.add_argument("--benchmark-timing", type=Path)
    args = parser.parse_args()
    run(args.output.resolve(), args.calibration.resolve(), args.benchmark, args.benchmark_timing)
