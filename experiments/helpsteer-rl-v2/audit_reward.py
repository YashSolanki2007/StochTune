"""Evaluate an existing reward adapter on human labels; never train a policy."""

import argparse
import hashlib
import json
from pathlib import Path

import numpy as np
import torch
from common import (
    ROOT,
    config,
    encoded_response,
    log_results,
    read_rows,
    reward_identity_sha256,
    reward_model,
    sequence_scores,
    start_tracking,
    write_json,
    write_rows,
)
from validation import validate_reward


def score_candidates(model, tokenizer, rows, cfg):
    one, two = [], []
    pairs = max(1, cfg["reward_inference_batch_size"] // 2)
    with torch.inference_mode():
        for start in range(0, len(rows), pairs):
            chunk = rows[start : start + pairs]
            sequences = [r["ids_1"] for r in chunk] + [r["ids_2"] for r in chunk]
            scores = sequence_scores(
                model, tokenizer, sequences, cfg["reward_max_sequence_length"]
            ).tolist()
            one.extend(scores[: len(chunk)])
            two.extend(scores[len(chunk) :])
    return one, two


def majority_direction():
    rows = read_rows(ROOT / "data/reward_train.jsonl")
    signs = [r["preference_strength"] for r in rows if r["preference_strength"]]
    return 1 if sum(s > 0 for s in signs) >= len(signs) / 2 else -1


def evaluate_adapter(model, tokenizer, rows, cfg, output):
    if cfg.get("reward_pretrained_full", False):
        rows = [
            {
                **r,
                "ids_1": encoded_response(tokenizer, r["prompt"], r["response_1"]),
                "ids_2": encoded_response(tokenizer, r["prompt"], r["response_2"]),
            }
            for r in rows
        ]
    one, two = score_candidates(model, tokenizer, rows, cfg)
    report = validate_reward(rows, one, two, majority_direction())
    write_rows(
        output / "pair-predictions.jsonl",
        [
            {
                "id": row["id"],
                "score_1": a,
                "score_2": b,
                **{
                    key: row[key]
                    for key in (
                        "preference_strength",
                        "coherence_1",
                        "coherence_2",
                        "response_tokens_1",
                        "response_tokens_2",
                    )
                },
            }
            for row, a, b in zip(rows, one, two)
        ],
    )
    values = np.array(one + two)
    report["validation_data_sha256"] = hashlib.sha256(
        (ROOT / "data/reward_validation.jsonl").read_bytes()
    ).hexdigest()
    write_json(
        output / "calibration.json",
        {
            "center": float(values.mean()),
            "scale": max(float(values.std()), 1e-6),
            "transform": "sigmoid((raw_score-center)/scale); not human win probabilities",
        },
    )
    return report


def audit(reward_output, output, candidate="legacy"):
    if not torch.cuda.is_available():
        raise RuntimeError("Explicitly approved scheduled GPU allocation required for model audit")
    if output.exists() and any(output.iterdir()):
        raise FileExistsError(output)
    output.mkdir(parents=True, exist_ok=True)
    cfg = config()
    model_config = {}
    if candidate == "skywork":
        model_config = {
            "reward_model_name": "Skywork/Skywork-Reward-V2-Qwen3-8B",
            "reward_model_revision": "6f19fdefb933293d4898bdb59a96f7223d998659",
            "reward_pretrained_full": True,
            "offline_model_loading": True,
            "reward_max_sequence_length": 4096,
        }
        cfg.update(model_config)
    with start_tracking(
        cfg, output, f"reward-validation-v2-{candidate}", policy_training=False, candidate=candidate
    ) as run:
        tokenizer, model = reward_model(cfg, 0, None if model_config else reward_output / "adapter")
        model.requires_grad_(False)
        report = evaluate_adapter(
            model, tokenizer, read_rows(ROOT / "data/reward_validation.jsonl"), cfg, output
        )
        report.update(
            reward_output=str(reward_output),
            policy_training_performed=False,
            candidate=candidate,
            calibration_sha256=hashlib.sha256(
                (output / "calibration.json").read_bytes()
            ).hexdigest(),
        )
        if model_config:
            report.update(
                model_identity_sha256=reward_identity_sha256(model_config),
                upstream_training_overlap="Unknown; public preference data may overlap. "
                "This is validation for engineering use, not a clean benchmark claim.",
            )
        else:
            report["adapter_sha256"] = hashlib.sha256(
                (reward_output / "adapter/adapter_model.safetensors").read_bytes()
            ).hexdigest()
        write_json(output / "checks.json", report)
        run.summary.update(report)
        log_results(run, output, adapter=False)
        # Each concurrent candidate writes its own pointer; the budget controller
        # selects ONE passing scorer before creating the policy's ready marker.
        if report["passed"]:
            pointer = {
                "path": str(output if model_config else reward_output),
                "audit_path": str(output),
                "wandb_url": run.url,
                "reward_kind": "pretrained" if model_config else "adapter",
                "reward_model_config": model_config,
            }
            write_json(output / "reward-ready.json", pointer)
        print(json.dumps(report), flush=True)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--reward-output", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--candidate", choices=["legacy", "skywork"], default="legacy")
    args = parser.parse_args()
    audit(args.reward_output.resolve(), args.output.resolve(), args.candidate)
