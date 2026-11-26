"""Choose a fixed length coefficient by training-only credit-scale replay, CPU only."""

import argparse
import hashlib
import json
from collections import defaultdict
from pathlib import Path

import numpy as np
from common import read_rows
from length_reward import apply_penalty


def calibrate(source):
    if source.get("selection_uses_test_generations") is not False:
        raise ValueError("Length calibration must explicitly exclude test generations")
    if source.get("source_steps") != [1, 50]:
        raise ValueError("Use only the predeclared early training window, updates 1--50")
    groups = defaultdict(list)
    for row in source["records"]:
        if not 1 <= row["step"] <= 50:
            raise ValueError("Row outside the calibration window")
        groups[row["step"], row["case_id"]].append(row)
    if len(groups) != 200 or any(len(group) != 4 for group in groups.values()):
        raise ValueError("Need 50 training updates, four questions and four answers each")
    quality_credit, length_credit = [], []
    for group in groups.values():
        q = np.array([r["quality_reward"] for r in group], dtype=float)
        lengths = np.array([r["word_count"] for r in group], dtype=float)
        units = (np.maximum(250 - lengths, 0) + np.maximum(lengths - 350, 0)) / 100
        if not np.isfinite(q).all() or not np.isfinite(lengths).all() or (lengths < 0).any():
            raise ValueError("Nonfinite score or invalid length")
        quality_credit.extend(q - (q.sum() - q) / 3)
        length_credit.extend(-(units - (units.sum() - units) / 3))
    qa, la = np.asarray(quality_credit), np.asarray(length_credit)
    qr, lr = float(np.sqrt(np.mean(qa**2))), float(np.sqrt(np.mean(la**2)))
    if qr <= 0 or lr <= 0:
        raise ValueError("Need nonzero quality and length credit to calibrate their relative scale")
    candidates = []
    for weight in (0.025, 0.05, 0.1, 0.2, 0.4):
        candidates.append(
            {
                "weight": weight,
                "length_credit_rms": weight * lr,
                "length_to_quality_credit_rms_ratio": weight * lr / qr,
                "combined_quality_credit_rms": float(np.sqrt(np.mean((qa + weight * la) ** 2))),
            }
        )
    eligible = [row for row in candidates if row["length_credit_rms"] <= qr]
    if not eligible:
        raise ValueError("No candidate meets the predeclared non-dominance criterion")
    return {
        "source_job_id": source["source_job_id"],
        "source_steps": [1, 50],
        "training_answers": len(source["records"]),
        "training_questions_groups": len(groups),
        "min_words": 250,
        "max_words": 350,
        "unit_words": 100,
        "raw_quality_credit_rms": qr,
        "unit_length_credit_rms": lr,
        "selected_weight": max(row["weight"] for row in eligible),
        "candidates": candidates,
        "selection_rule": "Largest candidate whose prompt-LOO length-credit RMS does not exceed "
        "the original prompt-LOO quality-credit RMS on training updates 1--50.",
        "selection_uses_test_generations": False,
        "fresh_generations": 0,
        "gpu_hours_used": 0,
        "source_rollout_sha256": source["source_rollout_sha256"],
        "limitations": [
            "Coefficient-scale heuristic, not evidence of policy improvement or an optimal weight.",
            "Replay uses changing early policies of a prior run, not a fresh fixed-base sample.",
            "Credit RMS is not gradient norm or estimator variance.",
        ],
    }


def write_calibration(training_path, output, validation_path=None):
    source = json.loads(training_path.read_text())
    report = calibrate(source)
    report["training_replay_sha256"] = hashlib.sha256(training_path.read_bytes()).hexdigest()
    if validation_path is not None:
        # This is a frozen-weight sanity check, not selection or a new policy evaluation.
        rows = read_rows(validation_path)
        quality = np.array([r["quality_reward"] for r in rows])
        adjusted, details = apply_penalty(
            rows, quality, {"length_penalty_weight": report["selected_weight"]}
        )
        report["validation_replay_sanity_check"] = {
            "answers": len(rows),
            "quality_before_length_penalty": float(quality.mean()),
            "counterfactual_adjusted_reward": float(adjusted.mean()),
            "mean_length_penalty": float(np.mean([r["length_penalty"] for r in details])),
            "length_in_range_rate": float(np.mean([r["length_in_range"] for r in details])),
            "source_sha256": hashlib.sha256(validation_path.read_bytes()).hexdigest(),
            "used_to_select_weight": False,
        }
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text(json.dumps(report, indent=2) + "\n")
    return report


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--training-replay", type=Path, required=True)
    parser.add_argument("--validation-replay", type=Path)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    report = write_calibration(args.training_replay, args.output, args.validation_replay)
    print(json.dumps({k: v for k, v in report.items() if k != "source_rollout_sha256"}, indent=2))
