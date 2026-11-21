"""Replay real saved answers on CPU; no generation, model loading, or GPU allocation."""

import argparse
import gzip
import hashlib
import json
from collections import defaultdict
from pathlib import Path

import numpy as np
from mmd import conditional_credit, mmd2, passage_credit

ROOT = Path(__file__).resolve().parent
REPO = ROOT.parents[1]


def read_rows(path):
    opener = gzip.open if path.suffix == ".gz" else open
    with opener(path, "rt") as stream:
        return [json.loads(line) for line in stream if line.strip()]


def sha(path):
    return hashlib.sha256(path.read_bytes()).hexdigest()


def rms(values):
    return float(np.sqrt(np.mean(np.asarray(values) ** 2)))


def write_json(path, value):
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(value, indent=2) + "\n")


def fixed_loss_divisor(quality_weight, distribution_weight, reference_weight):
    weights = np.array([quality_weight, distribution_weight, reference_weight], dtype=float)
    if np.any(weights < 0) or not np.all(np.isfinite(weights)) or weights.sum() <= 0:
        raise ValueError("Need finite nonnegative objective weights with positive total")
    return float(weights.sum())


def calibration_scale(costs, target_level=1.0):
    values = np.asarray(costs, dtype=float)
    if not len(values) or not np.all(np.isfinite(values)) or target_level <= 0:
        raise ValueError("Need finite calibration costs and positive target level")
    initial = float(values.mean())
    if initial <= 1e-8:
        raise ValueError("Mean calibration discrepancy is nonpositive or too small to scale safely")
    return target_level / initial


def audit(source, reference_data, output, calibration_steps=10, write_profiles=False):
    cfg = json.loads((source / "run_config.json").read_text())
    metadata_path = reference_data / "semantic-metadata.json"
    vectors_path = reference_data / "semantic-reference-vectors.npz"
    metadata = json.loads(metadata_path.read_text())
    archive = np.load(vectors_path, allow_pickle=False)
    references = dict(zip(archive["case_ids"].tolist(), archive["vectors"]))
    bandwidth = metadata["bandwidth_squared"]
    train_ids = set(metadata["training_prompt_ids"])
    rows = read_rows(source / "training-replay.jsonl.gz")
    by_step = defaultdict(list)
    for row in rows:
        if row["case_id"] not in train_ids:
            raise ValueError("Replay contains a held-out question")
        by_step[row["step"]].append(row)
    steps = sorted(by_step)
    if not steps or steps != list(range(1, steps[-1] + 1)) or len(steps) <= calibration_steps:
        raise ValueError("Need complete consecutive steps and a later diagnostic partition")
    grouped, replay_error, credits = [], 0.0, {"rbf": [], "energy": []}
    for step in steps:
        batch = sorted(by_step[step], key=lambda row: (row["rank"], row["id"]))
        if len(batch) != cfg["global_group_size"]:
            raise ValueError("Incomplete saved global batch")
        quality = np.array([r["quality_reward"] for r in batch])
        qa, original_da, _ = conditional_credit(batch, quality, references, bandwidth)
        error = np.abs(
            qa
            + cfg["distribution_weight"] * original_da
            - np.array([r["advantage"] for r in batch])
        ).max()
        replay_error = max(replay_error, float(error))
        for name in ("rbf", "energy"):
            _, da, _ = conditional_credit(batch, quality, references, bandwidth, kernel_name=name)
            credits[name].extend(da.tolist())
        questions = defaultdict(list)
        for row in batch:
            questions[row["case_id"]].append(row)
        for key, answers in questions.items():
            if len(answers) != 4:
                raise ValueError("Expected four independent answers for each exact question")
            vectors = np.array([r["semantic_embedding"] for r in answers])
            group = {
                "step": step,
                "case_id": key,
                "quality_advantages": (
                    np.array([r["quality_reward"] for r in answers])
                    - np.mean([r["quality_reward"] for r in answers])
                )
                * 4
                / 3,
            }
            for name in ("rbf", "energy"):
                cost, da = passage_credit(vectors, references[key], bandwidth, kernel_name=name)
                group[name] = {"cost": cost, "credit": da}
            grouped.append(group)
    if replay_error > 1e-6:
        raise ValueError(f"Replayed original credits disagree with actual training: {replay_error}")
    calibration_groups = [r for r in grouped if r["step"] <= calibration_steps]
    diagnostic_groups = [r for r in grouped if r["step"] > calibration_steps]
    scales = {
        name: calibration_scale([r[name]["cost"] for r in calibration_groups])
        for name in ("rbf", "energy")
    }
    divisor = fixed_loss_divisor(1, cfg["distribution_weight"], cfg["reference_kl_beta"])
    calibration = {
        "source_job_id": "504124",
        "calibration_steps": [1, calibration_steps],
        "calibration_question_groups": len(calibration_groups),
        "target_cost_level": 1.0,
        "kernel_scales": scales,
        "objective_loss_divisor": divisor,
        "source_reference_vectors_sha256": sha(vectors_path),
        "source_semantic_metadata_sha256": sha(metadata_path),
        "source_training_replay_sha256": hashlib.sha256(
            gzip.decompress((source / "training-replay.jsonl.gz").read_bytes())
        ).hexdigest(),
        "source_run_config_sha256": sha(source / "run_config.json"),
        "encoder_model": metadata["encoder_model"],
        "encoder_revision": metadata["encoder_revision"],
        "calibration_policy": "Early training policies from saved steps, "
        "not a newly sampled fixed base policy",
        "fixed_after_calibration": True,
        "selection_uses_final_test_answers": False,
    }
    output.mkdir(parents=True, exist_ok=True)
    write_json(output / "calibration.json", calibration)
    comparison = {}
    for name in ("rbf", "energy"):
        group = diagnostic_groups
        quality_credit = np.concatenate([r["quality_advantages"] for r in group])
        distribution_credit = np.concatenate([r[name]["credit"] for r in group])
        costs = np.array([r[name]["cost"] for r in group])
        scale = scales[name]
        comparison[name] = {
            "raw_calibration_mean_discrepancy": float(
                np.mean([r[name]["cost"] for r in calibration_groups])
            ),
            "kernel_scale_frozen_from_calibration": scale,
            "effective_base_kl_to_raw_discrepancy_weight": cfg["reference_kl_beta"]
            / (cfg["distribution_weight"] * scale),
            "raw_later_mean_discrepancy": float(costs.mean()),
            "scaled_later_mean_discrepancy": float(scale * costs.mean()),
            "raw_later_between_group_discrepancy_std": float(costs.std()),
            "raw_quality_credit_rms": rms(quality_credit),
            "original_weighted_distribution_credit_rms": rms(
                cfg["distribution_weight"] * distribution_credit
            ),
            "calibrated_weighted_distribution_credit_rms": rms(
                cfg["distribution_weight"] * scale * distribution_credit
            ),
            "normalized_quality_credit_rms": rms(quality_credit / divisor),
            "normalized_distribution_credit_rms": rms(
                cfg["distribution_weight"] * scale * distribution_credit / divisor
            ),
            "normalized_distribution_to_quality_credit_rms_ratio": rms(
                cfg["distribution_weight"] * scale * distribution_credit
            )
            / rms(quality_credit),
        }
    raw_a = np.concatenate([r["rbf"]["credit"] for r in diagnostic_groups])
    raw_b = np.concatenate([r["energy"]["credit"] for r in diagnostic_groups])
    heldout = {}
    for label in ("baseline", "final"):
        values = read_rows(source / f"{label}-samples.jsonl")
        questions = defaultdict(list)
        for row in values:
            questions[row["case_id"]].append(row)
        heldout[label] = {}
        for name in ("rbf", "energy"):
            costs = [
                mmd2(
                    np.array([r["semantic_embedding"] for r in answers]),
                    references[key],
                    bandwidth,
                    kernel_name=name,
                )
                for key, answers in questions.items()
            ]
            heldout[label][name] = float(np.mean(costs))
    report = {
        "status": "CPU replay and implementation validation only; no new trained policy",
        "source_job_id": "504124",
        "training_answers": len(rows),
        "source_updates": len(steps),
        "training_question_groups": len(grouped),
        "observed_training_questions": len({r["case_id"] for r in grouped}),
        "calibration_step_range": [1, calibration_steps],
        "diagnostic_step_range": [calibration_steps + 1, steps[-1]],
        "original_logged_advantage_max_abs_replay_error": replay_error,
        "later_kernel_credit_pearson_correlation": float(np.corrcoef(raw_a, raw_b)[0, 1]),
        "kernel_comparison": comparison,
        "heldout_v1_policy_posthoc_scores_not_new_training_results": heldout,
        "kl_centering_empirical_variance_measured": False,
        "kl_centering_missing_data": "Original run saved only aggregate log-ratio estimates, "
        "not individual log ratios",
        "gpu_allocations": 0,
        "new_generations": 0,
        "limitations": [
            "Early and later replay samples come from changing policies and different questions.",
            "Credit RMS and between-group spread are not fixed-policy gradient-variance estimates.",
            "Post hoc scores of the old final actor do not predict improvement from a new kernel.",
            "Two fixed references per question and the original encoder limitations remain.",
            "Scaling calibrates objective magnitudes; it does not prove optimal reward trade-offs.",
        ],
    }
    write_json(output / "report.json", report)
    group_records = [
        {
            "step": r["step"],
            "case_id": r["case_id"],
            **{f"{name}_cost": r[name]["cost"] for name in ("rbf", "energy")},
        }
        for r in grouped
    ]
    write_json(output / "question-group-costs.json", group_records)
    if write_profiles:
        base = json.loads((ROOT / "config.json").read_text())
        for name in ("rbf", "energy"):
            profile = {
                **base,
                "experiment_profile": f"conditional-mmd-kcgm-{name}-v2",
                "semantic_kernel_name": name,
                "semantic_kernel_scale": scales[name],
                "semantic_calibration_report": str((output / "calibration.json").relative_to(REPO)),
                "semantic_calibration_sha256": sha(output / "calibration.json"),
                "reference_kl_centering": "prompt-loo",
                "objective_loss_divisor": divisor,
                "record_policy_log_probs": True,
                "gpu_submission_authorized": False,
                "wandb_group": "helpsteer-conditional-mmd-kcgm-v2",
            }
            write_json(ROOT / f"kcgm-{name}-config.json", profile)
    print(json.dumps(report, indent=2))


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--source", type=Path, required=True)
    parser.add_argument("--reference-data", type=Path, default=ROOT / "data")
    parser.add_argument("--output", type=Path, default=REPO / "results/helpsteer-v1/kcgm-cpu-audit")
    parser.add_argument("--calibration-steps", type=int, default=10)
    parser.add_argument("--write-profiles", action="store_true")
    args = parser.parse_args()
    audit(
        args.source.resolve(),
        args.reference_data.resolve(),
        args.output.resolve(),
        args.calibration_steps,
        args.write_profiles,
    )
