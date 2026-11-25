"""Literary preference checks; never invent coherence labels absent from LitBench."""

import numpy as np


def interval(values, seed=42):
    values = np.asarray(values, dtype=float)
    rng = np.random.default_rng(seed)
    return np.quantile(
        rng.choice(values, size=(2000, len(values)), replace=True).mean(1), [0.025, 0.975]
    ).tolist()


def validate_reward(rows, one, two):
    one, two = np.asarray(one), np.asarray(two)
    if len(rows) != len(one) or len(rows) != len(two) or not np.isfinite(one + two).all():
        raise ValueError("Need finite scores for every held-out preference pair")
    target = np.sign([r["preference_strength"] for r in rows])
    direction = np.sign(two - one)
    correct = np.where(direction == 0, 0.5, direction == target).astype(float)
    n1 = np.array([r["response_tokens_1"] for r in rows])
    n2 = np.array([r["response_tokens_2"] for r in rows])
    length_direction = np.sign(n2 - n1)
    longer = np.where(length_direction == 0, 0.5, length_direction == target).astype(float)
    matched = np.minimum(n1, n2) / np.maximum(n1, n2) >= 0.8
    ci = interval(correct)
    difference = interval(correct - longer)
    report = {
        "protocol_version": 3,
        "pairs": len(rows),
        "non_tie_accuracy": float(correct.mean()),
        "accuracy_95pct": ci,
        "longer_answer_accuracy": float(longer.mean()),
        "model_minus_longer_95pct": difference,
        "length_matched_pairs": int(matched.sum()),
        "length_matched_accuracy": float(correct[matched].mean()) if matched.any() else None,
        "labels": (
            "Implicit Reddit-vote preferences curated by LitBench; not expert quality ratings"
        ),
    }
    checks = {
        "at_least_128_pairs": len(rows) >= 128,
        "above_chance": ci[0] > 0.5,
        "accuracy_at_least_55pct": correct.mean() >= 0.55,
        "beats_length_baseline": difference[0] > 0,
        "length_matched_evidence": matched.sum() >= 50 and correct[matched].mean() >= 0.55,
    }
    # NumPy comparison scalars must become native bools before JSON/W&B logging.
    checks = {name: bool(value) for name, value in checks.items()}
    report.update(checks=checks, passed=all(checks.values()))
    return report


def require_reward_audit(checks, expected_data_hash):
    if checks.get("protocol_version") != 3 or not checks.get("passed"):
        raise RuntimeError(
            "Literary reward failed its held-out preference/length checks; stop before RL"
        )
    if checks.get("validation_data_sha256") != expected_data_hash:
        raise RuntimeError("Literary reward certificate is bound to different held-out pairs")
