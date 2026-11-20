"""CPU reward-validation metrics and explicit policy eligibility checks."""

import numpy as np


def correctness(direction, target):
    return np.where(direction == 0, 0.5, (direction == target).astype(float))


def interval(values, seed=42):
    values = np.asarray(values, dtype=float)
    if not len(values):
        return None
    rng = np.random.default_rng(seed)
    estimates = rng.choice(values, size=(2000, len(values)), replace=True).mean(1)
    return np.quantile(estimates, [0.025, 0.975]).tolist()


def baselines(rows, majority_direction):
    strength = np.array([r["preference_strength"] for r in rows])
    selected = strength != 0
    target = np.sign(strength[selected])
    length1 = np.array([r["response_tokens_1"] for r in rows])[selected]
    length2 = np.array([r["response_tokens_2"] for r in rows])[selected]
    longer = correctness(np.sign(length2 - length1), target)
    majority = correctness(np.full(len(target), majority_direction), target)
    return {
        "non_tie_pairs": int(selected.sum()),
        "longer_answer_accuracy": float(longer.mean()) if len(longer) else None,
        "training_majority_direction": majority_direction,
        "majority_direction_accuracy": float(majority.mean()) if len(majority) else None,
    }


def validate_reward(rows, scores_one, scores_two, majority_direction):
    one, two = np.asarray(scores_one), np.asarray(scores_two)
    if len(one) != len(rows) or len(two) != len(rows) or not np.isfinite(one + two).all():
        raise ValueError("Need one finite score per candidate answer")
    strength = np.array([r["preference_strength"] for r in rows])
    mask = strength != 0
    direction = np.sign(two - one)
    target = np.sign(strength[mask])
    model = correctness(direction[mask], target)
    n1 = np.array([r["response_tokens_1"] for r in rows])
    n2 = np.array([r["response_tokens_2"] for r in rows])
    longer = correctness(np.sign(n2[mask] - n1[mask]), target)
    majority = correctness(np.full(len(target), majority_direction), target)
    report = {
        "protocol_version": 2,
        "baselines": baselines(rows, majority_direction),
        "non_tie_accuracy": float(model.mean()) if len(model) else None,
        "accuracy_95pct": interval(model),
        "model_minus_longer_95pct": interval(model - longer),
        "model_minus_majority_95pct": interval(model - majority),
    }
    slices = {
        "length_matched": mask & (np.minimum(n1, n2) / np.maximum(n1, n2) >= 0.8),
        "long_inputs": mask
        & np.array([max(len(r["ids_1"]), len(r["ids_2"])) > 1024 for r in rows]),
    }
    for name, selected in slices.items():
        result = correctness(direction[selected], np.sign(strength[selected]))
        report[name] = {
            "pairs": int(selected.sum()),
            "accuracy": float(result.mean()) if len(result) else None,
            "accuracy_95pct": interval(result),
        }
    c1 = np.array([r["coherence_1"] for r in rows])
    c2 = np.array([r["coherence_2"] for r in rows])
    selected = c1 != c2
    result = correctness(direction[selected], np.sign(c2[selected] - c1[selected]))
    report["human_coherence"] = {
        "pairs": int(selected.sum()),
        "accuracy": float(result.mean()) if len(result) else None,
        "accuracy_95pct": interval(result),
    }

    def lower(value, threshold):
        return value is not None and value[0] > threshold

    checks = {
        "at_least_100_non_tied_pairs": len(model) >= 100,
        "beats_longer_answer_baseline": lower(report["model_minus_longer_95pct"], 0),
        "beats_training_majority_baseline": lower(report["model_minus_majority_95pct"], 0),
        "length_matched_evidence": report["length_matched"]["pairs"] >= 30
        and lower(report["length_matched"]["accuracy_95pct"], 0.5),
        "coherence_evidence": report["human_coherence"]["pairs"] >= 30
        and lower(report["human_coherence"]["accuracy_95pct"], 0.5),
        "long_input_evidence": report["long_inputs"]["pairs"] >= 20
        and lower(report["long_inputs"]["accuracy_95pct"], 0.5),
    }
    report.update(
        checks=checks,
        passed=all(checks.values()),
        caveat="Exploratory paired bootstrap over held-out prompts; selection on this "
        "validation set prevents treating intervals as independent test evidence.",
    )
    return report


def require_reward_audit(checks, expected_data_hash):
    if checks.get("protocol_version") != 2 or not checks.get("passed"):
        raise RuntimeError("Reward has not passed v2 quality/length validation")
    if checks.get("validation_data_sha256") != expected_data_hash:
        raise RuntimeError("Reward audit uses different validation data")
