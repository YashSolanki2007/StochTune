"""Conditional MMD against fixed empirical references, with passage-level credit."""

from collections import defaultdict

import numpy as np


def kernel(x, y, bandwidth_squared, scales=(0.5, 1.0, 2.0)):
    x, y = np.asarray(x, dtype=np.float64), np.asarray(y, dtype=np.float64)
    if x.ndim != 2 or y.ndim != 2 or x.shape[1] != y.shape[1]:
        raise ValueError("Expected compatible matrices of answer embeddings")
    if not np.all(np.isfinite(x)) or not np.all(np.isfinite(y)) or bandwidth_squared <= 0:
        raise ValueError("Nonfinite embeddings or invalid bandwidth")
    squared = np.maximum((x * x).sum(1)[:, None] + (y * y).sum(1)[None, :] - 2 * x @ y.T, 0)
    return np.mean([np.exp(-squared / (2 * bandwidth_squared * s * s)) for s in scales], axis=0)


def mmd2(generated, reference, bandwidth_squared, unbiased=True):
    """Generated U-statistic; reference V-statistic defines the fixed empirical target.

    The U version is unbiased for population MMD² to this *fixed empirical*
    reference measure when generated answers are independent conditional on q.
    A finite estimate can be negative and must not be clipped.
    """
    generated, reference = np.asarray(generated), np.asarray(reference)
    n = len(generated)
    if n < 2 or len(reference) < 2:
        raise ValueError("Need at least two generated and two reference answers")
    gg = kernel(generated, generated, bandwidth_squared)
    own = (gg.sum() - np.trace(gg)) / (n * (n - 1)) if unbiased else gg.mean()
    return float(
        own
        + kernel(reference, reference, bandwidth_squared).mean()
        - 2 * kernel(generated, reference, bandwidth_squared).mean()
    )


def passage_credit(generated, reference, bandwidth_squared):
    generated = np.asarray(generated)
    n = len(generated)
    if n < 3:
        raise ValueError("Leave-one-out MMD requires at least three answers per question")
    cost = mmd2(generated, reference, bandwidth_squared)
    advantages = n * np.array(
        [
            mmd2(np.delete(generated, i, axis=0), reference, bandwidth_squared) - cost
            for i in range(n)
        ]
    )
    return cost, advantages


def conditional_credit(rows, quality, references, bandwidth_squared):
    """Keep every comparison and quality baseline inside its exact question."""
    quality = np.asarray(quality, dtype=np.float64)
    if len(rows) != len(quality) or not np.all(np.isfinite(quality)):
        raise ValueError("Expected one finite quality score per answer")
    groups = defaultdict(list)
    for i, row in enumerate(rows):
        groups[row["case_id"]].append(i)
    if len({len(indices) for indices in groups.values()}) != 1:
        raise ValueError("All training questions must have the same number of answers")
    qa, da, costs = np.zeros(len(rows)), np.zeros(len(rows)), []
    for case_id, indices in groups.items():
        generated = np.array([rows[i]["semantic_embedding"] for i in indices])
        cost, advantages = passage_credit(generated, references[case_id], bandwidth_squared)
        q = quality[indices]
        qa[indices] = q - (q.sum() - q) / (len(q) - 1)
        da[indices] = advantages
        costs.append(cost)
    return (
        qa,
        da,
        {
            "conditional_mmd2_u": float(np.mean(costs)),
            "mmd_questions": len(groups),
            "answers_per_question": len(next(iter(groups.values()))),
        },
    )


def per_question_metrics(rows, references, bandwidth_squared):
    groups = defaultdict(list)
    for row in rows:
        groups[row["case_id"]].append(row["semantic_embedding"])
    result = {}
    for key, values in groups.items():
        generated, reference = np.array(values), references[key]
        # This reports coverage of the two reference prototypes, not true semantic modes.
        nearest = (generated @ reference.T).argmax(1)
        result[key] = {
            "conditional_mmd2_u": mmd2(generated, reference, bandwidth_squared),
            "conditional_mmd2_v": mmd2(generated, reference, bandwidth_squared, unbiased=False),
            "reference_prototype_coverage": len(set(nearest.tolist())) / len(reference),
            "reference_prototype_one_fraction": float(np.mean(nearest == 0)),
            "generated_answers": len(generated),
        }
    return result
