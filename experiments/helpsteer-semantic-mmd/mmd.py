"""Conditional MMD against fixed empirical references, with passage-level credit."""

from collections import defaultdict

import numpy as np


def kernel(x, y, bandwidth_squared, scales=(0.5, 1.0, 2.0), *, name="rbf", scale=1.0):
    x, y = np.asarray(x, dtype=np.float64), np.asarray(y, dtype=np.float64)
    if x.ndim != 2 or y.ndim != 2 or x.shape[1] != y.shape[1]:
        raise ValueError("Expected compatible matrices of answer embeddings")
    if not np.all(np.isfinite(x)) or not np.all(np.isfinite(y)):
        raise ValueError("Nonfinite embeddings")
    if not np.isfinite(scale) or scale <= 0:
        raise ValueError("Kernel scale must be finite and positive")
    if name == "energy":
        from scipy.spatial.distance import cdist

        # This distance-induced kernel gives the energy-distance discrepancy.
        # It requires no RBF bandwidth; identical vectors have exact distance zero.
        return -scale * cdist(x, y, metric="euclidean")
    if name != "rbf":
        raise ValueError(f"Unknown semantic kernel: {name}")
    if bandwidth_squared <= 0 or any(s <= 0 for s in scales):
        raise ValueError("Invalid Gaussian bandwidth")
    squared = np.maximum((x * x).sum(1)[:, None] + (y * y).sum(1)[None, :] - 2 * x @ y.T, 0)
    return scale * np.mean(
        [np.exp(-squared / (2 * bandwidth_squared * s * s)) for s in scales], axis=0
    )


def mmd2(
    generated, reference, bandwidth_squared, unbiased=True, *, kernel_name="rbf", kernel_scale=1.0
):
    """Generated U-statistic; reference V-statistic defines the fixed empirical target.

    The U version is unbiased for population MMD² to this *fixed empirical*
    reference measure when generated answers are independent conditional on q.
    A finite estimate can be negative and must not be clipped.
    """
    generated, reference = np.asarray(generated), np.asarray(reference)
    n = len(generated)
    if n < 2 or len(reference) < 2:
        raise ValueError("Need at least two generated and two reference answers")
    kwargs = {"name": kernel_name, "scale": kernel_scale}
    gg = kernel(generated, generated, bandwidth_squared, **kwargs)
    own = (gg.sum() - np.trace(gg)) / (n * (n - 1)) if unbiased else gg.mean()
    return float(
        own
        + kernel(reference, reference, bandwidth_squared, **kwargs).mean()
        - 2 * kernel(generated, reference, bandwidth_squared, **kwargs).mean()
    )


def passage_credit(generated, reference, bandwidth_squared, *, kernel_name="rbf", kernel_scale=1.0):
    generated = np.asarray(generated)
    n = len(generated)
    if n < 3:
        raise ValueError("Leave-one-out MMD requires at least three answers per question")
    kwargs = {"kernel_name": kernel_name, "kernel_scale": kernel_scale}
    cost = mmd2(generated, reference, bandwidth_squared, **kwargs)
    advantages = n * np.array(
        [
            mmd2(np.delete(generated, i, axis=0), reference, bandwidth_squared, **kwargs) - cost
            for i in range(n)
        ]
    )
    return cost, advantages


def conditional_credit(
    rows, quality, references, bandwidth_squared, *, kernel_name="rbf", kernel_scale=1.0
):
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
        cost, advantages = passage_credit(
            generated,
            references[case_id],
            bandwidth_squared,
            kernel_name=kernel_name,
            kernel_scale=kernel_scale,
        )
        q = quality[indices]
        qa[indices] = q - (q.sum() - q) / (len(q) - 1)
        da[indices] = advantages
        costs.append(cost)
    return (
        qa,
        da,
        {
            "conditional_mmd2_u": float(np.mean(costs)),
            "conditional_discrepancy_u_unscaled": float(np.mean(costs)) / kernel_scale,
            "mmd_questions": len(groups),
            "answers_per_question": len(next(iter(groups.values()))),
        },
    )


def per_question_metrics(
    rows, references, bandwidth_squared, *, kernel_name="rbf", kernel_scale=1.0
):
    groups = defaultdict(list)
    for row in rows:
        groups[row["case_id"]].append(row["semantic_embedding"])
    result = {}
    for key, values in groups.items():
        generated, reference = np.array(values), references[key]
        # This reports coverage of the two reference prototypes, not true semantic modes.
        nearest = (generated @ reference.T).argmax(1)
        result[key] = {
            "conditional_mmd2_u": mmd2(
                generated,
                reference,
                bandwidth_squared,
                kernel_name=kernel_name,
                kernel_scale=kernel_scale,
            ),
            "conditional_mmd2_v": mmd2(
                generated,
                reference,
                bandwidth_squared,
                unbiased=False,
                kernel_name=kernel_name,
                kernel_scale=kernel_scale,
            ),
            "conditional_rbf_u_unscaled": mmd2(generated, reference, bandwidth_squared),
            "conditional_energy_u_unscaled": mmd2(
                generated, reference, bandwidth_squared, kernel_name="energy"
            ),
            "reference_prototype_coverage": len(set(nearest.tolist())) / len(reference),
            "reference_prototype_one_fraction": float(np.mean(nearest == 0)),
            "generated_answers": len(generated),
        }
    return result
