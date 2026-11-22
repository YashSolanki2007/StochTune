"""Fixed-question distribution-only diagnostic; selection never uses model scores."""

import random

import numpy as np


def select_questions(references, seed=42):
    eligible = []
    for row in references:
        lengths = [len(text.split()) for text in row["responses"]]
        if (
            row["split"] == "train"
            and min(lengths) >= 80
            and max(lengths) <= 350
            and max(lengths) / min(lengths) <= 2
            and 5 <= len(row["prompt"].split()) <= 100
            and "<extra_id_1>" not in row["prompt"]
        ):
            eligible.append(row)
    seen = random.Random(seed).sample(sorted(eligible, key=lambda r: r["case_id"]), 4)
    unseen = random.Random(seed + 1).sample(
        sorted([r for r in references if r["split"] == "validation"], key=lambda r: r["case_id"]), 4
    )
    return {
        "selection_seed": seed,
        "eligible_training_questions": len(eligible),
        "seen_ids": [r["case_id"] for r in seen],
        "unseen_ids": [r["case_id"] for r in unseen],
        "selection_uses_model_scores": False,
        "criteria": "Train only; single-turn prompt of 5–100 whitespace words; both references "
        "80–350 words; reference length ratio <=2; sorted IDs then seeded selection.",
    }


def select_cases(all_cases, ids):
    lookup = {r["id"]: r for r in all_cases}
    if len(ids) != len(set(ids)) or not set(ids) <= set(lookup):
        raise ValueError("Diagnostic question selection is duplicated or outside its split")
    return [lookup[key] for key in ids]


def zero_quality(rows):
    """No reward scorer participates in credit: every quality coefficient is exactly zero."""
    return np.zeros(len(rows), dtype=np.float64)


def comparison(baseline, final):
    result = {}
    for scope in ("seen", "unseen"):
        before, after = baseline[scope], final[scope]
        result[scope] = {
            "baseline": before,
            "final": after,
            "energy_change": after["conditional_energy_u_unscaled"]
            - before["conditional_energy_u_unscaled"],
            "rbf_change": after["conditional_rbf_u_unscaled"]
            - before["conditional_rbf_u_unscaled"],
        }
    result["interpretation"] = (
        "Seen-question fitting is the primary diagnostic; unseen questions are secondary. "
        "One seed, four questions per scope, and two fixed references per question. "
        "Small changes or failure in this budget do not establish that fitting is impossible. "
        "No text-quality reward or quality evaluation is provided by this diagnostic."
    )
    return result
