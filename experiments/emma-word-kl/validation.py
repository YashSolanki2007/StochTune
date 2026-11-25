"""Synthetic corruption sanity checks, not a human-preference benchmark."""

import math
import statistics


def validate_scores(rows, scores):
    lookup = {r["id"]: s for r, s in zip(rows, scores)}
    anchors = [r for r in rows if r["kind"] == "source"]
    if len(rows) != len(scores) or len(lookup) != len(rows):
        raise ValueError("Need one score per unique calibration row")
    comparisons = []
    for anchor in anchors:
        for kind, dimension in (
            ("shuffled", "coherence"),
            ("repetition", "coherence"),
            ("shuffled", "readability"),
        ):
            good = float(lookup[anchor["id"]]["scores"][dimension])
            bad = float(lookup[anchor["case_id"] + "-" + kind]["scores"][dimension])
            comparisons.append(
                {
                    "case_id": anchor["case_id"],
                    "control": kind,
                    "dimension": dimension,
                    "source": good,
                    "corruption": bad,
                    "source_higher": bool(good > bad),
                }
            )
    counts = {
        kind + "_" + dimension: sum(
            r["source_higher"]
            for r in comparisons
            if r["control"] == kind and r["dimension"] == dimension
        )
        for kind, dimension in (
            ("shuffled", "coherence"),
            ("repetition", "coherence"),
            ("shuffled", "readability"),
        )
    }
    margins = [
        float(lookup[r["id"]]["quality"]) - float(lookup[r["case_id"] + "-repetition"]["quality"])
        for r in anchors
    ]
    margin = statistics.mean(margins) if margins else float("nan")
    checks = {
        "eight_source_passages": len(anchors) == 8,
        "finite_scores": all(math.isfinite(float(s["quality"])) for s in scores),
        "source_beats_shuffle_coherence": counts["shuffled_coherence"] >= 7,
        "source_beats_repetition_coherence": counts["repetition_coherence"] >= 7,
        "source_beats_shuffle_readability": counts["shuffled_readability"] >= 7,
        "mean_quality_margin_at_least_point_one": math.isfinite(margin) and margin >= 0.1,
    }
    return {
        "protocol_version": 4,
        "passed": bool(all(checks.values())),
        "checks": {k: bool(v) for k, v in checks.items()},
        "wins_out_of_eight": counts,
        "mean_quality_margin_over_repetition": float(margin) if math.isfinite(margin) else None,
        "comparisons": comparisons,
        "human_validated": False,
        "limitation": (
            "Passing obvious-corruption checks does not establish nuanced quality accuracy"
        ),
    }


def require_reward_audit(checks, expected_data_hash):
    if checks.get("protocol_version") != 4 or not checks.get("passed"):
        raise RuntimeError("Frozen fiction judge failed calibration; no RL may start")
    if checks.get("validation_data_sha256") != expected_data_hash:
        raise RuntimeError("Judge certificate refers to different calibration data")
