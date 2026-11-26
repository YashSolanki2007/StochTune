"""Fixed word-range penalty, separate from frozen judge scores and histogram KL."""

import hashlib
import json
import math
from pathlib import Path

import numpy as np
from common import words


def settings(cfg):
    values = {
        "weight": cfg.get("length_penalty_weight", 0.0),
        "min_words": cfg.get("length_min_words", 250),
        "max_words": cfg.get("length_max_words", 350),
        "unit_words": cfg.get("length_penalty_unit_words", 100),
    }
    weight = values["weight"]
    if isinstance(weight, bool) or not isinstance(weight, (int, float)):
        raise ValueError("Length weight must be a finite nonnegative number")
    if not math.isfinite(weight) or weight < 0:
        raise ValueError("Length weight must be a finite nonnegative number")
    for key in ("min_words", "max_words", "unit_words"):
        if isinstance(values[key], bool) or not isinstance(values[key], int):
            raise ValueError("Length boundaries and unit must be integers")
    if not 0 <= values["min_words"] < values["max_words"] or values["unit_words"] <= 0:
        raise ValueError("Invalid fixed word range or penalty unit")
    return values


def apply_penalty(rows, quality, cfg):
    """Apply after the EOS penalty and before the prompt-LOO quality baseline.

    Never clip negative adjusted rewards or truncate generated text/counts.
    All lengths use the same word tokenizer as the histogram diagnostics.
    """
    spec = settings(cfg)
    quality = np.asarray(quality, dtype=np.float64)
    if quality.ndim != 1 or len(rows) != len(quality) or not np.isfinite(quality).all():
        raise ValueError("Need one finite pre-length quality reward per answer")
    diagnostics = []
    penalties = []
    for row, before in zip(rows, quality):
        length = len(words(row["passage"]))
        under = max(0, spec["min_words"] - length)
        over = max(0, length - spec["max_words"])
        units = (under + over) / spec["unit_words"]
        penalty = spec["weight"] * units
        penalties.append(penalty)
        diagnostics.append(
            {
                "quality_reward_before_length_penalty": float(before),
                "length_penalty": float(penalty),
                "length_penalty_units": float(units),
                "generated_word_count": length,
                "length_underflow_words": under,
                "length_overflow_words": over,
                "length_in_range": bool(under == over == 0),
            }
        )
    return quality - np.asarray(penalties), diagnostics


def transform_identity(cfg):
    """Bind the certificate to the exact deterministic reward transform."""
    value = {
        "length": settings(cfg),
        "truncation_penalty": cfg["truncation_penalty"],
        "implementation_sha256": hashlib.sha256(Path(__file__).read_bytes()).hexdigest(),
        "order": "judge normalization, EOS penalty, word-range penalty, LOO baseline",
    }
    return hashlib.sha256(json.dumps(value, sort_keys=True).encode()).hexdigest()
