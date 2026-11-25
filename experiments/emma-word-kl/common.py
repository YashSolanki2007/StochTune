"""Reuse tested policy utilities with an explicitly frozen fiction rubric reward."""

import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parent
REPO = ROOT.parents[1]
_impl = ROOT / "_common_impl.py"
if not _impl.exists():
    _impl = REPO / "experiments/helpsteer-rl-v2/common.py"
exec(compile(_impl.read_text().replace("helpsteer-", "emma-"), str(_impl), "exec"), globals())
# Keep this experiment's quality/validation modules ahead of support modules.
sys.path.insert(0, str(ROOT))
sys.path.append(str(REPO / "experiments/helpsteer-rl-v2"))

from pilot import words  # noqa: E402,F401
from word_distribution import BucketedWordDistribution  # noqa: E402,F401


def config():
    import json

    return json.loads((ROOT / "run-config.json").read_text())


class FrozenPreferenceReward:
    """Interface expected by the policy trainer; this reward has no learned adapter."""

    def __init__(self, cfg, output, device, calibration_path=None):
        from fiction_reward import FrozenRubricJudge

        self.cfg = cfg
        self.judge = FrozenRubricJudge(cfg, device)

    def score(self, rows):
        import numpy as np

        ratings = self.judge.score(rows)
        normalized = np.array([r["quality"] for r in ratings], dtype=float)
        quality, penalties = globals()["quality_with_truncation"](
            normalized, [r["terminated"] for r in rows], self.cfg["truncation_penalty"]
        )
        return [
            dict(
                raw_reward=float(1 + 4 * n),
                normalized_reward=float(n),
                quality_reward=float(q),
                truncation_penalty=float(p),
                rubric_scores=r["scores"],
                rating_token_mass=r["rating_token_mass"],
            )
            for n, q, p, r in zip(normalized, quality, penalties, ratings)
        ], quality


def verify_reward_identity(ready, checks):
    import hashlib

    from fiction_reward import scorer_identity

    if ready.get("reward_kind") != "frozen-fiction-rubric":
        raise RuntimeError("Emma must use the audited frozen fiction scorer")
    if checks.get("scorer_identity_sha256") != scorer_identity(config()):
        raise RuntimeError("The audited scorer model, rubric or code changed")
    calibration = Path(ready["audit_path"]) / "calibration.json"
    if checks.get("calibration_sha256") != hashlib.sha256(calibration.read_bytes()).hexdigest():
        raise RuntimeError("The scorer certificate changed")
