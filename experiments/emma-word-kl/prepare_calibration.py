"""Choose training-only source passages and word-preserving/shallow corruptions."""

import hashlib
import json
import random

from common import ROOT, read_rows, words


def prepare():
    cases = read_rows(ROOT / "data/policy_train.jsonl")
    selected = {}
    for r in sorted(cases, key=lambda r: hashlib.sha256(r["id"].encode()).hexdigest()):
        selected.setdefault(r["group_id"], r)
    anchors = list(selected.values())[:8]
    if len(anchors) != 8:
        raise ValueError("Need eight distinct training chapters for calibration")
    rows = []
    rng = random.Random(42)
    for case in anchors:
        tokens = words(case["response"])
        shuffled = list(tokens)
        rng.shuffle(shuffled)
        repeated = (tokens[:20] * (len(tokens) // 20 + 1))[: len(tokens)]
        for kind, passage in (
            ("source", case["response"]),
            ("shuffled", " ".join(shuffled)),
            ("repetition", " ".join(repeated)),
        ):
            rows.append(
                {
                    "id": case["id"] + "-" + kind,
                    "case_id": case["id"],
                    "group_id": case["group_id"],
                    "prompt": case["prompt"],
                    "passage": passage,
                    "terminated": True,
                    "kind": kind,
                }
            )
    path = ROOT / "data/reward_validation.jsonl"
    path.write_text("".join(json.dumps(r, ensure_ascii=False) + "\n" for r in rows))
    print(
        json.dumps(
            {
                "calibration_rows": len(rows),
                "source_chapters": len(anchors),
                "human_preference_labels": False,
            }
        )
    )


if __name__ == "__main__":
    prepare()
