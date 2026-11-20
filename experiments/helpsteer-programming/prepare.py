"""Prepare the reviewed programming corpus on CPU using only cached source files.

This script loads a tokenizer, never a model; it cannot submit a GPU job.
"""

import argparse
import gzip
import hashlib
import json
import random
import sys
from collections import Counter
from pathlib import Path

from topic_filter import CANDIDATE, opening_key, prompt_groups

ROOT = Path(__file__).resolve().parent
REPO = ROOT.parents[1]
V2 = REPO / "experiments/helpsteer-rl-v2"
sys.path.insert(0, str(V2))
from common import (  # noqa: E402
    distribution_diagnostics,
    make_distribution,
    messages,
    prompt_id,
    read_rows,
    words,
    write_json,
    write_rows,
)


def sha(path):
    return hashlib.sha256(path.read_bytes()).hexdigest()


def reviewed_candidates(reference_path, preference_path, review_path):
    references = {row["id"]: row["response"] for row in read_rows(reference_path)}
    prompts = {}
    with gzip.open(preference_path, "rt") as stream:
        for row in map(json.loads, stream):
            key = prompt_id(row["prompt"])
            if key in references and row["split"] == "train":
                prompts.setdefault(key, row["prompt"])
    candidates = {
        key: {"id": key, "prompt": prompt, "response": references[key]}
        for key, prompt in prompts.items()
        if CANDIDATE.search(prompt)
    }
    review = json.loads(review_path.read_text())
    if review["reference_sha256"] != sha(reference_path):
        raise ValueError("Reference changed: topic review must be revisited")
    if set(review["decisions"]) != set(candidates):
        raise ValueError("Every candidate needs a review, with no stale or missing IDs")
    for key, row in candidates.items():
        decision = review["decisions"][key]
        if hashlib.sha256(row["prompt"].encode()).hexdigest() != decision["prompt_sha256"]:
            raise ValueError("Reviewed prompt changed")
        row.update(topic_included=decision["include"], topic_reason=decision["reason"])
    return sorted(candidates.values(), key=lambda row: row["id"])


def split_candidates(candidates, seed, validation_count, test_count):
    groups = prompt_groups([row for row in candidates if row["topic_included"]])
    # One answer per conversation family prevents repeated continuations from
    # dominating the target. Selection uses only IDs, never response quality/length.
    selected = []
    for group in groups:
        chosen = min(
            group,
            key=lambda row: hashlib.sha256(f"{seed}:representative:{row['id']}".encode()).digest(),
        )
        group_id = hashlib.sha256(min(opening_key(r["prompt"]) for r in group).encode()).hexdigest()
        selected.append({**chosen, "group_id": group_id, "group_members": [r["id"] for r in group]})
    selected.sort(key=lambda row: row["id"])
    random.Random(seed).shuffle(selected)
    if len(selected) < validation_count + test_count + 32:
        raise ValueError("Too few distinct programming families for the declared splits")
    return {
        "validation": selected[:validation_count],
        "test": selected[validation_count : validation_count + test_count],
        "train": selected[validation_count + test_count :],
    }


def prepare(raw, source_data, out):
    cfg = json.loads((ROOT / "config.json").read_text())
    candidates = reviewed_candidates(
        source_data / "reference.jsonl", raw / "preference.jsonl.gz", ROOT / "topic-review.json"
    )
    reward_audit = read_rows(source_data / "reward_validation.jsonl")
    audit_ids = {row["id"] for row in reward_audit}
    reserved = set()
    for group in prompt_groups([r for r in candidates if r["topic_included"]] + reward_audit):
        if any(row["id"] in audit_ids for row in group):
            reserved.update(row["id"] for row in group)
    eligible = [r for r in candidates if r["id"] not in reserved]
    splits = split_candidates(
        eligible, cfg["seed"], cfg["validation_prompt_count"], cfg["test_prompt_count"]
    )
    from transformers import AutoTokenizer

    tokenizer = AutoTokenizer.from_pretrained(
        cfg["model"], revision=cfg["revision"], local_files_only=True
    )
    for rows in splits.values():
        for row in rows:
            row["prompt_ids"] = tokenizer.apply_chat_template(
                messages(row["prompt"]), tokenize=True, add_generation_prompt=True
            )
            if len(row["prompt_ids"]) > cfg["max_prompt_tokens"]:
                raise ValueError("Inherited reference exceeds actor prompt limit")
    reference_passages = [row["response"] for row in splits["train"]]
    target = make_distribution(cfg, reference_passages)
    counts = Counter(word for passage in reference_passages for word in words(passage))
    out.mkdir(parents=True, exist_ok=True)
    write_rows(out / "topic-candidates.jsonl", candidates)
    for name, rows in splits.items():
        write_rows(
            out / f"policy_{name}.jsonl",
            [{key: r[key] for key in ("id", "prompt", "prompt_ids", "group_id")} for r in rows],
        )
        write_rows(
            out / ("reference.jsonl" if name == "train" else f"heldout_{name}_reference.jsonl"),
            [{key: r[key] for key in ("id", "response", "group_id")} for r in rows],
        )
    # The existing scorer certificate remains bound to exactly its original
    # general-domain validation bytes. This is NOT a new domain-specific audit.
    (out / "reward_validation.jsonl").write_bytes(
        (source_data / "reward_validation.jsonl").read_bytes()
    )
    policy_ids = {r["id"] for rows in splits.values() for r in rows}
    reward_ids = {r["id"] for r in read_rows(out / "reward_validation.jsonl")}
    if policy_ids & reward_ids:
        raise ValueError("Policy prompts overlap the frozen scorer audit")
    vocabulary = {
        "normalization": "Lowercase [A-Za-z]+(?:['’][A-Za-z]+)*; normalize curly apostrophes.",
        "source": "Only preferred answers in policy_train; one per conversation family.",
        "max_words": cfg["vocabulary_max_words"],
        "min_count": cfg["vocabulary_min_count"],
        "reference_word_count": sum(counts.values()),
        "reference_word_types": len(counts),
        "other_index": target.unknown,
        "categories": [
            {"word": word, "count": counts[word], "probability": float(target.probabilities[i])}
            for word, i in target.index.items()
        ]
        + [
            {
                "word": None,
                "label": "OTHER",
                "count": sum(c for w, c in counts.items() if w not in target.index),
                "probability": float(target.probabilities[-1]),
            }
        ],
    }
    write_json(out / "vocabulary.json", vocabulary)
    threshold_sensitivity = []
    for minimum in (1, 2, 5, 10):
        alternative = make_distribution(
            {**cfg, "vocabulary_min_count": minimum}, reference_passages
        )
        threshold_sensitivity.append(
            {
                "min_count": minimum,
                "retained_words": alternative.unknown,
                "reference_other_mass": float(alternative.probabilities[-1]),
            }
        )
    report = {
        "status": "CPU data preparation only; no policy or reward model trained/evaluated",
        "candidate_count": len(candidates),
        "included_before_family_deduplication": sum(r["topic_included"] for r in candidates),
        "topic_included_reserved_for_reward_audit": sum(
            r["topic_included"] and r["id"] in reserved for r in candidates
        ),
        "excluded_reasons": dict(
            Counter(r["topic_reason"] for r in candidates if not r["topic_included"])
        ),
        "counts": {name: len(rows) for name, rows in splits.items()},
        "training_words": sum(counts.values()),
        "training_word_types": len(counts),
        "retained_words": target.unknown,
        "reference_other_mass": float(target.probabilities[-1]),
        "threshold_sensitivity": threshold_sensitivity,
        "heldout_gold_diagnostics_not_model_results": {
            name: distribution_diagnostics(target, [r["response"] for r in splits[name]])
            for name in ("validation", "test")
        },
        "limitations": [
            "Assistant topic review; no independent human domain annotation. "
            "Candidate search may miss relevant prompts.",
            "Word normalization excludes operators and numerals and splits code identifiers; "
            "this is not a code-token PMF.",
            "The capped KL allows redistribution within OTHER; "
            "full-vocabulary KL remains a separate diagnostic.",
            "Heldout data are newly split original-training examples; "
            "upstream scorer overlap is unknown.",
            "The existing Skywork general-domain certificate is unchanged; "
            "no programming-specific scorer audit performed.",
            "A narrower PMF does not fix finite-batch estimator bias "
            "or guarantee better generation quality.",
        ],
    }
    write_json(out / "cpu-report.json", report)
    manifest = {
        "dataset": cfg["dataset"],
        "revision": cfg["dataset_revision"],
        "source_reference_sha256": sha(source_data / "reference.jsonl"),
        "source_preferences_sha256": sha(raw / "preference.jsonl.gz"),
        "topic_review_sha256": sha(ROOT / "topic-review.json"),
        "configuration_sha256": sha(ROOT / "config.json"),
        "counts": report["counts"],
        "split_prompt_ids": {name: sorted(r["id"] for r in rows) for name, rows in splits.items()},
        "conversation_families": {
            name: {r["group_id"]: r["group_members"] for r in rows} for name, rows in splits.items()
        },
        "reference_prompt_ids": sorted(r["id"] for r in splits["train"]),
        "reserved_conversation_member_ids": sorted(reserved),
        "target_scope": "Only policy-training answers; "
        "heldout answers never fit the vocabulary or PMF.",
        "reward_validation_role": "Unchanged general-domain certificate for the frozen scorer; "
        "not policy evaluation.",
        "prepared_sha256": {
            p.name: sha(p)
            for p in sorted(out.iterdir())
            if p.is_file() and p.name != "manifest.json"
        },
    }
    write_json(out / "manifest.json", manifest)
    print(
        json.dumps(
            {
                key: report[key]
                for key in (
                    "candidate_count",
                    "included_before_family_deduplication",
                    "counts",
                    "training_words",
                    "training_word_types",
                    "retained_words",
                    "reference_other_mass",
                )
            },
            indent=2,
        )
    )


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--raw", type=Path, default=REPO / "outputs/helpsteer-data/raw")
    parser.add_argument("--source-data", type=Path, default=V2 / "data")
    parser.add_argument("--out", type=Path, default=ROOT / "data")
    args = parser.parse_args()
    prepare(args.raw.resolve(), args.source_data.resolve(), args.out.resolve())
