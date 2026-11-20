"""Prepare prompt-matched human references, encoder caches, and CPU MMD checks."""

import argparse
import gzip
import hashlib
import json
import shutil
import sys
from pathlib import Path

import numpy as np
from embedding import AnswerEncoder
from mmd import mmd2

ROOT = Path(__file__).resolve().parent
REPO = ROOT.parents[1]
sys.path.insert(0, str(REPO / "experiments/helpsteer-rl-v2"))
from common import prompt_id, read_rows, write_json, write_rows  # noqa: E402


def sha(path):
    return hashlib.sha256(path.read_bytes()).hexdigest()


def prepare(raw, source, out):
    import torch

    torch.set_num_threads(4)
    cfg = json.loads((ROOT / "config.json").read_text())
    upstream = json.loads((REPO / "experiments/helpsteer-rl-v2/data/manifest.json").read_text())
    if sha(raw) != upstream["raw_sha256"]["train.jsonl.gz"]:
        raise ValueError("Human-answer source differs from the pinned HelpSteer revision")
    source_manifest = json.loads((source / "manifest.json").read_text())
    for name, expected in source_manifest["prepared_sha256"].items():
        if sha(source / name) != expected:
            raise ValueError(f"Programming source changed: {name}")
    acceptable = {}
    with gzip.open(raw, "rt") as stream:
        for row in map(json.loads, stream):
            if min(row[k] for k in ("coherence", "helpfulness", "correctness")) < 3:
                continue
            key = prompt_id(row["prompt"])
            acceptable.setdefault(key, {})[row["response"]] = {
                k: row[k] for k in ("coherence", "helpfulness", "correctness")
            }
    splits, references = {}, []
    out.mkdir(parents=True, exist_ok=True)
    for name in ("train", "validation", "test"):
        original = read_rows(source / f"policy_{name}.jsonl")
        splits[name] = [row for row in original if len(acceptable.get(row["id"], {})) == 2]
        if len(splits[name]) < (32 if name == "train" else 8):
            raise ValueError(f"Too few two-reference questions in {name}")
        write_rows(out / f"policy_{name}.jsonl", splits[name])
        for row in splits[name]:
            responses = sorted(acceptable[row["id"]])
            references.append(
                {
                    "case_id": row["id"],
                    "prompt": row["prompt"],
                    "split": name,
                    "responses": responses,
                    "human_attributes": [acceptable[row["id"]][text] for text in responses],
                }
            )
    write_rows(out / "semantic-references.jsonl", references)
    encoder = AnswerEncoder(cfg, "cpu")
    print("Encoding complete human reference answers on CPU...", flush=True)
    flat = encoder.encode([text for row in references for text in row["responses"]])
    vectors = flat.reshape(len(references), 2, -1)
    case_ids = np.array([row["case_id"] for row in references])
    np.savez_compressed(out / "semantic-reference-vectors.npz", case_ids=case_ids, vectors=vectors)
    train_mask = np.array([row["split"] == "train" for row in references])
    distances = ((vectors[train_mask, 0] - vectors[train_mask, 1]) ** 2).sum(1)
    positive = distances[distances > 1e-6]
    if len(positive) < 0.9 * sum(train_mask):
        raise ValueError("Encoder cannot distinguish enough training reference pairs")
    bandwidth = float(np.median(positive))
    audit_rows = []
    training_vectors = vectors[train_mask]
    for i, reference in enumerate(training_vectors):
        matched = reference[[0, 0, 1, 1]]
        collapsed = reference[[0, 0, 0, 0]]
        wrong = training_vectors[(i + 1) % len(training_vectors)][[0, 0, 0, 0]]
        audit_rows.append(
            {
                "matched": mmd2(matched, reference, bandwidth),
                "collapsed": mmd2(collapsed, reference, bandwidth),
                "wrong_question": mmd2(wrong, reference, bandwidth),
            }
        )
    collapse_rate = float(np.mean([r["matched"] < r["collapsed"] for r in audit_rows]))
    wrong_rate = float(np.mean([r["matched"] < r["wrong_question"] for r in audit_rows]))
    if min(collapse_rate, wrong_rate) < 0.95:
        raise ValueError("CPU representation/kernel checks failed")
    metadata = {
        "encoder_model": cfg["semantic_encoder_model"],
        "encoder_revision": cfg["semantic_encoder_revision"],
        "pooling": "Attention-masked mean, unit norm per 256-token chunk; content-length weighted "
        "mean of all nonoverlapping chunk vectors, then unit norm. No text tail is dropped.",
        "bandwidth_squared": bandwidth,
        "bandwidth_fit": "Median nonzero within-question squared distance "
        "of training references only",
        "kernel_bandwidth_scales": [0.5, 1.0, 2.0],
        "training_prompt_ids": sorted(r["id"] for r in splits["train"]),
        "target": "Fixed equal-weight empirical distribution over "
        "two acceptable references per question",
        "estimator": "Generated U-statistic plus fixed-reference V-statistic; "
        "do not clip negative values",
        "conditionality": "Exact canonical conversation, not topic or shared opening alone",
    }
    write_json(out / "semantic-metadata.json", metadata)
    # Preserve the previous programming word PMF strictly as an evaluation diagnostic.
    for name in ("reference.jsonl", "vocabulary.json", "reward_validation.jsonl"):
        shutil.copy2(source / name, out / name)
    report = {
        "counts": {name: len(rows) for name, rows in splits.items()},
        "references_per_question": 2,
        "reference_answers": len(references) * 2,
        "collapsed_reference_detected_fraction": collapse_rate,
        "wrong_question_detected_fraction": wrong_rate,
        "bandwidth_squared": bandwidth,
        "median_training_reference_cosine": float(
            np.median((training_vectors[:, 0] * training_vectors[:, 1]).sum(1))
        ),
        "encoder_model": cfg["semantic_encoder_model"],
        "encoder_revision": cfg["semantic_encoder_revision"],
        "new_teacher_generation": 0,
        "gpu_allocations": 0,
        "limitations": [
            "Two model-written, human-rated answers are prototypes, "
            "not a well-estimated human answer distribution.",
            "The encoder tests are controlled corruption checks, "
            "not independent validation of semantic correctness.",
            "Equal prototype weights are an experimental choice, "
            "not measured human mode frequencies.",
            "Chunk-pooled MiniLM may overlook code correctness and fine semantic distinctions.",
            "MMD measures the feature distribution; word KL need not improve.",
        ],
    }
    write_json(out / "cpu-audit.json", report)
    write_json(
        out / "manifest.json",
        {
            "dataset": cfg["dataset"],
            "revision": cfg["dataset_revision"],
            "source_programming_manifest_sha256": sha(source / "manifest.json"),
            "raw_human_answers_sha256": sha(raw),
            "configuration_sha256": sha(ROOT / "config.json"),
            "counts": report["counts"],
            "prompt_ids": {name: sorted(r["id"] for r in rows) for name, rows in splits.items()},
            "reference_scope": "MMD uses only the exact question's two human-rated answers; "
            "training cannot access held-out MMD references. "
            "Word KL is diagnostic against the old training-only PMF.",
            "prepared_sha256": {
                p.name: sha(p)
                for p in sorted(out.iterdir())
                if p.is_file() and p.name != "manifest.json"
            },
        },
    )
    print(json.dumps(report, indent=2), flush=True)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--raw", type=Path, default=REPO / "outputs/helpsteer-data/raw/train.jsonl.gz"
    )
    parser.add_argument(
        "--source", type=Path, default=REPO / "experiments/helpsteer-programming/data"
    )
    parser.add_argument("--out", type=Path, default=ROOT / "data")
    args = parser.parse_args()
    prepare(args.raw.resolve(), args.source.resolve(), args.out.resolve())
