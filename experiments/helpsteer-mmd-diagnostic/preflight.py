"""CPU validation of the fixed-question, zero-quality diagnostic runtime."""

import hashlib
import json
from pathlib import Path

import numpy as np
from diagnostic import select_questions

ROOT = Path(__file__).resolve().parent


def check(cfg, data):
    if not cfg["gpu_submission_authorized"]:
        raise ValueError("Diagnostic GPU budget not authorized")
    if cfg["approved_allocation_seconds"] > 5400 or cfg["approved_maximum_gpus"] != 2:
        raise ValueError("Diagnostic is capped at two GPUs for 90 minutes total")
    if not 0 < cfg["train_budget_seconds"] <= cfg["approved_allocation_seconds"] - 1200:
        raise ValueError("Reserve twenty minutes for evaluation and saving")
    if cfg.get("quality_weight") != 0 or cfg.get("resume_checkpoint"):
        raise ValueError("This diagnostic must start fresh without quality reward")
    if (cfg["prompts_per_group"], cfg["samples_per_question"], cfg["global_group_size"]) != (
        4,
        16,
        64,
    ):
        raise ValueError("Need four fixed training questions and sixteen answers each")
    if cfg["distribution_weight"] != 0.5 or cfg["reference_kl_beta"] != 0.002:
        raise ValueError("Retain the original distribution and base-model KL weights")
    if cfg["objective_loss_divisor"] != 1.502 or cfg["reference_kl_centering"] != "prompt-loo":
        raise ValueError("Retain fixed loss scaling and same-question KL centering")
    if cfg["semantic_kernel_name"] != "energy":
        raise ValueError("This diagnostic tests the energy-distance kernel")
    selection = json.loads((data / "diagnostic-selection.json").read_text())
    refs = [json.loads(s) for s in (data / "semantic-references.jsonl").read_text().splitlines()]
    if selection != select_questions(refs, cfg["seed"]):
        raise ValueError("Fixed question selection differs from the preregistered procedure")
    metadata = json.loads((data / "semantic-metadata.json").read_text())
    if not set(selection["seen_ids"]) <= set(metadata["training_prompt_ids"]):
        raise ValueError("Diagnostic training questions must belong to the original training split")
    if set(selection["unseen_ids"]) & set(metadata["training_prompt_ids"]):
        raise ValueError("Unseen questions leaked into training")
    expected = {r["case_id"]: r["split"] for r in refs}
    for scope, split in (("seen", "train"), ("unseen", "validation")):
        rows = [json.loads(s) for s in (data / f"policy_{split}.jsonl").read_text().splitlines()]
        if not set(selection[f"{scope}_ids"]) <= {r["id"] for r in rows}:
            raise ValueError("Diagnostic questions are missing prepared tokenization")
        if any(expected[key] != split for key in selection[f"{scope}_ids"]):
            raise ValueError("Diagnostic split mismatch")
    calibration_file = data / "semantic-calibration.json"
    if (
        hashlib.sha256(calibration_file.read_bytes()).hexdigest()
        != cfg["semantic_calibration_sha256"]
    ):
        raise ValueError("Frozen calibration changed")
    calibration = json.loads(calibration_file.read_text())
    if cfg["semantic_kernel_scale"] != calibration["kernel_scales"]["energy"]:
        raise ValueError("Energy scale differs from the completed-run calibration")
    for filename, key in (
        ("semantic-reference-vectors.npz", "source_reference_vectors_sha256"),
        ("semantic-metadata.json", "source_semantic_metadata_sha256"),
    ):
        if hashlib.sha256((data / filename).read_bytes()).hexdigest() != calibration[key]:
            raise ValueError("Reference data differs from frozen calibration")
    if (metadata["encoder_model"], metadata["encoder_revision"]) != (
        cfg["semantic_encoder_model"],
        cfg["semantic_encoder_revision"],
    ):
        raise ValueError("Reference and answer encoder identities differ")
    vectors = np.load(data / "semantic-reference-vectors.npz", allow_pickle=False)
    if not set(selection["seen_ids"] + selection["unseen_ids"]) <= set(
        vectors["case_ids"].tolist()
    ):
        raise ValueError("Missing cached reference vectors")
    if not np.isfinite(vectors["vectors"]).all():
        raise ValueError("Nonfinite reference vectors")
    return {
        "preflight": "passed",
        "quality_weight": 0,
        "training_questions": 4,
        "samples_per_question": 16,
        "global_batch": 64,
        "unseen_questions": 4,
        "maximum_gpus": 2,
        "maximum_wall_seconds": cfg["approved_allocation_seconds"],
    }


def preflight():
    cfg = json.loads((ROOT / "config.json").read_text())
    data = ROOT / "data"
    manifest = json.loads((data / "manifest.json").read_text())
    if (
        hashlib.sha256((ROOT / "config.json").read_bytes()).hexdigest()
        != manifest["configuration_sha256"]
    ):
        raise ValueError("Diagnostic configuration changed after deployment")
    for name, expected in manifest["prepared_sha256"].items():
        if hashlib.sha256((data / name).read_bytes()).hexdigest() != expected:
            raise ValueError(f"Prepared data changed: {name}")
    print(json.dumps(check(cfg, data)), flush=True)


if __name__ == "__main__":
    preflight()
