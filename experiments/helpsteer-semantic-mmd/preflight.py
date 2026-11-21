"""CPU checks of the authorized semantic-MMD runtime, data, and scorer identity."""

import hashlib
import json
from pathlib import Path

import numpy as np
from common import ROOT, config, make_distribution, read_rows, verify_reward_identity
from validation import require_reward_audit


def preflight():
    cfg = config()
    if not cfg.get("gpu_submission_authorized"):
        raise RuntimeError("GPU submission has not been authorized for this runtime")
    if not 600 < cfg["approved_allocation_seconds"] <= 5400 or cfg["approved_maximum_gpus"] != 2:
        raise ValueError("Each policy is capped at two GPUs and at most 90 minutes")
    if not 0 < cfg["train_budget_seconds"] <= cfg["approved_allocation_seconds"] - 600:
        raise ValueError("Reserve at least ten minutes per policy for evaluation and saving")
    if cfg["distribution_objective"] != "conditional-semantic-mmd":
        raise ValueError("The actual policy reward must include conditional semantic MMD")
    if cfg.get("reference_kl_centering", "none") not in ("none", "prompt-loo"):
        raise ValueError("Unknown KL centering mode")
    if cfg.get("semantic_kernel_name", "rbf") not in ("rbf", "energy"):
        raise ValueError("Unknown semantic kernel")
    if (cfg["prompts_per_group"], cfg["global_group_size"]) != (4, 16):
        raise ValueError("Expected four questions with four answers each")
    if cfg["distribution_weight"] != 0.5 or cfg["semantic_kernel_scales"] != [0.5, 1.0, 2.0]:
        raise ValueError("Unexpected reward weight or kernel scales")
    if cfg["max_prompt_tokens"] + cfg["max_new_tokens"] + 64 > cfg["reward_max_sequence_length"]:
        raise ValueError("Reward context cannot fit the full rollout")
    data = ROOT / "data"
    manifest = json.loads((data / "manifest.json").read_text())
    if (
        hashlib.sha256((ROOT / "config.json").read_bytes()).hexdigest()
        != manifest["configuration_sha256"]
    ):
        raise ValueError("Runtime config differs from its manifest")
    for name, expected in manifest["prepared_sha256"].items():
        if hashlib.sha256((data / name).read_bytes()).hexdigest() != expected:
            raise ValueError(f"Prepared data changed: {name}")
    sets = {
        name: {r["id"] for r in read_rows(data / f"policy_{name}.jsonl")}
        for name in ("train", "validation", "test")
    }
    for a, b in (("train", "validation"), ("train", "test"), ("validation", "test")):
        if sets[a] & sets[b]:
            raise ValueError("Policy splits overlap")
    refs = read_rows(data / "semantic-references.jsonl")
    if {r["case_id"] for r in refs} != set.union(*sets.values()):
        raise ValueError("Every policy question must have exactly its own references")
    for row in refs:
        if row["case_id"] not in sets[row["split"]] or len(set(row["responses"])) != 2:
            raise ValueError("Reference-to-question association is invalid")
        if any(min(attributes.values()) < 3 for attributes in row["human_attributes"]):
            raise ValueError("Reference fails the human-quality floor")
    metadata = json.loads((data / "semantic-metadata.json").read_text())
    if cfg.get("semantic_calibration_runtime_file"):
        calibration_path = ROOT / cfg["semantic_calibration_runtime_file"]
        if (
            hashlib.sha256(calibration_path.read_bytes()).hexdigest()
            != cfg["semantic_calibration_sha256"]
        ):
            raise ValueError("Frozen calibration report changed")
        calibration = json.loads(calibration_path.read_text())
        for name, field in (
            ("semantic-reference-vectors.npz", "source_reference_vectors_sha256"),
            ("semantic-metadata.json", "source_semantic_metadata_sha256"),
        ):
            if hashlib.sha256((data / name).read_bytes()).hexdigest() != calibration[field]:
                raise ValueError("Calibration belongs to different semantic reference data")
        if (
            cfg["semantic_kernel_scale"]
            != calibration["kernel_scales"][cfg["semantic_kernel_name"]]
        ):
            raise ValueError("Runtime kernel scale differs from the frozen calibration")
        expected_divisor = 1 + cfg["distribution_weight"] + cfg["reference_kl_beta"]
        if abs(cfg["objective_loss_divisor"] - expected_divisor) > 1e-12:
            raise ValueError("Objective normalization must preserve all three coefficient ratios")
    elif cfg.get("semantic_kernel_scale", 1.0) != 1.0:
        raise ValueError("A scaled kernel requires a frozen calibration report")
    if set(metadata["training_prompt_ids"]) != sets["train"]:
        raise ValueError("Training reference whitelist differs from policy-training IDs")
    if (
        metadata["encoder_model"] != cfg["semantic_encoder_model"]
        or metadata["encoder_revision"] != cfg["semantic_encoder_revision"]
    ):
        raise ValueError("Reference and rollout encoder identities differ")
    archive = np.load(data / "semantic-reference-vectors.npz", allow_pickle=False)
    if set(archive["case_ids"].tolist()) != {r["case_id"] for r in refs}:
        raise ValueError("Missing or mismatched cached reference vectors")
    vectors = archive["vectors"]
    if vectors.shape != (len(refs), 2, 384) or not np.all(np.isfinite(vectors)):
        raise ValueError("Invalid reference vectors")
    np.testing.assert_allclose(np.linalg.norm(vectors, axis=-1), 1, atol=1e-5)
    train = vectors[[key in sets["train"] for key in archive["case_ids"].tolist()]]
    distances = ((train[:, 0] - train[:, 1]) ** 2).sum(1)
    if abs(float(np.median(distances[distances > 1e-6])) - metadata["bandwidth_squared"]) > 1e-8:
        raise ValueError("Kernel calibration must use training reference pairs only")
    audit = json.loads((data / "cpu-audit.json").read_text())
    if (
        min(
            audit["collapsed_reference_detected_fraction"],
            audit["wrong_question_detected_fraction"],
        )
        < 0.95
    ):
        raise ValueError("CPU representation checks did not pass")
    word_target = make_distribution(
        cfg, [r["response"] for r in read_rows(data / "reference.jsonl")]
    )
    vocabulary = json.loads((data / "vocabulary.json").read_text())
    np.testing.assert_array_equal(
        word_target.probabilities, [r["probability"] for r in vocabulary["categories"]]
    )
    ready = json.loads((ROOT / "outputs/reward-ready.json").read_text())
    for key in ("reward_model_name", "reward_model_revision", "reward_pretrained_full"):
        if ready["reward_model_config"][key] != cfg[key]:
            raise ValueError("Frozen scorer differs from its audited identity")
    checks = json.loads((Path(ready["audit_path"]) / "checks.json").read_text())
    require_reward_audit(
        checks, hashlib.sha256((data / "reward_validation.jsonl").read_bytes()).hexdigest()
    )
    verify_reward_identity(ready, checks)
    print(
        json.dumps(
            {
                "preflight": "passed",
                "objective": cfg["distribution_objective"],
                "counts": {k: len(v) for k, v in sets.items()},
                "references_per_question": 2,
                "questions_per_update": 4,
                "answers_per_question": 4,
                "lambda": cfg["distribution_weight"],
                "teacher_training": False,
            }
        ),
        flush=True,
    )


if __name__ == "__main__":
    preflight()
