"""Package the isolated two-H200 run, including the cached frozen small encoder."""

import argparse
import hashlib
import json
import shutil
from pathlib import Path

ROOT = Path(__file__).resolve().parent
REPO = ROOT.parents[1]


def stage(output, reward_ready, config_path=None):
    from huggingface_hub import snapshot_download

    output.mkdir(parents=True, exist_ok=False)
    for name in (
        "common.py",
        "train_policy.py",
        "audit.py",
        "validation.py",
        "word_distribution.py",
    ):
        shutil.copy2(REPO / "experiments/helpsteer-rl-v2" / name, output / name)
    for name in (
        "mmd.py",
        "embedding.py",
        "semantic_runtime.py",
        "preflight.py",
        "run_policy.py",
        "policy.sbatch",
        "README.md",
    ):
        shutil.copy2(ROOT / name, output / name)
    support = output / "support"
    support.mkdir()
    for name in (
        "ddp_passage_rl.py",
        "judge.py",
        "passage_reward.py",
        "passage_rl.py",
        "pilot.py",
        "quality.py",
    ):
        shutil.copy2(REPO / "experiments/shakespeare-quality-pilot" / name, support / name)
    shutil.copytree(ROOT / "data", output / "data")
    (output / "outputs").mkdir()
    shutil.copy2(reward_ready, output / "outputs/reward-ready.json")
    cfg = json.loads((config_path or ROOT / "config.json").read_text())
    if cfg.get("semantic_calibration_report"):
        calibration = REPO / cfg["semantic_calibration_report"]
        if (
            hashlib.sha256(calibration.read_bytes()).hexdigest()
            != cfg["semantic_calibration_sha256"]
        ):
            raise ValueError("Calibrated profile no longer matches its CPU calibration report")
        destination = output / "data/semantic-calibration.json"
        shutil.copy2(calibration, destination)
        cfg["semantic_calibration_runtime_file"] = "data/semantic-calibration.json"
    encoder_cache = Path(
        snapshot_download(
            cfg["semantic_encoder_model"],
            revision=cfg["semantic_encoder_revision"],
            local_files_only=True,
        )
    )
    encoder_dir = output / "encoder"
    encoder_dir.mkdir()
    for name in (
        "config.json",
        "model.safetensors",
        "tokenizer.json",
        "tokenizer_config.json",
        "special_tokens_map.json",
        "vocab.txt",
        "sentence_bert_config.json",
    ):
        shutil.copy2(encoder_cache / name, encoder_dir / name)
    cfg.update(
        approved_allocation_seconds=cfg.get("approved_allocation_seconds", 5400),
        approved_maximum_gpus=2,
        approved_maximum_gpu_hours=2 * cfg.get("approved_allocation_seconds", 5400) / 3600,
        semantic_encoder_local_path="encoder",
    )
    # Packaging a new profile must not reuse the completed v1 run's spending approval.
    cfg.setdefault("gpu_submission_authorized", False)
    (output / "config.json").write_text(json.dumps(cfg, indent=2) + "\n")
    manifest_path = output / "data/manifest.json"
    manifest = json.loads(manifest_path.read_text())
    manifest["prepared_configuration_sha256"] = manifest["configuration_sha256"]
    manifest["configuration_sha256"] = hashlib.sha256(
        (output / "config.json").read_bytes()
    ).hexdigest()
    if cfg.get("semantic_calibration_report"):
        manifest["prepared_sha256"]["semantic-calibration.json"] = cfg[
            "semantic_calibration_sha256"
        ]
    manifest_path.write_text(json.dumps(manifest, indent=2) + "\n")
    hashes = {
        str(p.relative_to(output)): hashlib.sha256(p.read_bytes()).hexdigest()
        for p in sorted(output.rglob("*"))
        if p.is_file()
    }
    (output / "deployment-manifest.json").write_text(json.dumps(hashes, indent=2) + "\n")
    print(json.dumps({"staged": str(output), "files": len(hashes), "submission": False}))


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--reward-ready", type=Path, required=True)
    parser.add_argument("--config", type=Path, default=ROOT / "config.json")
    args = parser.parse_args()
    stage(args.output.resolve(), args.reward_ready.resolve(), args.config.resolve())
