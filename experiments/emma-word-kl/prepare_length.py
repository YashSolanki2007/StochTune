"""Stage one length-controlled REINFORCE experiment without transferring GPU approval."""

import argparse
import hashlib
import json
from pathlib import Path

from common import ROOT
from prepare_long import long_profile
from stage_runtime import stage


def length_profile():
    path = ROOT / "length-penalty-calibration.json"
    calibration = json.loads(path.read_text())
    if calibration["selection_uses_test_generations"] is not False:
        raise ValueError("Coefficient selection must exclude test generations")
    cfg = long_profile(gpus=4)
    cfg.update(
        experiment_profile="emma-reinforce-length-4gpu-v1",
        wandb_group="emma-reinforce-length-4gpu-v1",
        length_penalty_weight=calibration["selected_weight"],
        length_min_words=250,
        length_max_words=350,
        length_penalty_unit_words=100,
        length_penalty_calibration_file=path.name,
        length_penalty_calibration_sha256=hashlib.sha256(path.read_bytes()).hexdigest(),
        authorization="Prepared length-penalty experiment only; final approval of a new GPU "
        "allocation is still required. Prior job approval is not reused.",
        experiment_purpose="Same 500-step/four-rank training setup as job 524624; add a fixed "
        "word-range penalty before prompt-LOO quality advantages. "
        "The target word PMF and both KL weights remain unchanged.",
    )
    assert not cfg["gpu_submission_authorized"]
    return cfg


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--stage", action="store_true")
    args = parser.parse_args()
    output = args.output.resolve()
    cfg = length_profile()
    config_path = output.with_suffix(".json") if args.stage else output
    if config_path.exists():
        raise FileExistsError(config_path)
    config_path.parent.mkdir(parents=True, exist_ok=True)
    config_path.write_text(json.dumps(cfg, indent=2) + "\n")
    if args.stage:
        stage(output, config_path)
    print(
        json.dumps(
            {
                "prepared": str(output),
                "length_weight": cfg["length_penalty_weight"],
                "word_range": [250, 350],
                "steps": 500,
                "gpu_submission_authorized": False,
                "jobs_submitted": 0,
            }
        )
    )
