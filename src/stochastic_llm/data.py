"""Reproducible data splits with no target-parameter overlap."""

from pathlib import Path

import numpy as np

from .distributions import (
    HELD_OUT_FAMILIES,
    TRAIN_FAMILIES,
    distribution,
    format_number,
    prompt_for,
    sample_parameters,
)
from .io import file_sha256, fingerprint, require_empty_directory, write_json, write_jsonl


def generate_data(config):
    output = Path(config["output_dir"])
    counts = {
        key: int(config[key]) for key in ("train_configs", "validation_configs", "test_configs")
    }
    if min(counts.values()) < 1 or int(config["samples_per_config"]) < 1:
        raise ValueError("All split sizes and samples_per_config must be positive")
    output = require_empty_directory(output)
    rng = np.random.default_rng(config["seed"])
    decimals = int(config.get("decimals", 5))
    used = set()

    def make_cases(count, split):
        cases = []
        for i in range(count):
            category = (
                ("unseen_parameters", "unseen_family", "paraphrase")[i % 3]
                if split == "test"
                else "id"
            )
            families = HELD_OUT_FAMILIES if category == "unseen_family" else TRAIN_FAMILIES
            # Avoid coupling family selection to the three-way category cycle.
            family = families[(i // 3 if split == "test" else i) % len(families)]
            while True:
                parameters = sample_parameters(family, rng, outside=category == "unseen_parameters")
                key = fingerprint({"family": family, "parameters": parameters})
                if key not in used:
                    used.add(key)
                    break
            case = {
                "id": f"{split}-{i:05d}",
                "family": family,
                "parameters": parameters,
                "category": category,
            }
            template = 2 if category == "paraphrase" else int(rng.integers(0, 2))
            case["prompt"] = prompt_for(case, template)
            cases.append(case)
        return cases

    training = make_cases(counts["train_configs"], "train")
    validation = make_cases(counts["validation_configs"], "validation")
    test = make_cases(counts["test_configs"], "test")
    for name, rows in (
        ("train_configs", training),
        ("validation_configs", validation),
        ("test", test),
    ):
        write_jsonl(output / f"{name}.jsonl", rows)

    def completions(cases, samples):
        for case in cases:
            values = distribution(case).rvs(size=samples, random_state=rng)
            for value in np.atleast_1d(values):
                yield {
                    "case_id": case["id"],
                    "prompt": case["prompt"],
                    "completion": format_number(value, case["family"], decimals),
                }

    write_jsonl(output / "train.jsonl", completions(training, int(config["samples_per_config"])))
    write_jsonl(
        output / "validation.jsonl",
        completions(validation, int(config.get("validation_samples_per_config", 8))),
    )
    files = sorted(output.glob("*.jsonl"))
    manifest = {
        "config": config,
        "format_version": 1,
        "files": {p.name: file_sha256(p) for p in files},
        "training_examples": len(training) * int(config["samples_per_config"]),
    }
    write_json(output / "manifest.json", manifest)
    return manifest
