import json

import pytest

from stochastic_llm.data import generate_data
from stochastic_llm.io import fingerprint, read_jsonl


def test_reproducible_data_and_disjoint_parameter_splits(tmp_path):
    config = {
        "seed": 3,
        "decimals": 5,
        "train_configs": 24,
        "samples_per_config": 3,
        "validation_configs": 12,
        "validation_samples_per_config": 2,
        "test_configs": 36,
    }
    first = generate_data(config | {"output_dir": str(tmp_path / "first")})
    second = generate_data(config | {"output_dir": str(tmp_path / "second")})
    assert first["files"] == second["files"]
    assert len(read_jsonl(tmp_path / "first" / "train.jsonl")) == 72
    keys = []
    for name in ("train_configs", "validation_configs", "test"):
        rows = read_jsonl(tmp_path / "first" / f"{name}.jsonl")
        keys.extend(
            fingerprint({"family": r["family"], "parameters": r["parameters"]}) for r in rows
        )
    assert len(keys) == len(set(keys))
    test = read_jsonl(tmp_path / "first" / "test.jsonl")
    assert {r["category"] for r in test} == {"unseen_parameters", "unseen_family", "paraphrase"}
    assert {r["family"] for r in test if r["category"] == "unseen_family"} == {"weibull", "poisson"}
    # Existing data must not be silently replaced.
    with pytest.raises(FileExistsError):
        generate_data(config | {"output_dir": str(tmp_path / "first")})
    assert json.loads((tmp_path / "first" / "manifest.json").read_text())["training_examples"] == 72
