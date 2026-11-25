"""Check data provenance, chapter isolation and the exact training-only target; CPU only."""

import hashlib
import json

import numpy as np
from common import ROOT, BucketedWordDistribution


def rows(path):
    return [json.loads(s) for s in path.read_text().splitlines() if s.strip()]


def preflight():
    cfg = json.loads((ROOT / "config.json").read_text())
    data = ROOT / "data"
    manifest = json.loads((data / "manifest.json").read_text())
    assert (
        hashlib.sha256((ROOT / "config.json").read_bytes()).hexdigest()
        == manifest["configuration_sha256"]
    )
    for name, expected in manifest["prepared_sha256"].items():
        assert hashlib.sha256((data / name).read_bytes()).hexdigest() == expected, name
    sets = {name: set(ids) for name, ids in manifest["chapter_ids"].items()}
    assert [len(sets[n]) for n in ("train", "validation", "test")] == [39, 8, 8]
    for i, name in enumerate(sets):
        for other in list(sets)[i + 1 :]:
            assert sets[name].isdisjoint(sets[other]), (name, other)
    cases = {name: rows(data / f"policy_{name}.jsonl") for name in sets}
    for name, values in cases.items():
        assert len(values) >= 16
        assert len(values) == manifest["continuation_counts"][name]
        assert all(r["group_id"] in sets[name] for r in values)
        assert all(len(r["prompt_ids"]) <= cfg["max_prompt_tokens"] for r in values)
        assert len({r["id"] for r in values}) == len(values)
    references = rows(data / "reference.jsonl")
    assert {r["id"] for r in references} == {r["id"] for r in cases["train"]}
    assert all(r["group_id"] in sets["train"] for r in references)
    reference_lookup = {r["id"]: r["response"] for r in references}
    assert all(reference_lookup[r["id"]] == r["response"] for r in cases["train"])
    heldout_hashes = {
        hashlib.sha256(r["response"].encode()).hexdigest()
        for name in ("validation", "test")
        for r in cases[name]
    }
    assert not any(
        hashlib.sha256(r["response"].encode()).hexdigest() in heldout_hashes for r in references
    )
    target = BucketedWordDistribution(
        [r["response"] for r in references],
        cfg["vocabulary_max_words"],
        cfg["vocabulary_min_count"],
    )
    vocabulary = json.loads((data / "vocabulary.json").read_text())["categories"]
    np.testing.assert_allclose(target.probabilities, [r["probability"] for r in vocabulary])
    assert cfg["distribution_weight"] == 0.1 and cfg["reference_kl_beta"] == 0.002
    assert not manifest["human_preference_labels"]
    assert not manifest["gpu_used"]
    assert not cfg["gpu_submission_authorized"]
    return {
        "passed": True,
        "cpu_only": True,
        "chapter_counts": {k: len(v) for k, v in sets.items()},
        "continuation_counts": manifest["continuation_counts"],
        "word_categories": len(target.probabilities),
        "gpu_submission_authorized": False,
    }


if __name__ == "__main__":
    print(json.dumps(preflight(), indent=2))
