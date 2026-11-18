import json

import pytest
import torch

from stochastic_llm.evaluate import compare_runs, generate_samples
from stochastic_llm.io import file_sha256, write_json, write_jsonl
from stochastic_llm.smoke import create_tiny_model


def test_generation_preserves_invalid_outputs_and_uses_fresh_batch_streams(tmp_path):
    from transformers import AutoTokenizer

    tokenizer = AutoTokenizer.from_pretrained(create_tiny_model(tmp_path / "tiny"))
    endings = ["1.25000", "2", "12345678", "<unk>3"]

    class FakeModel:
        device = torch.device("cpu")

        def __init__(self):
            self.seeds = []
            self.next = 0

        def generate(self, input_ids, attention_mask, generation_config):
            self.seeds.append(torch.initial_seed())
            assert generation_config.temperature == 1
            assert generation_config.top_k == 0
            assert generation_config.top_p == 1
            assert generation_config.repetition_penalty == 1
            outputs = []
            for _ in range(len(input_ids)):
                text = endings[self.next]
                ids = tokenizer.encode(text, add_special_tokens=False)
                if self.next != 2:
                    ids += [tokenizer.eos_token_id]
                self.next += 1
                ids += [tokenizer.pad_token_id] * (8 - len(ids))
                outputs.append(ids)
            return torch.cat((input_ids, torch.tensor(outputs)), dim=1)

    model = FakeModel()
    config = {"samples_per_prompt": 4, "batch_size": 2, "max_new_tokens": 8, "seed": 4}
    samples = list(
        generate_samples(model, tokenizer, {"id": "test-1", "prompt": "Number:"}, config)
    )
    assert [s["valid"] for s in samples] == [True, True, False, False]
    assert [s["value"] for s in samples] == [1.25, 2.0, None, None]
    assert not samples[2]["terminated"]
    assert "<unk>" in samples[3]["text"]
    assert len(set(model.seeds)) == 2


def test_comparison_rejects_revision_mismatch(tmp_path):
    cases = tmp_path / "cases.jsonl"
    write_jsonl(cases, [{"id": "test-1"}])
    for name, revision in (("base", "commit-a"), ("adapter", "commit-b")):
        folder = tmp_path / name
        folder.mkdir()
        write_json(
            folder / "identity.json",
            {
                "config": {
                    "model": "example",
                    "revision": "main",
                    "cases": str(cases),
                    "samples_per_prompt": 1,
                    "max_new_tokens": 32,
                    "decimals": 5,
                    "batch_size": 1,
                    "seed": 1,
                },
                "cases_sha256": file_sha256(cases),
            },
        )
        write_json(folder / "metadata.json", {"resolved_revision": revision})
        write_json(
            folder / "results.json",
            [
                {
                    "id": "test-1",
                    "family": "normal",
                    "category": "test",
                    "requested": 1,
                    "valid": 0,
                    "normalized_wasserstein": None,
                    "ks_statistic": None,
                }
            ],
        )
    with pytest.raises(ValueError, match="revisions"):
        compare_runs([tmp_path / "base", tmp_path / "adapter"], tmp_path / "comparison.json")
    assert (
        json.loads((tmp_path / "base" / "metadata.json").read_text())["resolved_revision"]
        == "commit-a"
    )
