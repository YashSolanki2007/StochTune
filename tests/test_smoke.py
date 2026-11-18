import json

from stochastic_llm.smoke import smoke


def test_offline_end_to_end(tmp_path, monkeypatch):
    monkeypatch.setenv("HF_HUB_OFFLINE", "1")
    monkeypatch.setenv("TRANSFORMERS_OFFLINE", "1")
    result = smoke(tmp_path / "smoke", with_tracking=True)
    assert result["passed"]
    assert (tmp_path / "smoke" / "sft" / "adapter" / "adapter_model.safetensors").is_file()
    comparison = json.loads((tmp_path / "smoke" / "comparison.json").read_text())
    assert len(comparison) == 2
    assert comparison[0]["summary"]["all"]["requested"] == 9
    assert (tmp_path / "smoke" / "sft" / "wandb_run.json").is_file()
