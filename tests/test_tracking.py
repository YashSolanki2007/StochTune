import json

import pytest

from stochastic_llm.io import write_json
from stochastic_llm.tracking import tracked, tracking_settings


def test_required_online_logging_cannot_be_silently_disabled(monkeypatch):
    config = {"wandb": {"enabled": True, "mode": "online"}}
    monkeypatch.setenv("WANDB_MODE", "offline")
    with pytest.raises(RuntimeError, match="disagrees"):
        tracking_settings(config)
    monkeypatch.delenv("WANDB_MODE")
    monkeypatch.setenv("WANDB_DISABLED", "true")
    with pytest.raises(RuntimeError, match="required"):
        tracking_settings(config)


def test_offline_tracking_lifecycle_and_artifacts(tmp_path, monkeypatch):
    monkeypatch.delenv("WANDB_MODE", raising=False)
    monkeypatch.delenv("WANDB_DISABLED", raising=False)
    config = {
        "output_dir": str(tmp_path / "result"),
        "wandb": {
            "enabled": True,
            "mode": "offline",
            "entity": "local-test",
            "project": "stochastic-test",
            "directory": str(tmp_path / "tracking"),
        },
    }

    @tracked("test")
    def operation(settings):
        write_json(tmp_path / "result" / "metrics.json", {"loss": 0.5})
        return {"loss": 0.5}

    assert operation(config) == {"loss": 0.5}
    record = json.loads((tmp_path / "result" / "wandb_run.json").read_text())
    assert record["id"]
    assert list((tmp_path / "tracking").glob("wandb/offline-run-*/run-*.wandb"))
