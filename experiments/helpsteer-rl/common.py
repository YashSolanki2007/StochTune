"""Shared HelpSteer experiment utilities; no model loading at import time."""

import hashlib
import json
import os
import re
import sys
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parent
REPO = ROOT.parents[1]
SUPPORT = ROOT / "support"
if not SUPPORT.exists():
    SUPPORT = REPO / "experiments/shakespeare-quality-pilot"
sys.path.insert(0, str(SUPPORT))
from passage_reward import PassageDistribution, quality_advantages  # noqa: E402, F401
from pilot import read_rows, words, write_json  # noqa: E402, F401


def config():
    return json.loads((ROOT / "config.json").read_text())


def prompt_id(prompt):
    # Group whitespace-equivalent conversations before any splitting.
    return hashlib.sha256(" ".join(prompt.split()).encode()).hexdigest()


def messages(prompt):
    parts = re.split(r"<extra_id_1>(User|Assistant)\s*\n?", prompt)
    result = [{"role": "user", "content": parts[0].strip()}]
    for role, text in zip(parts[1::2], parts[2::2]):
        result.append({"role": role.lower(), "content": text.strip()})
    if not result[0]["content"]:
        result.pop(0)
    if not result or result[-1]["role"] != "user":
        raise ValueError("Conversation must end with a user request")
    return result


def encoded_response(tokenizer, prompt, response):
    return tokenizer.apply_chat_template(
        messages(prompt) + [{"role": "assistant", "content": response}],
        tokenize=True,
        add_generation_prompt=False,
    )


def write_rows(path, rows):
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text("".join(json.dumps(row, ensure_ascii=False) + "\n" for row in rows))


def bt_loss(scores_one, scores_two, strengths):
    """Positive strength prefers response TWO; ties have a soft target of 0.5."""
    import torch
    import torch.nn.functional as F

    target = (torch.sign(strengths.to(scores_one.dtype)) + 1) / 2
    return F.binary_cross_entropy_with_logits(scores_two - scores_one, target)


def reward_model(cfg, device, adapter=None):
    import torch
    from peft import LoraConfig, PeftModel, get_peft_model
    from transformers import AutoModelForSequenceClassification, AutoTokenizer

    tokenizer = AutoTokenizer.from_pretrained(
        cfg["model"], revision=cfg["revision"], padding_side="right"
    )
    if tokenizer.pad_token_id is None:
        tokenizer.pad_token = tokenizer.eos_token
    base = AutoModelForSequenceClassification.from_pretrained(
        cfg["model"],
        revision=cfg["revision"],
        num_labels=1,
        torch_dtype=torch.bfloat16,
        device_map={"": device},
        attn_implementation="sdpa",
    )
    base.config.pad_token_id = tokenizer.pad_token_id
    if adapter is not None:
        model = PeftModel.from_pretrained(base, adapter, is_trainable=False)
    else:
        model = get_peft_model(
            base,
            LoraConfig(
                task_type="SEQ_CLS",
                r=cfg["lora_rank"],
                lora_alpha=cfg["lora_alpha"],
                lora_dropout=0.0,
                target_modules=["q_proj", "k_proj", "v_proj", "o_proj"],
                modules_to_save=["score"],
                bias="none",
            ),
        )
    return tokenizer, model.eval()


def sequence_scores(model, tokenizer, sequences):
    if not sequences or max(map(len, sequences)) > 1024:
        raise ValueError("Reward input empty or exceeds the explicit 1024-token limit")
    batch = tokenizer.pad({"input_ids": sequences}, padding=True, return_tensors="pt")
    return model(**batch.to(model.device), use_cache=False).logits[:, 0].float()


class FrozenPreferenceReward:
    def __init__(self, cfg, output, device):
        self.cfg = cfg
        self.tokenizer, self.model = reward_model(cfg, device, output / "adapter")
        self.model.requires_grad_(False)
        self.calibration = json.loads((output / "calibration.json").read_text())

    def score(self, rows):
        import torch

        raw = []
        for start in range(0, len(rows), 8):
            sequences = [
                encoded_response(self.tokenizer, r["prompt"], r["passage"])
                for r in rows[start : start + 8]
            ]
            with torch.inference_mode():
                raw.extend(sequence_scores(self.model, self.tokenizer, sequences).tolist())
        z = (np.array(raw) - self.calibration["center"]) / self.calibration["scale"]
        normalized = 1 / (1 + np.exp(-np.clip(z, -30, 30)))
        quality = normalized * np.array([r["terminated"] for r in rows])
        return [
            dict(raw_reward=float(r), normalized_reward=float(n), quality_reward=float(q))
            for r, n, q in zip(raw, normalized, quality)
        ], quality


def start_tracking(cfg, output, stage, **extra):
    import wandb

    if os.environ.get("WANDB_MODE", "online") != "online" or os.environ.get(
        "WANDB_DISABLED", "false"
    ).lower() in ("1", "true", "yes"):
        raise RuntimeError("Online W&B logging is required")
    run = wandb.init(
        entity=cfg["wandb_entity"],
        project=cfg["wandb_project"],
        name=f"helpsteer-{stage}-{os.environ.get('SLURM_JOB_ID')}",
        group="helpsteer-human-preference-v1",
        job_type=stage,
        mode="online",
        config={**cfg, **extra},
        dir=str(output),
        settings=wandb.Settings(base_url=cfg["wandb_base_url"]),
    )
    write_json(output / "wandb_run.json", {"id": run.id, "url": run.url})
    write_json(output / "run_config.json", {**cfg, **extra})
    artifact = wandb.Artifact(f"helpsteer-source-{run.id}", type="source")
    paths = list(ROOT.glob("*.py")) + list(ROOT.glob("*.json")) + list(ROOT.glob("*.sbatch"))
    paths += [ROOT / "data/manifest.json"]
    paths += [
        SUPPORT / name
        for name in (
            "ddp_passage_rl.py",
            "passage_reward.py",
            "pilot.py",
            "judge.py",
            "passage_rl.py",
            "quality.py",
        )
    ]
    hashes = {}
    for path in paths:
        name = str(path.relative_to(REPO))
        hashes[name] = hashlib.sha256(path.read_bytes()).hexdigest()
        artifact.add_file(str(path), name=name)
    write_json(output / "source-manifest.json", hashes)
    artifact.add_dir(str(ROOT / "data"), name="prepared-data")
    run.log_artifact(artifact)
    return run


def log_results(run, output, adapter=True):
    import wandb

    artifact = wandb.Artifact(f"helpsteer-results-{run.id}", type="results")
    for path in output.iterdir():
        if path.is_file() and path.suffix in (".json", ".jsonl", ".md"):
            artifact.add_file(str(path), name=path.name)
    if adapter:
        artifact.add_dir(str(output / "adapter"), name="adapter")
    run.log_artifact(artifact)
