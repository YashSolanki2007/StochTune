import json

import numpy as np
import pytest
import torch
from peft import LoraConfig, PeftModel, get_peft_model
from safetensors.torch import load_file
from transformers import AutoModelForCausalLM, AutoTokenizer

from stochastic_llm.data import generate_data
from stochastic_llm.io import read_config
from stochastic_llm.rl import completion_log_probs, policy_update, train_rl
from stochastic_llm.smoke import create_tiny_model


def tiny_actor(folder):
    model_path = create_tiny_model(folder / "base")
    tokenizer = AutoTokenizer.from_pretrained(model_path)
    base = AutoModelForCausalLM.from_pretrained(model_path)
    model = get_peft_model(
        base,
        LoraConfig(
            r=4,
            lora_alpha=8,
            lora_dropout=0.05,
            target_modules=["q_proj", "k_proj", "v_proj", "o_proj"],
            task_type="CAUSAL_LM",
        ),
    )
    initial = folder / "initial-adapter"
    model.save_pretrained(initial)
    model.load_adapter(initial, adapter_name="reference", is_trainable=False)
    model.set_adapter("default")
    return model, tokenizer, model_path, initial


def test_completion_scores_include_eos_but_not_prompt_or_padding(tmp_path):
    model, tokenizer, _, _ = tiny_actor(tmp_path)
    model.eval()
    rows = [
        {
            "prompt_ids": tokenizer.encode("Number:"),
            "completion_ids": tokenizer.encode(text) + [tokenizer.eos_token_id],
        }
        for text in ("1", "123")
    ]
    batched = completion_log_probs(model, rows, tokenizer.pad_token_id)
    separate = torch.cat(
        [completion_log_probs(model, [row], tokenizer.pad_token_id) for row in rows]
    )
    torch.testing.assert_close(batched, separate)
    for row, actual in zip(rows, batched):
        ids = torch.tensor([row["prompt_ids"] + row["completion_ids"]])
        logits = model(input_ids=ids).logits[:, :-1].log_softmax(-1)
        start = len(row["prompt_ids"]) - 1
        expected = logits[0, start:].gather(-1, ids[0, start + 1 :, None]).sum()
        torch.testing.assert_close(actual, expected)


def test_actor_changes_while_reference_stays_frozen(tmp_path):
    model, tokenizer, _, _ = tiny_actor(tmp_path)
    before = {name: p.detach().clone() for name, p in model.named_parameters() if "lora_" in name}
    optimizer = torch.optim.AdamW([p for p in model.parameters() if p.requires_grad], lr=0.01)
    rows = [
        {
            "prompt_ids": tokenizer.encode("Number:"),
            "completion_ids": tokenizer.encode(text) + [tokenizer.eos_token_id],
        }
        for text in ("1", "2")
    ]
    result = policy_update(
        model,
        optimizer,
        [{"rows": rows, "advantages": np.array([1.0, -1.0])}],
        tokenizer.pad_token_id,
        microbatch_size=1,
        reference_kl_beta=0.02,
    )
    assert np.isfinite(result["policy_gradient_loss"])
    after = dict(model.named_parameters())
    assert all(
        torch.equal(value, after[name]) for name, value in before.items() if ".reference." in name
    )
    assert any(
        not torch.equal(value, after[name]) for name, value in before.items() if ".default." in name
    )


def test_reference_penalty_has_the_correct_score_function_gradient(monkeypatch):
    import stochastic_llm.rl as rl

    class CategoricalPolicy(torch.nn.Module):
        def __init__(self):
            super().__init__()
            self.logits = torch.nn.Parameter(torch.zeros(3, dtype=torch.float64))
            self.register_buffer("reference", torch.tensor([0.7, 0.2, 0.1], dtype=torch.float64))
            self.adapter = "default"
            self.device = torch.device("cpu")

        def set_adapter(self, adapter):
            self.adapter = adapter

    def scores(model, rows, pad_token_id):
        log_probs = (
            model.reference.log() if model.adapter == "reference" else model.logits.log_softmax(0)
        )
        return log_probs[torch.tensor([row["action"] for row in rows])]

    monkeypatch.setattr(rl, "completion_log_probs", scores)
    model = CategoricalPolicy()
    p = model.logits.softmax(0)
    exact = (p * (p.log() - model.reference.log())).sum()
    gradient = torch.autograd.grad(exact, model.logits)[0]
    assert gradient.norm() < 1  # The test update does not invoke norm clipping.
    before = model.logits.detach().clone()
    optimizer = torch.optim.SGD(model.parameters(), lr=0.1)
    groups = [{"rows": [{"action": i} for i in range(3)], "advantages": np.zeros(3)}]
    policy_update(model, optimizer, groups, 0, microbatch_size=1, reference_kl_beta=1.0)
    torch.testing.assert_close(model.logits, before - 0.1 * gradient)


@pytest.mark.integration
def test_real_rl_training_checkpoint_resume_and_adapter_export(tmp_path):
    torch.set_num_threads(2)
    _, _, model_path, initial = tiny_actor(tmp_path)
    data = tmp_path / "data"
    generate_data(
        {
            "output_dir": str(data),
            "seed": 4,
            "decimals": 5,
            "train_configs": 6,
            "samples_per_config": 1,
            "validation_configs": 2,
            "validation_samples_per_config": 1,
            "test_configs": 3,
        }
    )
    config = read_config("configs/batch-kl-rl-h200.yaml")
    config.update(
        {
            "model": model_path,
            "revision": "main",
            "initial_adapter": str(initial),
            "data_dir": str(data),
            "output_dir": str(tmp_path / "rl"),
            "device": "cpu",
            "precision": "fp32",
            "attention": "eager",
            "group_size": 4,
            "prompts_per_update": 1,
            "rollout_batch_size": 2,
            "microbatch_size": 2,
            "max_sequence_length": 256,
            "max_new_tokens": 4,
            "max_steps": 2,
            "reward_bins": 4,
            "validation_cases": 2,
            "validation_samples_per_case": 4,
            "validation_steps": 1,
            "save_steps": 1,
            "wandb": {"enabled": False},
        }
    )
    result = train_rl(config)
    assert result["generated_training_responses"] == 8
    adapter = tmp_path / "rl/adapter/adapter_model.safetensors"
    uninterrupted = load_file(adapter)
    checkpoint = tmp_path / "rl/checkpoint-1"
    train_rl(config, resume_from_checkpoint=str(checkpoint))
    resumed = load_file(adapter)
    assert uninterrupted.keys() == resumed.keys()
    for name in uninterrupted:
        torch.testing.assert_close(uninterrupted[name], resumed[name], atol=1e-7, rtol=1e-6)
    metadata = json.loads((tmp_path / "rl/run_metadata.json").read_text())
    assert metadata["supervised_answer_labels_used"] is False
    assert (tmp_path / "rl/rollouts-00002.jsonl").is_file()
    reloaded = PeftModel.from_pretrained(
        AutoModelForCausalLM.from_pretrained(model_path), adapter.parent
    )
    assert reloaded.active_adapter == "default"
