"""Exact gradient and real tiny-model checks for the new passage objective."""

import itertools
import tempfile
import unittest
from pathlib import Path
from types import SimpleNamespace

import numpy as np
import torch
from passage_reward import PassageDistribution, quality_advantages
from passage_rl import checkpoint, update
from peft import LoraConfig, get_peft_model
from transformers import Qwen2Config, Qwen2ForCausalLM

from stochastic_llm.rl import completion_log_probs


class PassageRewardTests(unittest.TestCase):
    def test_leave_one_passage_out_gradient_with_variable_lengths(self):
        target = PassageDistribution(["a b a c b", "a b a b"], 1.0)
        choices = ["a b", "a c b"]
        logits = torch.tensor([0.3, -0.2], dtype=torch.float64, requires_grad=True)
        probabilities = logits.softmax(0)
        objective = torch.tensor(0.0, dtype=torch.float64)
        surrogate = torch.tensor(0.0, dtype=torch.float64)
        for indices in itertools.product(range(2), repeat=2):
            selected = [choices[i] for i in indices]
            probability = probabilities[list(indices)].prod()
            result = target.credit(selected)
            objective = objective + probability * -result["kl"]
            # Averaging N-scaled leave-one-out credit preserves the group gradient.
            surrogate = (
                surrogate
                + probability.detach()
                * sum(
                    result["advantages"][i] * probabilities[index].log()
                    for i, index in enumerate(indices)
                )
                / 2
            )
        exact = torch.autograd.grad(objective, logits, retain_graph=True)[0]
        estimated = torch.autograd.grad(surrogate, logits)[0]
        torch.testing.assert_close(exact, estimated, rtol=1e-10, atol=1e-10)

    def test_quality_baseline_does_not_erase_common_reward(self):
        # Individual rewards get their own baseline; group KL uses separate credit.
        np.testing.assert_allclose(quality_advantages([0.2, 0.8]), [-0.6, 0.6])
        target = PassageDistribution(["a b a b", "a c b"], 1.0)
        result = target.credit(["a b", "a b"])
        self.assertTrue(np.any(np.abs(result["advantages"]) > 1e-8))

    def test_empty_and_unknown_passages_not_dropped(self):
        target = PassageDistribution(["a b a b"], 1.0)
        counts = target.counts(["", "z q"])
        np.testing.assert_array_equal(counts[:, -1], [1, 1])
        self.assertTrue(np.isfinite(target.credit(["", "z q"])["kl"]))

    def test_real_update_changes_only_lora_and_checkpoint_exports(self):
        torch.manual_seed(123)
        base = Qwen2ForCausalLM(
            Qwen2Config(
                vocab_size=24,
                hidden_size=16,
                intermediate_size=32,
                num_hidden_layers=1,
                num_attention_heads=2,
                num_key_value_heads=1,
                max_position_embeddings=64,
                attention_dropout=0.0,
                pad_token_id=0,
                eos_token_id=2,
            )
        )
        actor = get_peft_model(
            base,
            LoraConfig(
                task_type="CAUSAL_LM",
                r=2,
                lora_alpha=4,
                lora_dropout=0.0,
                target_modules=["q_proj", "v_proj"],
            ),
        )
        actor.eval()
        rows = [
            {"prompt_ids": [1, 4], "completion_ids": [5, 6, 2]},
            {"prompt_ids": [1, 7], "completion_ids": [8, 2]},
        ]
        frozen = {
            name: p.detach().clone() for name, p in actor.named_parameters() if not p.requires_grad
        }
        with torch.no_grad(), actor.disable_adapter():
            reference_before = completion_log_probs(actor, rows, 0).clone()
        optimizer = torch.optim.AdamW([p for p in actor.parameters() if p.requires_grad], lr=0.01)
        metrics = update(
            actor,
            optimizer,
            rows,
            np.array([1.0, -1.0]),
            SimpleNamespace(pad_token_id=0),
            {"gradient_microbatch_size": 1, "reference_kl_beta": 0.002},
        )
        self.assertGreater(metrics["gradient_norm"], 0)
        for name, p in actor.named_parameters():
            if name in frozen:
                torch.testing.assert_close(p, frozen[name], rtol=0, atol=0)
        with torch.no_grad(), actor.disable_adapter():
            torch.testing.assert_close(reference_before, completion_log_probs(actor, rows, 0))
        self.assertTrue(
            any(
                torch.count_nonzero(p) > 0
                for name, p in actor.named_parameters()
                if "lora_B" in name
            )
        )
        # Checkpoint API expects CUDA state, but its serializer is independently checked on CPU.
        import random

        scheduler = torch.optim.lr_scheduler.LambdaLR(optimizer, lambda _: 1.0)
        tokenizer = SimpleNamespace(save_pretrained=lambda path: (path / "tokenizer-test").touch())
        with tempfile.TemporaryDirectory() as temporary:
            checkpoint(actor, tokenizer, optimizer, scheduler, random.Random(1), 1, Path(temporary))
            self.assertTrue((Path(temporary) / "checkpoint-1/complete.json").exists())
            self.assertTrue(
                (Path(temporary) / "checkpoint-1/adapter/adapter_model.safetensors").exists()
            )


if __name__ == "__main__":
    unittest.main()
