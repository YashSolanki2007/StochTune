"""CPU checks for preference direction, ties, saved reward heads, and DDP credit."""

import json
import tempfile
import unittest

import numpy as np
import torch
import torch.distributed as dist
import torch.multiprocessing as mp
from budget_run import choose_scorer
from common import (
    ROOT,
    PassageDistribution,
    bt_loss,
    distribution_diagnostics,
    messages,
    prompt_id,
    quality_advantages,
    quality_with_truncation,
    sequence_scores,
)
from peft import LoraConfig, PeftModel, get_peft_model
from train_policy import global_credit, terminal_ids, training_assignment, trim_completion
from transformers import Qwen2Config, Qwen2ForSequenceClassification
from transformers.tokenization_utils_base import BatchEncoding
from validation import require_reward_audit, validate_reward


def tiny_reward():
    torch.manual_seed(11)
    return Qwen2ForSequenceClassification(
        Qwen2Config(
            vocab_size=32,
            hidden_size=16,
            intermediate_size=32,
            num_hidden_layers=1,
            num_attention_heads=2,
            num_key_value_heads=1,
            num_labels=1,
            pad_token_id=0,
        )
    )


def credit_worker(rank, folder):
    torch.set_num_threads(1)
    dist.init_process_group("gloo", init_method=f"file://{folder}/store", rank=rank, world_size=2)
    try:
        passages = ["a b", "b c a", "z", ""]
        quality = np.array([0.1, 0.8, 0.7, 0.3])
        local = [dict(passage=p) for p in passages[rank * 2 : rank * 2 + 2]]
        target = PassageDistribution(["a a b c"], order=1)
        for weight in (0.0, 0.1):
            credit, metrics = global_credit(
                local, quality[rank * 2 : rank * 2 + 2], target, weight, torch.device("cpu")
            )
            expected = quality_advantages(quality) + weight * target.credit(passages)["advantages"]
            np.testing.assert_allclose(credit.numpy(), expected[rank * 2 : rank * 2 + 2], atol=1e-7)
            assert metrics["unigram_kl"] == target.credit(passages)["kl"]
    finally:
        dist.destroy_process_group()


class HelpSteerTests(unittest.TestCase):
    def test_preference_sign_and_tie_loss(self):
        a, b = torch.tensor([0.0]), torch.tensor([2.0])
        self.assertLess(bt_loss(a, b, torch.tensor([1])), bt_loss(a, b, torch.tensor([-1])))
        self.assertLess(bt_loss(a, a, torch.tensor([0])), bt_loss(a, b, torch.tensor([0])))

    def test_bradley_terry_gradient(self):
        scores = torch.tensor([0.0, 0.0], requires_grad=True)
        bt_loss(scores[:1], scores[1:], torch.tensor([1])).backward()
        self.assertGreater(scores.grad[0], 0)
        self.assertLess(scores.grad[1], 0)

    def test_multiturn_context_and_canonical_id(self):
        conversation = messages("Hi\n<extra_id_1>Assistant\nHello\n<extra_id_1>User\nExplain.")
        self.assertEqual([m["role"] for m in conversation], ["user", "assistant", "user"])
        self.assertEqual(prompt_id(" hello\nworld "), prompt_id("hello world"))
        self.assertRaises(ValueError, messages, "Hi<extra_id_1>Assistant\nHello")

    def test_reward_head_trains_and_survives_export(self):
        model = get_peft_model(
            tiny_reward(),
            LoraConfig(
                task_type="SEQ_CLS",
                r=2,
                lora_alpha=4,
                lora_dropout=0,
                target_modules=["q_proj", "v_proj"],
                modules_to_save=["score"],
            ),
        ).eval()
        frozen = {k: p.detach().clone() for k, p in model.named_parameters() if not p.requires_grad}
        inputs = torch.tensor([[1, 3, 2, 0], [1, 4, 5, 2]])
        masks = inputs != 0
        optimizer = torch.optim.AdamW([p for p in model.parameters() if p.requires_grad], lr=0.01)
        scores = model(input_ids=inputs, attention_mask=masks).logits[:, 0]
        bt_loss(scores[:1], scores[1:], torch.tensor([1])).backward()
        optimizer.step()
        for name, p in model.named_parameters():
            if name in frozen:
                torch.testing.assert_close(p, frozen[name], rtol=0, atol=0)
        with torch.no_grad():
            expected = model(input_ids=inputs, attention_mask=masks).logits
        with tempfile.TemporaryDirectory() as folder:
            model.save_pretrained(folder)
            loaded = PeftModel.from_pretrained(tiny_reward(), folder).eval()
            with torch.no_grad():
                actual = loaded(input_ids=inputs, attention_mask=masks).logits
            torch.testing.assert_close(expected, actual)

    def test_two_rank_credit_both_arms(self):
        with tempfile.TemporaryDirectory() as folder:
            mp.spawn(credit_worker, args=(folder,), nprocs=2, join=True)

    def test_prepared_splits_are_disjoint(self):
        manifest_path = ROOT / "data/manifest.json"
        if not manifest_path.exists():
            self.skipTest("Run prepare.py for the full data-integrity check")
        manifest = json.loads(manifest_path.read_text())
        groups = list(manifest["prompt_ids"].values())
        for i, group in enumerate(groups):
            for other in groups[i + 1 :]:
                self.assertFalse(set(group).intersection(other))

    def test_reference_stays_broad_and_excludes_all_evaluation_prompts(self):
        if not (ROOT / "data/manifest.json").exists():
            self.skipTest("Run v2 prepare.py before the data-integrity checks")
        manifest = json.loads((ROOT / "data/manifest.json").read_text())
        references = set(manifest["reference_prompt_ids"])
        self.assertEqual(manifest["counts"]["policy_train"], 128)
        self.assertGreater(len(references), 1000)
        for name in ("reward_validation", "policy_validation", "policy_test"):
            self.assertFalse(references.intersection(manifest["prompt_ids"][name]))
        self.assertFalse(references.intersection(manifest["long_reward_audit_ids"]))

    def test_truncation_preserves_quality_and_logs_a_separate_penalty(self):
        quality, penalty = quality_with_truncation([0.8, 0.8, 0.05], [True, False, False], 0.1)
        np.testing.assert_allclose(quality, [0.8, 0.7, -0.05])
        np.testing.assert_allclose(penalty, [0, 0.1, 0.1])
        self.assertRaises(ValueError, quality_with_truncation, [0.5], [False], -1)

    def test_sixteen_distinct_prompts_in_global_batch(self):
        selected = [{"id": str(i)} for i in range(16)]
        cfg = {"prompts_per_group": 16, "global_group_size": 16}
        a, na = training_assignment(selected, cfg, 0, 2)
        b, nb = training_assignment(selected, cfg, 1, 2)
        self.assertEqual((len(a), len(b), na, nb), (8, 8, 1, 1))
        self.assertFalse({r["id"] for r in a}.intersection(r["id"] for r in b))
        self.assertRaises(ValueError, training_assignment, selected[:8] * 2, cfg, 0, 2)

    def test_both_checkpoint_terminal_tokens_stop_the_response(self):
        from types import SimpleNamespace

        actor = SimpleNamespace(generation_config=SimpleNamespace(eos_token_id=[2, 3]))
        tokenizer = SimpleNamespace(eos_token_id=2)
        stops = terminal_ids(actor, tokenizer)
        self.assertEqual(trim_completion([5, 3, 0, 0], stops), ([5, 3], True))
        self.assertEqual(trim_completion([5, 2, 0], stops), ([5, 2], True))
        self.assertEqual(trim_completion([5, 6, 7], stops), ([5, 6, 7], False))

    def test_reward_length_limit_is_configurable_without_truncating(self):
        class Tokenizer:
            def pad(self, batch, **kwargs):
                return BatchEncoding({"input_ids": torch.tensor(batch["input_ids"])})

        class Model:
            device = torch.device("cpu")
            seen = 0

            def __call__(self, input_ids, **kwargs):
                from types import SimpleNamespace

                self.seen = input_ids.shape[1]
                return SimpleNamespace(logits=torch.ones((len(input_ids), 1)))

        model = Model()
        sequence_scores(model, Tokenizer(), [[1] * 2048], max_length=2048)
        self.assertEqual(model.seen, 2048)
        self.assertRaises(ValueError, sequence_scores, model, Tokenizer(), [[1] * 2049], 2048)

    def test_missing_mass_is_reported_without_changing_kl_objective(self):
        target = PassageDistribution(["a a b c"], order=1)
        metrics = distribution_diagnostics(target, ["a a", "z"])
        self.assertAlmostEqual(metrics["unobserved_reference_mass"], 0.5)
        self.assertAlmostEqual(metrics["unigram_kl"], target.credit(["a a", "z"])["kl"])
        self.assertGreaterEqual(metrics["distribution_js"], 0)

    @staticmethod
    def synthetic_validation():
        rows = []
        for i in range(100):
            sign = 1 if i % 2 else -1
            n = 1100 if i < 50 else 100
            rows.append(
                {
                    "preference_strength": sign,
                    "response_tokens_1": n,
                    "response_tokens_2": n + 2,
                    "ids_1": [1] * n,
                    "ids_2": [1] * (n + 2),
                    "coherence_1": 3 if sign > 0 else 4,
                    "coherence_2": 4 if sign > 0 else 3,
                }
            )
        return rows

    def test_reward_gate_rejects_length_only_scorer(self):
        rows = self.synthetic_validation()
        report = validate_reward(rows, np.zeros(100), np.ones(100), 1)
        self.assertFalse(report["passed"])
        self.assertFalse(report["checks"]["beats_longer_answer_baseline"])

    def test_reward_gate_accepts_informative_scorer_and_binds_data(self):
        rows = self.synthetic_validation()
        report = validate_reward(
            rows, np.zeros(100), np.array([r["preference_strength"] for r in rows]), 1
        )
        self.assertTrue(report["passed"])
        report["validation_data_sha256"] = "correct"
        require_reward_audit(report, "correct")
        self.assertRaises(RuntimeError, require_reward_audit, report, "changed")
        self.assertRaises(RuntimeError, require_reward_audit, {"passed": True}, "correct")

    def test_budget_selector_never_uses_a_failed_scorer(self):
        self.assertIsNone(
            choose_scorer({"legacy": {"passed": False}, "skywork": {"passed": False}})
        )
        self.assertEqual(
            choose_scorer(
                {
                    "legacy": {"passed": False, "non_tie_accuracy": 0.99},
                    "skywork": {"passed": True, "non_tie_accuracy": 0.8},
                }
            ),
            "skywork",
        )
        self.assertEqual(
            choose_scorer(
                {
                    "legacy": {"passed": True, "non_tie_accuracy": 0.8},
                    "skywork": {"passed": True, "non_tie_accuracy": 0.8},
                }
            ),
            "legacy",
        )


if __name__ == "__main__":
    unittest.main()
