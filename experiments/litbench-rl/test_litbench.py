"""CPU data integrity, real tiny-model DDP updates, and reward-certificate checks."""

import copy
import hashlib
import json
import tempfile
import unittest

import numpy as np
import torch
import torch.distributed as dist
import torch.multiprocessing as mp
from common import ROOT, bt_loss, make_distribution, quality_advantages
from peft import LoraConfig, PeftModel, get_peft_model
from preflight import preflight
from prepare import fingerprint, split_components
from torch.nn.parallel import DistributedDataParallel as DDP
from train_policy import global_credit, training_assignment
from train_reward import RewardPolicy, update
from transformers import Qwen2Config, Qwen2ForSequenceClassification
from transformers.tokenization_utils_base import BatchEncoding
from validation import require_reward_audit, validate_reward


class Padding:
    def pad(self, batch, **kwargs):
        sequences = batch["input_ids"]
        width = max(map(len, sequences))
        return BatchEncoding(
            {
                "input_ids": torch.tensor([s + [0] * (width - len(s)) for s in sequences]),
                "attention_mask": torch.tensor(
                    [[1] * len(s) + [0] * (width - len(s)) for s in sequences]
                ),
            }
        )


def tiny_base():
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
    ).eval()


def tiny_reward():
    return get_peft_model(
        tiny_base(),
        LoraConfig(
            task_type="SEQ_CLS",
            r=2,
            lora_alpha=4,
            lora_dropout=0,
            target_modules=["q_proj", "k_proj", "v_proj", "o_proj"],
            modules_to_save=["score"],
        ),
    ).eval()


def worker(rank, folder):
    torch.set_num_threads(1)
    dist.init_process_group("gloo", init_method=f"file://{folder}/store", rank=rank, world_size=2)
    try:
        rows = [
            {
                "ids_1": [1, 2 + i, 3],
                "ids_2": [1, 8 + i, 4, 5],
                "preference_strength": 1 if i % 2 else -1,
            }
            for i in range(4)
        ]
        cfg = {"reward_pair_microbatch_size": 1, "reward_max_sequence_length": 16}
        model = tiny_reward()
        expected = copy.deepcopy(model)
        saved = {k: p.detach().clone() for k, p in model.named_parameters()}
        policy = DDP(RewardPolicy(model))
        optimizer = torch.optim.SGD([p for p in model.parameters() if p.requires_grad], lr=0.02)
        update(policy, optimizer, Padding(), rows[rank::2], cfg)
        seqs = [r["ids_1"] for r in rows] + [r["ids_2"] for r in rows]
        scores = RewardPolicy(expected)(seqs, Padding(), 16)
        loss = bt_loss(
            scores[:4], scores[4:], torch.tensor([r["preference_strength"] for r in rows])
        )
        loss.backward()
        torch.nn.utils.clip_grad_norm_([p for p in expected.parameters() if p.requires_grad], 1)
        torch.optim.SGD([p for p in expected.parameters() if p.requires_grad], lr=0.02).step()
        head_changed = False
        for (name, actual), wanted in zip(model.named_parameters(), expected.parameters()):
            torch.testing.assert_close(actual, wanted, atol=1e-7, rtol=1e-6)
            if not actual.requires_grad:
                torch.testing.assert_close(actual, saved[name], atol=0, rtol=0)
            if "score.modules_to_save" in name:
                head_changed |= not torch.equal(actual, saved[name])
        assert head_changed
        if rank == 0:
            model.save_pretrained(folder + "/adapter")
            loaded = PeftModel.from_pretrained(tiny_base(), folder + "/adapter").eval()
            with torch.inference_mode():
                torch.testing.assert_close(
                    RewardPolicy(model)(seqs, Padding(), 16),
                    RewardPolicy(loaded)(seqs, Padding(), 16),
                )
        passages = ["a a", "b c", "d a", "c c c"]
        qualities = np.array([0.1, 0.7, 0.4, 0.9])
        target = make_distribution(
            {
                "word_distribution": "frequency-capped-other",
                "vocabulary_min_count": 1,
                "vocabulary_max_words": 10,
            },
            ["a a b b c c"],
        )
        local = [{"passage": p} for p in passages[rank * 2 : rank * 2 + 2]]
        credit, metrics = global_credit(
            local, qualities[rank * 2 : rank * 2 + 2], target, 0.1, torch.device("cpu")
        )
        expected = quality_advantages(qualities) + 0.1 * target.credit(passages)["advantages"]
        np.testing.assert_allclose(credit.numpy(), expected[rank * 2 : rank * 2 + 2], atol=1e-7)
        assert metrics["quality_advantage_rms"] > 0
        assert metrics["distribution_advantage_rms"] > 0
    finally:
        dist.destroy_process_group()


class LitBenchTests(unittest.TestCase):
    def test_real_two_rank_update_and_both_policy_credits(self):
        with tempfile.TemporaryDirectory() as folder:
            mp.spawn(worker, args=(folder,), nprocs=2, join=True)

    def test_shared_story_connects_different_prompts(self):
        rows = [
            {"prompt": "First", "chosen_story": "shared &amp; text", "rejected_story": "a"},
            {"prompt": "Second", "chosen_story": " SHARED & text ", "rejected_story": "b"},
        ]
        split = split_components(rows)
        self.assertEqual(split[fingerprint("First")], split[fingerprint("Second")])
        self.assertEqual(split, split_components(rows[::-1]))

    @staticmethod
    def synthetic():
        return [
            {
                "preference_strength": 1 if i % 2 else -1,
                "response_tokens_1": 100,
                "response_tokens_2": 110,
            }
            for i in range(256)
        ]

    def test_length_only_and_constant_rewards_fail(self):
        rows = self.synthetic()
        self.assertFalse(validate_reward(rows, np.zeros(256), np.ones(256))["passed"])
        self.assertFalse(validate_reward(rows, np.zeros(256), np.zeros(256))["passed"])

    def test_informative_scorer_passes_and_hash_is_required(self):
        rows = self.synthetic()
        report = validate_reward(rows, np.zeros(256), [r["preference_strength"] for r in rows])
        self.assertTrue(report["passed"])
        self.assertEqual(json.loads(json.dumps(report)), report)
        self.assertTrue(all(type(value) is bool for value in report["checks"].values()))
        report["validation_data_sha256"] = "audit"
        require_reward_audit(report, "audit")
        self.assertRaises(RuntimeError, require_reward_audit, report, "selection")

    def test_real_data_passes_and_pmfs_match(self):
        report = preflight()
        self.assertTrue(report["passed"])
        rows = [json.loads(s) for s in (ROOT / "data/reference.jsonl").read_text().splitlines()]
        cfg = json.loads((ROOT / "config.json").read_text())
        target = make_distribution(cfg, [r["response"] for r in rows])
        categories = json.loads((ROOT / "data/vocabulary.json").read_text())["categories"]
        np.testing.assert_allclose(target.probabilities, [r["probability"] for r in categories])

    def test_certificate_uses_audit_not_model_selection_pairs(self):
        from train_policy import _source

        self.assertIn('ROOT / "data/reward_audit.jsonl"', _source)
        audit = hashlib.sha256((ROOT / "data/reward_audit.jsonl").read_bytes()).hexdigest()
        selection = hashlib.sha256((ROOT / "data/reward_validation.jsonl").read_bytes()).hexdigest()
        self.assertNotEqual(audit, selection)

    def test_four_samples_per_prompt_split_across_two_ranks(self):
        cfg = json.loads((ROOT / "config.json").read_text())
        rows = [{"id": str(i)} for i in range(4)]
        first, n = training_assignment(rows, cfg, 0, 2)
        second, m = training_assignment(rows, cfg, 1, 2)
        self.assertEqual((len(first), len(second), n, m), (2, 2, 4, 4))
        self.assertFalse({r["id"] for r in first} & {r["id"] for r in second})


if __name__ == "__main__":
    unittest.main()
