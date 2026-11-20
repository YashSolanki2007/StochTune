"""CPU checks for preference direction, ties, saved reward heads, and DDP credit."""

import json
import tempfile
import unittest
from pathlib import Path

import numpy as np
import torch
import torch.distributed as dist
import torch.multiprocessing as mp
from common import ROOT, PassageDistribution, bt_loss, messages, prompt_id, quality_advantages
from peft import LoraConfig, PeftModel, get_peft_model
from prepare_subset import prepare_subset
from train_policy import global_credit
from transformers import Qwen2Config, Qwen2ForSequenceClassification


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

    def test_subset_has_matching_pmf_and_bounded_two_arm_allocation(self):
        if not (ROOT / "data/manifest.json").exists():
            self.skipTest("Run prepare.py before staging a real subset")
        with tempfile.TemporaryDirectory() as folder:
            output = Path(folder) / "subset"
            prepare_subset(output)
            cfg = json.loads((output / "config.json").read_text())
            manifest = json.loads((output / "data/manifest.json").read_text())
            ids = set(manifest["prompt_ids"]["train"])
            refs = [
                json.loads(line)
                for line in (output / "data/reference.jsonl").read_text().splitlines()
            ]
            self.assertEqual(len(ids), 128)
            self.assertEqual(ids, {r["id"] for r in refs})
            self.assertEqual(cfg["steps"], 200)
            self.assertLessEqual(cfg["subset_allocation_seconds"], 7200)
            self.assertLess(2 * cfg["train_budget_seconds"], cfg["subset_allocation_seconds"])
            for split in ("validation", "test"):
                self.assertFalse(ids.intersection(manifest["prompt_ids"][split]))
            self.assertIn("#SBATCH --time=01:45:00", (output / "subset.sbatch").read_text())


if __name__ == "__main__":
    unittest.main()
