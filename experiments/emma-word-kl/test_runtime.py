"""Exercise scorer decoding, full certificate export and a real two-rank policy update."""

import json
import tempfile
import unittest
from pathlib import Path
from types import SimpleNamespace

import numpy as np
import torch
import torch.distributed as dist
import torch.multiprocessing as mp
from calibrate import write_certificate
from common import (
    ROOT,
    FrozenPreferenceReward,
    config,
    make_distribution,
    read_rows,
    verify_reward_identity,
)
from ddp_passage_rl import SequencePolicy, distributed_update
from fiction_reward import FrozenRubricJudge
from peft import LoraConfig, get_peft_model
from preflight_gpu import preflight_gpu
from torch.nn.parallel import DistributedDataParallel as DDP
from train_policy import global_credit
from transformers import Qwen2Config, Qwen2ForCausalLM
from transformers.tokenization_utils_base import BatchEncoding
from validation import require_reward_audit, validate_scores


def perfect_scores(rows):
    return [
        {
            "id": r["id"],
            "quality": np.float64(0.875 if r["kind"] == "source" else 0.125),
            "scores": {
                k: np.float64(4.5 if r["kind"] == "source" else 1.5)
                for k in ("coherence", "readability", "prompt_fulfillment")
            },
            "rating_token_mass": {
                k: 1.0 for k in ("coherence", "readability", "prompt_fulfillment")
            },
        }
        for r in rows
    ]


class Tokenizer:
    def apply_chat_template(self, messages, **kwargs):
        return "sample"

    def __call__(self, prompts, **kwargs):
        return BatchEncoding({"input_ids": torch.ones(len(prompts), 2, dtype=torch.long)})


class Model:
    device = torch.device("cpu")

    def __init__(self, valid=True):
        self.valid = valid

    def __call__(self, input_ids, **kwargs):
        logits = torch.full((len(input_ids), 1, 8), -4.0)
        logits[:, :, 4 if self.valid else 0] = 20
        return SimpleNamespace(logits=logits)


def policy_worker(rank, folder):
    torch.set_num_threads(1)
    dist.init_process_group("gloo", init_method=f"file://{folder}/store", rank=rank, world_size=2)
    try:
        torch.manual_seed(11)
        actor = get_peft_model(
            Qwen2ForCausalLM(
                Qwen2Config(
                    vocab_size=32,
                    hidden_size=16,
                    intermediate_size=32,
                    num_hidden_layers=1,
                    num_attention_heads=2,
                    num_key_value_heads=1,
                    pad_token_id=0,
                )
            ),
            LoraConfig(
                task_type="CAUSAL_LM",
                r=2,
                lora_alpha=4,
                lora_dropout=0,
                target_modules=["q_proj", "k_proj", "v_proj", "o_proj"],
            ),
        ).eval()
        policy = DDP(SequencePolicy(actor))
        all_rows = [
            {
                "case_id": f"q{i // 2}",
                "prompt_ids": [1, 2],
                "completion_ids": [i + 3, 4, 2],
                "passage": ["a a", "b", "a c", "c c"][i % 4],
            }
            for i in range(8)
        ]
        rows = all_rows[rank * 4 : rank * 4 + 4]
        target = make_distribution(
            {
                "word_distribution": "frequency-capped-other",
                "vocabulary_min_count": 1,
                "vocabulary_max_words": 10,
            },
            ["a a b c c"],
        )
        quality = np.array([0.1, 0.9, 0.4, 0.7])
        credit, metrics = global_credit(rows, quality, target, 0.1, torch.device("cpu"))
        assert metrics["quality_advantage_rms"] > 0 and metrics["distribution_advantage_rms"] > 0
        before = {n: p.detach().clone() for n, p in actor.named_parameters() if p.requires_grad}
        optimizer = torch.optim.AdamW([p for p in actor.parameters() if p.requires_grad], lr=0.001)
        update = distributed_update(
            policy,
            optimizer,
            rows,
            credit,
            0,
            1,
            0.002,
            reference_kl_centering="prompt-loo",
            record_log_probs=True,
        )
        assert np.isfinite(update["gradient_norm"]) and update["gradient_norm"] > 0
        assert any(
            not torch.equal(before[n], p) for n, p in actor.named_parameters() if n in before
        )
        assert len(update["_row_metrics"]) == 4
        json.dumps({k: v for k, v in update.items() if k != "_row_metrics"}, allow_nan=False)
    finally:
        dist.destroy_process_group()


class RuntimeTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.rows = read_rows(ROOT / "data/reward_validation.jsonl")

    def test_actual_certificate_serializes_and_binds_data_scorer_and_transform(self):
        checks = validate_scores(self.rows, perfect_scores(self.rows))
        self.assertTrue(checks["passed"])
        json.dumps(checks, allow_nan=False)
        with tempfile.TemporaryDirectory() as folder:
            output = Path(folder)
            report = write_certificate(
                output, config(), checks, ROOT / "data/reward_validation.jsonl"
            )
            reread = json.loads((output / "checks.json").read_text())
            self.assertEqual(reread, report)
            require_reward_audit(reread, report["validation_data_sha256"])
            ready = {"reward_kind": "frozen-fiction-rubric", "audit_path": folder}
            verify_reward_identity(ready, reread)
            self.assertRaises(RuntimeError, require_reward_audit, reread, "changed")
            (output / "calibration.json").write_text("{}")
            self.assertRaises(RuntimeError, verify_reward_identity, ready, reread)

    def test_constant_scorer_fails(self):
        scores = perfect_scores(self.rows)
        for s in scores:
            s.update(quality=0.5, scores={k: 3.0 for k in s["scores"]})
        self.assertFalse(validate_scores(self.rows, scores)["passed"])

    def judge(self, valid=True):
        judge = FrozenRubricJudge.__new__(FrozenRubricJudge)
        judge.cfg = config()
        judge.tokenizer, judge.model, judge.digits = Tokenizer(), Model(valid), [1, 2, 3, 4, 5]
        return judge

    def test_actual_rating_decoding_normalization_and_format_rejection(self):
        result = self.judge().score(self.rows[:2])
        self.assertAlmostEqual(result[0]["quality"], 0.75, places=5)
        self.assertEqual(
            set(result[0]["scores"]), {"coherence", "readability", "prompt_fulfillment"}
        )
        json.dumps(result, allow_nan=False)
        self.assertRaises(RuntimeError, self.judge(False).score, self.rows[:1])

    def test_no_silent_truncation_and_separate_penalty(self):
        judge = self.judge()
        judge.cfg = {**judge.cfg, "reward_max_sequence_length": 1}
        self.assertRaises(ValueError, judge.score, self.rows[:1])
        reward = FrozenPreferenceReward.__new__(FrozenPreferenceReward)
        reward.cfg, reward.judge = config(), self.judge()
        data = [{**self.rows[0], "terminated": True}, {**self.rows[1], "terminated": False}]
        records, values = reward.score(data)
        np.testing.assert_allclose(values, [0.75, 0.65], atol=1e-5)
        self.assertEqual(records[1]["truncation_penalty"], 0.1)
        json.dumps(records, allow_nan=False)

    def test_complete_gpu_preflight(self):
        result = preflight_gpu()
        self.assertTrue(result["passed"])
        self.assertLess(result["maximum_cumulative_gpu_hours"], 4)

    def test_two_rank_policy_update_uses_both_rewards(self):
        with tempfile.TemporaryDirectory() as folder:
            mp.spawn(policy_worker, args=(folder,), nprocs=2, join=True)


if __name__ == "__main__":
    unittest.main()
