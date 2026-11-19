"""Two CPU processes: global credit, gradient equivalence, checkpoint and RNG."""

import random
import tempfile
import unittest
from pathlib import Path
from types import SimpleNamespace

import numpy as np
import torch
import torch.distributed as dist
import torch.multiprocessing as mp
from ddp_passage_rl import (
    SequencePolicy,
    distributed_update,
    global_credit,
    restore_rank_rng,
    save_checkpoint,
)
from passage_reward import PassageDistribution, quality_advantages
from passage_rl import update
from peft import LoraConfig, get_peft_model
from torch.nn.parallel import DistributedDataParallel as DDP
from transformers import Qwen2Config, Qwen2ForCausalLM


def tiny_actor():
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
    return get_peft_model(
        base,
        LoraConfig(
            task_type="CAUSAL_LM",
            r=2,
            lora_alpha=4,
            lora_dropout=0.0,
            target_modules=["q_proj", "v_proj"],
        ),
    ).eval()


def example_rows():
    return [
        {"prompt_ids": [1, 4], "completion_ids": [5, 6, 2], "passage": "a b a b"},
        {"prompt_ids": [1, 7], "completion_ids": [8, 2], "passage": "a c b"},
        {"prompt_ids": [1, 4, 9], "completion_ids": [5, 9, 2], "passage": "a b c"},
        {"prompt_ids": [1, 7], "completion_ids": [2], "passage": ""},
    ]


def distributed_worker(rank, folder):
    torch.set_num_threads(1)
    device = torch.device("cpu")
    dist.init_process_group("gloo", init_method=f"file://{folder}/store", rank=rank, world_size=2)
    try:
        actor = tiny_actor()
        policy = DDP(SequencePolicy(actor), broadcast_buffers=False)
        optimizer = torch.optim.SGD(
            [p for p in actor.parameters() if p.requires_grad], lr=0.02, momentum=0.9
        )
        scheduler = torch.optim.lr_scheduler.LambdaLR(optimizer, lambda _: 1.0)
        rows = example_rows()[rank * 2 : (rank + 1) * 2]
        quality = np.array([1.0, 0.25, 0.4, 0.0])[rank * 2 : (rank + 1) * 2]
        target = PassageDistribution(["a b a c b", "a b a b"])
        credit, metrics = global_credit(rows, quality, target, 0.1, device)
        distributed_update(policy, optimizer, rows, credit, 0, 1, 0.002)
        first = {
            name: p.detach().clone() for name, p in actor.named_parameters() if p.requires_grad
        }
        torch.manual_seed(44 + rank)
        random.seed(44 + rank)
        tokenizer = SimpleNamespace(save_pretrained=lambda path: (path / "tokenizer-test").touch())
        path = save_checkpoint(
            policy,
            tokenizer,
            optimizer,
            scheduler,
            random.Random(12),
            1,
            Path(folder),
            "config-hash",
            {"source": "hash"},
            device,
        )
        expected_random = (torch.rand(3), random.random())
        state = torch.load(path / "state.pt", map_location="cpu", weights_only=False)
        restore_rank_rng(state["rank_rng"][rank], device)
        torch.testing.assert_close(expected_random[0], torch.rand(3))
        assert expected_random[1] == random.random()
        # Ensure the next optimizer update is reproducible after adapter/state restore.
        distributed_update(policy, optimizer, rows, credit * 0.7, 0, 1, 0.002)
        expected_second = {
            name: p.detach().clone() for name, p in actor.named_parameters() if p.requires_grad
        }
        from peft.utils.save_and_load import load_peft_weights, set_peft_model_state_dict

        restored = tiny_actor()
        set_peft_model_state_dict(
            restored, load_peft_weights(str(path / "adapter")), adapter_name="default"
        )
        restored_policy = DDP(SequencePolicy(restored), broadcast_buffers=False)
        restored_optimizer = torch.optim.SGD(
            [p for p in restored.parameters() if p.requires_grad], lr=0.02, momentum=0.9
        )
        restored_optimizer.load_state_dict(state["optimizer"])
        distributed_update(restored_policy, restored_optimizer, rows, credit * 0.7, 0, 1, 0.002)
        for name, p in restored.named_parameters():
            if p.requires_grad:
                torch.testing.assert_close(p, expected_second[name], rtol=1e-6, atol=1e-8)
        torch.save(
            {"parameters": first, "credits": credit, "metrics": metrics, "resume_checked": True},
            Path(folder) / f"rank-{rank}.pt",
        )
    finally:
        dist.destroy_process_group()


class DistributedTests(unittest.TestCase):
    def test_two_ranks_equal_one_global_update_and_resume(self):
        torch.set_num_threads(1)
        actor = tiny_actor()
        rows = example_rows()
        target = PassageDistribution(["a b a c b", "a b a b"])
        reward = target.credit([r["passage"] for r in rows])
        credits = quality_advantages([1.0, 0.25, 0.4, 0.0]) + 0.1 * reward["advantages"]
        optimizer = torch.optim.SGD(
            [p for p in actor.parameters() if p.requires_grad], lr=0.02, momentum=0.9
        )
        update(
            actor,
            optimizer,
            rows,
            credits,
            SimpleNamespace(pad_token_id=0),
            {"gradient_microbatch_size": 1, "reference_kl_beta": 0.002},
        )
        expected = {
            name: p.detach().clone() for name, p in actor.named_parameters() if p.requires_grad
        }
        with tempfile.TemporaryDirectory() as temporary:
            mp.spawn(distributed_worker, args=(temporary,), nprocs=2, join=True)
            for rank in range(2):
                result = torch.load(Path(temporary) / f"rank-{rank}.pt", weights_only=True)
                np.testing.assert_allclose(
                    result["credits"], credits[rank * 2 : (rank + 1) * 2], rtol=1e-6, atol=1e-7
                )
                self.assertEqual(result["metrics"]["global_group_size"], 4)
                self.assertTrue(result["resume_checked"])
                for name, value in result["parameters"].items():
                    torch.testing.assert_close(value, expected[name], rtol=1e-5, atol=1e-7)


if __name__ == "__main__":
    unittest.main()
