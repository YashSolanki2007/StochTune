"""A real, entirely offline CPU training/evaluation integration run."""

from pathlib import Path

import torch
from tokenizers import Tokenizer, decoders, models
from transformers import PreTrainedTokenizerFast, Qwen2Config, Qwen2ForCausalLM

from .data import generate_data
from .evaluate import compare_runs, evaluate
from .io import require_empty_directory, write_json
from .train import train


def create_tiny_model(output):
    output = Path(output)
    output.mkdir(parents=True, exist_ok=True)
    tokens = ["<unk>", "<eos>", "<pad>"] + [chr(i) for i in range(9, 127)]
    backend = Tokenizer(
        models.BPE(vocab={t: i for i, t in enumerate(tokens)}, merges=[], unk_token="<unk>")
    )
    backend.decoder = decoders.Fuse()
    tokenizer = PreTrainedTokenizerFast(
        tokenizer_object=backend,
        eos_token="<eos>",
        pad_token="<pad>",
        unk_token="<unk>",
        model_max_length=256,
    )
    tokenizer.save_pretrained(output)
    torch.manual_seed(0)
    model = Qwen2ForCausalLM(
        Qwen2Config(
            vocab_size=len(tokenizer),
            hidden_size=32,
            intermediate_size=64,
            num_hidden_layers=2,
            num_attention_heads=4,
            num_key_value_heads=2,
            max_position_embeddings=256,
            eos_token_id=tokenizer.eos_token_id,
            pad_token_id=tokenizer.pad_token_id,
            bos_token_id=None,
            tie_word_embeddings=True,
        )
    )
    model.save_pretrained(output)
    return str(output.resolve())


def smoke(output, with_tracking=False):
    output = require_empty_directory(output).resolve()
    torch.set_num_threads(2)
    model = create_tiny_model(output / "tiny-model")
    data_dir = str(output / "data")
    generate_data(
        {
            "output_dir": data_dir,
            "seed": 7,
            "decimals": 5,
            "train_configs": 12,
            "samples_per_config": 2,
            "validation_configs": 4,
            "validation_samples_per_config": 2,
            "test_configs": 3,
        }
    )
    config = {
        "model": model,
        "revision": "main",
        "data_dir": data_dir,
        "method": "sft",
        "output_dir": str(output / "sft"),
        "seed": 42,
        "device": "cpu",
        "precision": "fp32",
        "attention": "eager",
        "max_sequence_length": 256,
        "max_steps": 2,
        "microbatch_size": 2,
        "gradient_accumulation_steps": 1,
        "eval_batch_size": 2,
        "learning_rate": 0.0002,
        "weight_decay": 0.01,
        "warmup_ratio": 0.03,
        "lora_rank": 4,
        "lora_alpha": 8,
        "lora_dropout": 0.05,
        "target_modules": ["q_proj", "k_proj", "v_proj", "o_proj"],
        "logging_steps": 1,
        "eval_steps": 1,
        "save_steps": 1,
        "save_total_limit": 2,
        "dataloader_num_workers": 0,
    }
    if with_tracking:
        config["wandb"] = {
            "enabled": True,
            "mode": "offline",
            "entity": "local-test",
            "project": "stochastic-test",
            "directory": str(output / "wandb"),
        }
    train(config)
    train(config, resume_from_checkpoint=str(output / "sft" / "checkpoint-1"))
    eval_config = {
        "model": model,
        "revision": "main",
        "cases": str(output / "data" / "test.jsonl"),
        "output_dir": str(output / "eval-base"),
        "adapter": None,
        "seed": 11,
        "device": "cpu",
        "precision": "fp32",
        "attention": "eager",
        "samples_per_prompt": 3,
        "batch_size": 2,
        "max_new_tokens": 12,
        "decimals": 5,
    }
    if with_tracking:
        eval_config["wandb"] = config["wandb"]
    evaluate(eval_config)
    evaluate(eval_config, resume=True)
    adapted = eval_config | {
        "adapter": str(output / "sft" / "adapter"),
        "output_dir": str(output / "eval-adapter"),
    }
    evaluate(adapted)
    compare_runs([output / "eval-base", output / "eval-adapter"], output / "comparison.json")
    write_json(
        output / "smoke-result.json",
        {
            "passed": True,
            "checked": [
                "offline tiny Qwen2 SFT",
                "checkpoint restore",
                "base and adapter generation",
                "evaluation resume",
                "comparison export",
            ],
            "note": "Random tiny-model outputs do not demonstrate numeric sampling quality.",
        },
    )
    return {"passed": True, "output_dir": str(output)}
