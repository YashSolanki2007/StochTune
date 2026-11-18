"""Explicit device selection; no implicit CPU fallback for H200 runs."""

import importlib.metadata

import torch
from transformers import AutoModelForCausalLM, AutoTokenizer


def device_and_dtype(config):
    device = config.get("device", "cuda")
    if device == "cuda" and not torch.cuda.is_available():
        raise RuntimeError("CUDA is unavailable. Use --device cpu for the tiny smoke test.")
    if device == "mps" and not torch.backends.mps.is_available():
        raise RuntimeError("Apple MPS is unavailable")
    precision = config.get("precision", "bf16")
    dtypes = {"bf16": torch.bfloat16, "fp16": torch.float16, "fp32": torch.float32}
    if precision not in dtypes:
        raise ValueError(f"Unknown precision: {precision}")
    if device == "cpu" and precision != "fp32":
        raise ValueError("Use fp32 for CPU smoke tests")
    if device == "cuda" and precision == "bf16" and not torch.cuda.is_bf16_supported():
        raise RuntimeError("This GPU does not support BF16")
    return device, dtypes[precision]


def load_model_and_tokenizer(config, training=False):
    device, dtype = device_and_dtype(config)
    kwargs = {"revision": config.get("revision", "main"), "trust_remote_code": False}
    tokenizer = AutoTokenizer.from_pretrained(config["model"], **kwargs)
    if tokenizer.eos_token_id is None:
        raise ValueError("Tokenizer must define EOS")
    if tokenizer.pad_token_id is None:
        tokenizer.pad_token = tokenizer.eos_token
    tokenizer.padding_side = "right" if training else "left"
    model = AutoModelForCausalLM.from_pretrained(
        config["model"],
        dtype=dtype,
        attn_implementation=config.get("attention", "sdpa"),
        **kwargs,
    ).to(device)
    model.config.pad_token_id = tokenizer.pad_token_id
    model.config.use_cache = not training
    return model, tokenizer


def environment_info():
    packages = {}
    for name in ("torch", "transformers", "peft", "accelerate", "numpy", "scipy", "wandb"):
        packages[name] = importlib.metadata.version(name)
    return {
        "packages": packages,
        "cuda_version": torch.version.cuda,
        "gpu": torch.cuda.get_device_name(0) if torch.cuda.is_available() else None,
    }
