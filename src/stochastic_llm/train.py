"""Sampled-target supervised LoRA fine-tuning."""

import json
import signal
import time
from pathlib import Path

import numpy as np
import torch
from peft import LoraConfig, TaskType, get_peft_model
from transformers import Trainer, TrainerCallback, TrainingArguments, set_seed

from .io import file_sha256, require_empty_directory, write_json
from .modeling import device_and_dtype, environment_info, load_model_and_tokenizer
from .tracking import tracked
from .training_data import CompletionCollator, CompletionDataset


class TimingCallback(TrainerCallback):
    def __init__(self, output_dir, warmup_steps=20):
        self.output_dir = Path(output_dir)
        self.warmup_steps = warmup_steps
        self.last_end = None
        self.durations = []

    def on_step_end(self, args, state, control, **kwargs):
        if torch.cuda.is_available() and args.device.type == "cuda":
            torch.cuda.synchronize()
        now = time.perf_counter()
        if self.last_end is not None and state.global_step > self.warmup_steps:
            self.durations.append(now - self.last_end)
        self.last_end = now

    def on_train_end(self, args, state, control, **kwargs):
        mean = float(np.mean(self.durations)) if self.durations else None
        write_json(
            self.output_dir / "timing.json",
            {
                "measured_updates": len(self.durations),
                "seconds_per_update": mean,
                "projected_2000_update_minutes": mean * 2000 / 60 if mean else None,
                "note": "Warmup excluded; inter-update validation/checkpoint time is included.",
            },
        )


class GracefulStopCallback(TrainerCallback):
    def __init__(self):
        self.requested = False

    def request_stop(self, signum, frame):
        self.requested = True

    def on_step_end(self, args, state, control, **kwargs):
        if self.requested:
            control.should_save = True
            control.should_training_stop = True
            control.should_evaluate = False
        return control


@tracked("training")
def train(config, resume_from_checkpoint=None):
    config = dict(config)
    device_and_dtype(config)
    if config["method"] != "sft":
        raise ValueError("This snapshot supports SFT only")
    data_dir = Path(config["data_dir"])
    data_manifest = json.loads((data_dir / "manifest.json").read_text())
    for name, digest in data_manifest["files"].items():
        if file_sha256(data_dir / name) != digest:
            raise ValueError(f"Data checksum mismatch: {name}")
    output = Path(config["output_dir"])
    if resume_from_checkpoint:
        checkpoint = Path(resume_from_checkpoint)
        if not (checkpoint / "trainer_state.json").exists():
            raise ValueError("Resume requires a Trainer checkpoint containing trainer_state.json")
        previous = json.loads((output / "run_config.json").read_text())
        if previous != config:
            raise ValueError("Resume config differs from the original run_config.json")
    else:
        require_empty_directory(output)
    set_seed(config["seed"])
    load_config = dict(config)
    if resume_from_checkpoint:
        previous_metadata = json.loads((output / "run_metadata.json").read_text())
        if previous_metadata.get("resolved_revision"):
            load_config["revision"] = previous_metadata["resolved_revision"]
        if previous_metadata["data_manifest"] != data_manifest:
            raise ValueError("Resume data manifest differs from the original run")
    model, tokenizer = load_model_and_tokenizer(load_config, training=True)
    model = get_peft_model(
        model,
        LoraConfig(
            task_type=TaskType.CAUSAL_LM,
            revision=getattr(model.config, "_commit_hash", None),
            r=config["lora_rank"],
            lora_alpha=config["lora_alpha"],
            lora_dropout=config["lora_dropout"],
            target_modules=config["target_modules"],
            bias="none",
        ),
    )
    model.print_trainable_parameters()
    training = CompletionDataset(data_dir / "train.jsonl", tokenizer, config["max_sequence_length"])
    validation = CompletionDataset(
        data_dir / "validation.jsonl", tokenizer, config["max_sequence_length"]
    )
    trainer_class = Trainer
    use_cuda = config["device"] == "cuda"
    args = TrainingArguments(
        output_dir=str(output),
        max_steps=config["max_steps"],
        num_train_epochs=config.get("num_train_epochs", 1),
        per_device_train_batch_size=config["microbatch_size"],
        per_device_eval_batch_size=config.get("eval_batch_size", config["microbatch_size"]),
        gradient_accumulation_steps=config["gradient_accumulation_steps"],
        learning_rate=config["learning_rate"],
        weight_decay=config["weight_decay"],
        warmup_ratio=config["warmup_ratio"],
        lr_scheduler_type="cosine",
        max_grad_norm=config.get("max_grad_norm", 1.0),
        optim="adamw_torch",
        bf16=use_cuda and config["precision"] == "bf16",
        fp16=use_cuda and config["precision"] == "fp16",
        tf32=use_cuda,
        use_cpu=config["device"] == "cpu",
        logging_steps=config["logging_steps"],
        eval_strategy="steps",
        eval_steps=config["eval_steps"],
        save_strategy="steps",
        save_steps=config["save_steps"],
        save_total_limit=config.get("save_total_limit", 3),
        dataloader_num_workers=config.get("dataloader_num_workers", 0),
        dataloader_pin_memory=use_cuda,
        remove_unused_columns=False,
        gradient_checkpointing=config.get("gradient_checkpointing", False),
        gradient_checkpointing_kwargs={"use_reentrant": False},
        seed=config["seed"],
        data_seed=config["seed"],
        report_to=["wandb"] if config.get("wandb", {}).get("enabled", False) else "none",
        label_names=["labels"],
        prediction_loss_only=True,
    )
    base_config = model.get_base_model().config
    metadata = {
        "model": config["model"],
        "requested_revision": config.get("revision", "main"),
        "resolved_revision": getattr(base_config, "_commit_hash", None),
        "data_manifest": data_manifest,
        "environment": environment_info(),
        "objective": config["method"],
    }
    write_json(output / "run_config.json", config)
    write_json(output / "run_metadata.json", metadata)
    if config.get("wandb", {}).get("enabled", False):
        import wandb

        wandb.run.config.update({"provenance": metadata}, allow_val_change=True)
    tokenizer.save_pretrained(output / "tokenizer")
    stop = GracefulStopCallback()
    trainer = trainer_class(
        model=model,
        args=args,
        train_dataset=training,
        eval_dataset=validation,
        data_collator=CompletionCollator(tokenizer.pad_token_id),
        processing_class=tokenizer,
        callbacks=[TimingCallback(output), stop],
    )
    started = time.perf_counter()
    previous_handler = signal.signal(signal.SIGUSR1, stop.request_stop)
    try:
        result = trainer.train(resume_from_checkpoint=resume_from_checkpoint)
    finally:
        signal.signal(signal.SIGUSR1, previous_handler)
    metrics = result.metrics | {"wall_seconds": time.perf_counter() - started}
    if stop.requested:
        metrics["interrupted"] = True
        write_json(output / "metrics.json", metrics)
        raise InterruptedError(
            "Saved a checkpoint after SIGUSR1; resume this run before evaluation"
        )
    metrics.update(trainer.evaluate())
    trainer.save_model(str(output / "adapter"))
    tokenizer.save_pretrained(output / "adapter")
    adapter_config = output / "adapter" / "adapter_config.json"
    if metadata["resolved_revision"]:
        value = json.loads(adapter_config.read_text())
        value["revision"] = metadata["resolved_revision"]
        write_json(adapter_config, value)
    write_json(output / "metrics.json", metrics)
    return metrics
