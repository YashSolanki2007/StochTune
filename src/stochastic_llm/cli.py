"""Command-line interface; heavyweight training imports are deliberately lazy."""

import argparse
import json

from .io import read_config


def main():
    parser = argparse.ArgumentParser(description="Controlled numeric sampling experiments")
    commands = parser.add_subparsers(dest="command", required=True)
    generate = commands.add_parser(
        "generate", help="Generate deterministic target data and held-out cases"
    )
    generate.add_argument("--config", default="configs/data.yaml")
    generate.add_argument("--output-dir")
    train = commands.add_parser("train", help="Train SFT or soft-KL LoRA adapters")
    train.add_argument("--config", default="configs/sft-h200.yaml")
    train.add_argument("--steps", type=int, dest="max_steps")
    train.add_argument("--resume-from-checkpoint")
    rl = commands.add_parser("train-rl", help="On-policy RL with a negative batch KL reward")
    rl.add_argument("--config", default="configs/batch-kl-rl-h200.yaml")
    rl.add_argument("--steps", type=int, dest="max_steps")
    rl.add_argument("--resume-from-checkpoint")
    evaluate = commands.add_parser("evaluate", help="Measure independent numeric sampling")
    evaluate.add_argument("--config", default="configs/evaluate.yaml")
    evaluate.add_argument("--adapter")
    evaluate.add_argument("--cases")
    evaluate.add_argument("--samples-per-prompt", type=int)
    evaluate.add_argument("--batch-size", type=int)
    evaluate.add_argument("--resume", action="store_true")
    for command in (train, rl, evaluate):
        command.add_argument("--model")
        command.add_argument("--revision")
        command.add_argument("--output-dir")
        command.add_argument("--device", choices=["cuda", "cpu", "mps"])
        command.add_argument("--precision", choices=["bf16", "fp16", "fp32"])
    smoke = commands.add_parser(
        "smoke", help="Exercise the complete pipeline on a tiny offline CPU model"
    )
    smoke.add_argument("--output-dir", default="outputs/smoke")
    compare = commands.add_parser("compare", help="Compare complete evaluations of the same cases")
    compare.add_argument("runs", nargs="+", help="Evaluation output directories")
    compare.add_argument("--output", default="outputs/comparison.json")
    prepare = commands.add_parser("prepare-model", help="Cache model weights before GPU allocation")
    prepare.add_argument("--model", default="Qwen/Qwen2.5-3B")
    prepare.add_argument("--revision", default="main")
    tracking = commands.add_parser(
        "tracking-check", help="Verify W&B login and project access; no run"
    )
    tracking.add_argument("--config", default="configs/sft-h200.yaml")
    tracking.add_argument("--create-project", action="store_true")
    args = parser.parse_args()
    if args.command == "tracking-check":
        from .tracking import tracking_check

        result = tracking_check(read_config(args.config), args.create_project)
    elif args.command == "prepare-model":
        from huggingface_hub import snapshot_download

        result = {
            "snapshot": snapshot_download(
                args.model,
                revision=args.revision,
                allow_patterns=["*.json", "*.safetensors", "*.txt", "*.model", "*.tiktoken"],
            )
        }
    elif args.command == "smoke":
        from .smoke import smoke

        result = smoke(args.output_dir)
    elif args.command == "compare":
        from .evaluate import compare_runs

        result = compare_runs(args.runs, args.output)
    else:
        config = read_config(args.config)
        excluded = {"config", "command", "resume", "resume_from_checkpoint"}
        config.update(
            {
                key: value
                for key, value in vars(args).items()
                if key not in excluded and value is not None
            }
        )
        if args.command == "generate":
            from .data import generate_data

            result = generate_data(config)
        elif args.command == "train":
            from .train import train

            result = train(config, args.resume_from_checkpoint)
        elif args.command == "train-rl":
            from .rl import train_rl

            result = train_rl(config, args.resume_from_checkpoint)
        else:
            from .evaluate import evaluate

            result = evaluate(config, args.resume)
    print(json.dumps(result, indent=2, allow_nan=False))
