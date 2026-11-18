"""Required online tracking for production; offline tracking for local tests only."""

import functools
import json
import os
from pathlib import Path

from .io import fingerprint, write_json


def tracking_settings(config):
    settings = dict(config.get("wandb", {}))
    if not settings.get("enabled", False):
        return None
    if settings.get("mode", "online") not in ("online", "offline"):
        raise ValueError("W&B mode must be online or offline")
    requested = settings.get("mode", "online")
    if os.environ.get("WANDB_MODE", requested) != requested:
        raise RuntimeError("WANDB_MODE disagrees with the required configured tracking mode")
    if os.environ.get("WANDB_DISABLED", "false").lower() in ("true", "1", "yes"):
        raise RuntimeError("W&B tracking is required but WANDB_DISABLED is set")
    return settings


def tracking_check(config, create_project=False):
    import wandb

    settings = tracking_settings(config)
    if settings is None or settings.get("mode", "online") != "online":
        raise ValueError("Tracking preflight requires enabled online tracking")
    base_url = settings.get("base_url", "https://api.forge.coreweave.com")
    if not wandb.login(host=base_url, verify=True, prompt=False, timeout=20):
        raise RuntimeError("W&B login required before submitting the job")
    api = wandb.Api(overrides={"base_url": base_url}, timeout=20)
    project, entity = settings["project"], settings["entity"]
    if create_project:
        api.create_project(project, entity)
    remote = api.project(project, entity)
    if remote is None:
        raise RuntimeError("W&B project not found or not accessible")
    return {
        "authenticated": True,
        "entity": entity,
        "project": project,
        "url": f"https://forge.coreweave.com/wandb/{entity}/{project}",
    }


def tracked(job_type):
    """Keep training/evaluation and their artifacts in a single tracked lifecycle."""

    def decorate(function):
        @functools.wraps(function)
        def wrapped(config, *args, **kwargs):
            settings = tracking_settings(config)
            if settings is None:
                return function(config, *args, **kwargs)
            import wandb

            if settings.get("mode", "online") == "online":
                # Fail before loading a multi-billion-parameter model if login is missing.
                tracking_check(config)
            output = Path(config["output_dir"])
            resuming = bool(kwargs.get("resume", False) or kwargs.get("resume_from_checkpoint"))
            if args:
                resuming = resuming or bool(args[0])
            if not resuming and output.exists() and any(output.iterdir()):
                raise FileExistsError(f"Directory is not empty: {output}")
            root = Path(settings.get("directory", "outputs/wandb"))
            root.mkdir(parents=True, exist_ok=True)
            record = (
                root / "identities" / f"{job_type}-{fingerprint(str(output.resolve()))[:20]}.json"
            )
            run_id = None
            if resuming:
                if not record.exists():
                    raise ValueError(
                        "Cannot resume tracking without its persisted W&B run identity"
                    )
                previous = json.loads(record.read_text())
                if previous["config_hash"] != fingerprint(config):
                    raise ValueError("Cannot resume tracking with a different config")
                run_id = previous["id"]
            os.environ["WANDB_LOG_MODEL"] = "false"
            run = wandb.init(
                entity=settings["entity"],
                project=settings["project"],
                name=settings.get("name", f"{job_type}-{output.name}"),
                group=settings.get("group", "qwen25-3b-numeric-v1"),
                job_type=job_type,
                config={"experiment": config, "slurm_job_id": os.environ.get("SLURM_JOB_ID")},
                dir=str(root),
                mode=settings.get("mode", "online"),
                id=run_id,
                resume="must" if run_id else "never",
                save_code=False,
                settings=wandb.Settings(
                    base_url=settings.get("base_url", "https://api.forge.coreweave.com"),
                    init_timeout=60,
                ),
            )
            write_json(record, {"id": run.id, "url": run.url, "config_hash": fingerprint(config)})
            try:
                code = wandb.Artifact(f"source-{run.id}", type="experiment-source")
                source_files = [Path("README.md"), Path("pyproject.toml")]
                for directory in ("src", "configs", "slurm", "scripts"):
                    source_files.extend(Path(directory).rglob("*"))
                for path in source_files:
                    if path.is_file() and path.suffix in (
                        ".py",
                        ".yaml",
                        ".sbatch",
                        ".sh",
                        ".md",
                        ".toml",
                    ):
                        code.add_file(str(path), name=str(path))
                run.log_artifact(code)
                result = function(config, *args, **kwargs)
                write_json(
                    output / "wandb_run.json",
                    {
                        "id": run.id,
                        "url": run.url,
                        "entity": settings["entity"],
                        "project": settings["project"],
                    },
                )
                metrics = {}

                def flatten(value, prefix):
                    for key, item in value.items():
                        name = f"{prefix}/{key}"
                        if isinstance(item, dict):
                            flatten(item, name)
                        elif isinstance(item, (float, int)) and not isinstance(item, bool):
                            metrics[name] = item

                flatten(result, job_type)
                run.summary.update(metrics)
                run.log(metrics)
                artifact = wandb.Artifact(f"{job_type}-{run.id}-results", type="experiment-results")
                for path in output.rglob("*"):
                    if path.is_file() and "checkpoint-" not in str(path.relative_to(output)):
                        if path.suffix in (".json", ".jsonl", ".csv", ".safetensors", ".txt"):
                            artifact.add_file(str(path), name=str(path.relative_to(output)))
                run.log_artifact(artifact)
                run.summary["status"] = "completed"
            except BaseException as error:
                run.summary["status"] = (
                    "interrupted" if isinstance(error, InterruptedError) else "failed"
                )
                run.finish(exit_code=1)
                raise
            run.finish(exit_code=0)
            return result

        return wrapped

    return decorate


def log_evaluation_case(case, result, rows, index):
    import wandb

    if wandb.run is None:
        return
    log = {
        f"sampling/{key}": value
        for key, value in result.items()
        if isinstance(value, (float, int)) and not isinstance(value, bool)
    }
    log["sampling/prompt_index"] = index
    wandb.log(log)
    table = wandb.Table(
        columns=["case_id", "family", "category", "prompt", "text", "value", "valid"]
    )
    for row in rows[:32]:
        table.add_data(
            case["id"],
            case["family"],
            case["category"],
            case["prompt"],
            row["text"],
            row["value"],
            row["valid"],
        )
    wandb.log({f"samples/{case['id']}": table})
