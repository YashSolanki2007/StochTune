"""Independent, unconstrained numeric requests, batched for GPU throughput."""

import csv
import json
import time
from pathlib import Path

import torch
from peft import PeftConfig, PeftModel
from transformers import GenerationConfig, set_seed

from .io import file_sha256, fingerprint, read_jsonl, require_empty_directory, write_json
from .metrics import parse_number, score_samples, summarize
from .modeling import environment_info, load_model_and_tokenizer
from .tracking import log_evaluation_case, tracked


def generate_samples(model, tokenizer, case, config):
    count = int(config["samples_per_prompt"])
    batch_size = int(config["batch_size"])
    generation = GenerationConfig(
        do_sample=True,
        temperature=1.0,
        top_p=1.0,
        top_k=0,
        typical_p=1.0,
        repetition_penalty=1.0,
        num_beams=1,
        max_new_tokens=config["max_new_tokens"],
        eos_token_id=tokenizer.eos_token_id,
        pad_token_id=tokenizer.pad_token_id,
        bos_token_id=tokenizer.bos_token_id,
        use_cache=True,
    )
    for start in range(0, count, batch_size):
        # Independent deterministic streams per batch; never reuse the same seed
        # for every batch. Resume reruns only incomplete prompt files.
        seed = int(fingerprint([config["seed"], case["id"], start])[:8], 16)
        set_seed(seed)
        size = min(batch_size, count - start)
        inputs = tokenizer(
            [case["prompt"]] * size, padding=True, return_tensors="pt", return_token_type_ids=False
        )
        inputs = {key: value.to(model.device) for key, value in inputs.items()}
        with torch.inference_mode():
            generated = model.generate(**inputs, generation_config=generation)
        completions = generated[:, inputs["input_ids"].shape[1] :].cpu().tolist()
        for offset, ids in enumerate(completions):
            terminated = tokenizer.eos_token_id in ids
            before_eos = ids[: ids.index(tokenizer.eos_token_id)] if terminated else ids
            # Preserve unexpected special tokens so they cannot become valid numbers
            # merely because a decoder silently removed them.
            text = tokenizer.decode(before_eos, skip_special_tokens=False)
            parsed = parse_number(text)
            valid = parsed is not None and terminated
            yield {
                "case_id": case["id"],
                "index": start + offset,
                "text": text,
                "terminated": terminated,
                "valid": valid,
                "value": parsed if valid else None,
            }


@tracked("evaluation")
def evaluate(config, resume=False):
    config = dict(config)
    for key in ("samples_per_prompt", "batch_size", "max_new_tokens"):
        if int(config[key]) < 1:
            raise ValueError(f"{key} must be positive")
    cases = read_jsonl(config["cases"])
    if not cases or len({case["id"] for case in cases}) != len(cases):
        raise ValueError("Evaluation cases must be nonempty and have unique IDs")
    # IDs are used as filenames; reject anything other than simple identifiers.
    if any(not c["id"].replace("-", "").replace("_", "").isalnum() for c in cases):
        raise ValueError("Case IDs may contain only letters, digits, hyphens, and underscores")
    adapter_hashes = None
    if config.get("adapter"):
        adapter = Path(config["adapter"])
        if not (adapter / "adapter_model.safetensors").is_file():
            raise ValueError("Expected a local PEFT adapter containing adapter_model.safetensors")
        adapter_hashes = {
            name: file_sha256(adapter / name)
            for name in ("adapter_config.json", "adapter_model.safetensors")
        }
        peft_config = PeftConfig.from_pretrained(adapter)
        if peft_config.base_model_name_or_path != config["model"]:
            raise ValueError("Adapter base model does not match evaluation model")
        if peft_config.revision:
            requested_revision = config.get("revision", "main")
            if requested_revision not in ("main", peft_config.revision):
                raise ValueError(
                    "Explicit evaluation revision differs from the adapter base revision"
                )
            config["revision"] = peft_config.revision
    output = Path(config["output_dir"])
    identity = {
        "config": dict(config),
        "cases_sha256": file_sha256(config["cases"]),
        "adapter_sha256": adapter_hashes,
    }
    if resume:
        previous = json.loads((output / "identity.json").read_text())
        if previous != identity:
            raise ValueError("Evaluation inputs differ from the existing run")
        metadata = json.loads((output / "metadata.json").read_text())
        if metadata.get("resolved_revision"):
            config["revision"] = metadata["resolved_revision"]
    else:
        require_empty_directory(output)
    samples_dir = output / "samples"
    samples_dir.mkdir(exist_ok=True)
    model, tokenizer = load_model_and_tokenizer(config)
    if config.get("adapter"):
        model = PeftModel.from_pretrained(model, config["adapter"])
    model.eval()
    base = model.get_base_model() if config.get("adapter") else model
    write_json(output / "identity.json", identity)
    write_json(
        output / "metadata.json",
        {
            "environment": environment_info(),
            "resolved_revision": getattr(base.config, "_commit_hash", None),
            "metrics_note": (
                "Fidelity is conditional on valid numeric responses; "
                "inspect validity alongside it. "
                "Reference: the full law rounded to decimal precision. No KS p-values."
            ),
        },
    )
    started = time.perf_counter()
    results = []
    for case in cases:
        sample_path = samples_dir / f"{case['id']}.jsonl"
        if resume and sample_path.exists():
            rows = read_jsonl(sample_path)
            if (
                len(rows) != config["samples_per_prompt"]
                or [r["index"] for r in rows] != list(range(len(rows)))
                or any(r["case_id"] != case["id"] for r in rows)
            ):
                raise ValueError(f"Invalid completed sample file: {sample_path}")
        else:
            temporary = sample_path.with_suffix(".jsonl.tmp")
            rows = []
            with temporary.open("w") as handle:
                for row in generate_samples(model, tokenizer, case, config):
                    rows.append(row)
                    handle.write(json.dumps(row, allow_nan=False) + "\n")
            temporary.replace(sample_path)
        result = {
            "id": case["id"],
            "family": case["family"],
            "category": case["category"],
            **score_samples(case, rows, config.get("decimals", 5)),
        }
        results.append(result)
        if config.get("wandb", {}).get("enabled", False):
            log_evaluation_case(case, result, rows, len(results) - 1)
        write_json(output / "results.json", results)
        write_json(output / "summary.json", summarize(results))
        print(
            f"{case['id']}: valid={result['numeric_valid_rate']:.1%}, "
            f"normalized W1={result['normalized_wasserstein']}",
            flush=True,
        )
    with (output / "results.csv").open("w", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(results[0]))
        writer.writeheader()
        writer.writerows(results)
    write_json(output / "timing.json", {"invocation_seconds": time.perf_counter() - started})
    return summarize(results)


def compare_runs(paths, output):
    comparisons = []
    identities = []
    revisions = []
    for path in paths:
        path = Path(path)
        identity = json.loads((path / "identity.json").read_text())
        identities.append(identity)
        metadata = json.loads((path / "metadata.json").read_text())
        revisions.append(metadata.get("resolved_revision") or identity["config"].get("revision"))
        results = json.loads((path / "results.json").read_text())
        cases = read_jsonl(identity["config"]["cases"])
        if {r["id"] for r in results} != {c["id"] for c in cases}:
            raise ValueError(f"Incomplete evaluation: {path}")
        comparisons.append({"run": str(path), "summary": summarize(results)})
    reference = identities[0]
    for index, identity in enumerate(identities[1:], start=1):
        if (
            identity["config"]["model"] != reference["config"]["model"]
            or revisions[index] != revisions[0]
        ):
            raise ValueError("Cannot compare different base models or resolved model revisions")
        if identity["cases_sha256"] != reference["cases_sha256"]:
            raise ValueError("Cannot compare runs evaluated on different cases")
        for key in ("samples_per_prompt", "max_new_tokens", "decimals", "batch_size", "seed"):
            if identity["config"][key] != reference["config"][key]:
                raise ValueError(f"Cannot compare runs with different {key}")
    write_json(output, comparisons)
    return comparisons
