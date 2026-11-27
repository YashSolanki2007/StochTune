"""CPU-only grid preparation, validation ranking and checkpoint-bound promotion.

This CLI never submits jobs or transfers allocation approval. Every prepared
runtime requires separate GPU authorization through the existing preflight.
"""

import argparse
import hashlib
import json
import math
import statistics
from collections import Counter
from pathlib import Path

PROFILE = "emma-grid-search-v1"
KL_WEIGHTS = (0.1, 0.3, 1.0)
LENGTH_WEIGHTS = (0.0, 0.2, 0.4)
RUNGS = (150, 400, 1000)
SURVIVORS = {150: 3, 400: 2}
CONTROL = "kl010-len020"
# Per-allocation ceilings, not estimates. No allocations are submitted here.
WALL_SECONDS = {150: 5400, 400: 7200, 1000: 18000}
TRACKING_FIELDS = {"distribution_weight_actual", "reward_model_wandb_url", "human_validated"}


def read(path):
    return json.loads(Path(path).read_text())


def write(path, value):
    Path(path).write_text(json.dumps(value, indent=2, allow_nan=False) + "\n")


def sha(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def training_sha(path):
    # stage_runtime changes only Slurm directives for each allocation. Bind the
    # launch body here; the complete staged script has a deployment-manifest hash.
    path = Path(path)
    if path.suffix == ".sbatch":
        body = "\n".join(
            line for line in path.read_text().splitlines() if not line.startswith("#SBATCH ")
        )
        return hashlib.sha256(body.encode()).hexdigest()
    return sha(path)


def digest(value):
    return hashlib.sha256(
        json.dumps(value, sort_keys=True, separators=(",", ":"), allow_nan=False).encode()
    ).hexdigest()


def trials():
    return {
        f"kl{round(kl * 100):03d}-len{round(length * 100):03d}": (kl, length)
        for kl in KL_WEIGHTS
        for length in LENGTH_WEIGHTS
    }


def resource_limits(rung):
    if type(rung) is not int or rung not in RUNGS:
        raise ValueError("Unknown grid rung")
    return {
        "allocation_seconds": WALL_SECONDS[rung],
        "gpus": 4,
        "calibration_seconds": 600,
        "evaluation_reserve_seconds": 1800 if rung == 1000 else 1200,
        "controller_reserve_seconds": 60,
    }


def training_sources():
    """Use the same repository/staged resolution as stage_runtime."""
    from common import REPO, ROOT

    sources = {
        name: ROOT / name
        for name in (
            "common.py",
            "fiction_reward.py",
            "length_reward.py",
            "train_policy.py",
            "run.py",
            "calibrate.py",
            "policy.sbatch",
        )
    }
    for name, staged in (
        ("common.py", "_common_impl.py"),
        ("train_policy.py", "_policy_impl.py"),
        ("resume_policy.py", "resume_policy.py"),
        ("word_distribution.py", "word_distribution.py"),
    ):
        local = ROOT / staged
        sources[staged] = local if local.exists() else REPO / "experiments/helpsteer-rl-v2" / name
    for name in (
        "ddp_passage_rl.py",
        "passage_reward.py",
        "pilot.py",
        "judge.py",
        "passage_rl.py",
        "quality.py",
    ):
        local = ROOT / "support" / name
        sources[f"support/{name}"] = (
            local if local.exists() else REPO / "experiments/shakespeare-quality-pilot" / name
        )
    return sources


def make_plan(quality_tolerance=0.01, compliance_floor=0.7):
    from common import ROOT
    from prepare_long import long_profile

    import stochastic_llm.rl as likelihood

    cfg = long_profile(gpus=4)
    cfg.update(
        experiment_profile=PROFILE,
        wandb_group=PROFILE,
        steps=1000,
        warmup_steps=50,
        validation_prompt_count=16,
        eval_samples_per_prompt=4,
        final_evaluation_split="validation",
        length_min_words=250,
        length_max_words=350,
        length_penalty_unit_words=100,
        authorization="Grid code/configuration only; no GPU spending approval.",
        policy_initialization="Fresh pinned actor per trial; full 1000-step schedule from outset",
        experiment_purpose="Predeclared two-dimensional reward-weight grid; validation only; "
        "no automated submission, retry, test evaluation or seed-confirmation jobs.",
    )
    plan = {
        "schema_version": 1,
        "profile": PROFILE,
        "word_kl_weights": list(KL_WEIGHTS),
        "length_weights": list(LENGTH_WEIGHTS),
        "quality_weight": 1.0,
        "rungs": list(RUNGS),
        "control_trial": CONTROL,
        "quality_tolerance": quality_tolerance,
        "compliance_floor": compliance_floor,
        "violation_quality_unit": 0.01,
        "violation_compliance_unit": 0.1,
        "base_config": cfg,
        "data_sha256": {
            name: sha(ROOT / name)
            for name in ("data/reference.jsonl", "data/policy_validation.jsonl")
        },
        "training_code_sha256": {
            name: training_sha(path) for name, path in training_sources().items()
        },
        "likelihood_implementation_sha256": sha(likelihood.__file__),
    }
    validate_plan(plan)
    return plan


def validate_plan(plan):
    if (
        plan["schema_version"] != 1
        or plan["profile"] != PROFILE
        or plan["word_kl_weights"] != list(KL_WEIGHTS)
        or plan["length_weights"] != list(LENGTH_WEIGHTS)
        or plan["quality_weight"] != 1.0
        or plan["rungs"] != list(RUNGS)
        or plan["control_trial"] != CONTROL
    ):
        raise ValueError("Unexpected search protocol")
    for key in ("quality_tolerance", "compliance_floor"):
        value = plan[key]
        if isinstance(value, bool) or not isinstance(value, (int, float)):
            raise ValueError("Selection thresholds must be numbers")
        if not math.isfinite(value) or not 0 <= value <= 1:
            raise ValueError("Invalid selection threshold")
    if plan["violation_quality_unit"] != 0.01 or plan["violation_compliance_unit"] != 0.1:
        raise ValueError("Unexpected fallback ranking scale")
    cfg = plan["base_config"]
    expected = {
        "experiment_profile": PROFILE,
        "wandb_group": PROFILE,
        "steps": 1000,
        "warmup_steps": 50,
        "policy_update": "reinforce",
        "quality_baseline": "prompt-loo",
        "distribution_objective": "word-kl",
        "reference_kl_beta": 0.002,
        "global_group_size": 16,
        "prompts_per_group": 4,
        "validation_prompt_count": 16,
        "eval_samples_per_prompt": 4,
        "validation_every": 50,
        "checkpoint_every": 25,
        "requested_gpus": 4,
        "distributed_world_size": 4,
        "final_evaluation_split": "validation",
        "length_min_words": 250,
        "length_max_words": 350,
        "length_penalty_unit_words": 100,
        "gpu_submission_authorized": False,
        "approved_allocation_seconds": 0,
        "approved_maximum_gpus": 0,
        "approved_maximum_gpu_hours": 0,
    }
    if any(cfg.get(k) != v for k, v in expected.items()):
        raise ValueError("Unexpected base configuration or inherited approval")
    if cfg.get("resume_checkpoint") or any(k.startswith("search_") for k in cfg):
        raise ValueError("Plan base must be fresh and independent of prior searches")


def trial_config(plan, trial, rung):
    validate_plan(plan)
    if trial not in trials():
        raise ValueError("Unknown grid trial")
    limits = resource_limits(rung)
    kl, length = trials()[trial]
    return {
        **plan["base_config"],
        "distribution_weight": kl,
        "length_penalty_weight": length,
        "search_plan_file": "grid-search-plan.json",
        "search_plan_sha256": digest(plan),
        "search_trial_id": trial,
        "search_rung": rung,
        "stop_after_steps": rung,
        "requested_allocation_seconds": limits["allocation_seconds"],
        "evaluation_reserve_seconds": limits["evaluation_reserve_seconds"],
        "train_budget_seconds": limits["allocation_seconds"] - limits["evaluation_reserve_seconds"],
    }


def validate_search_config(cfg, root):
    """Check the frozen plan and, for promotions, the actual optimizer checkpoint."""
    from resume_policy import validate_resume_checkpoint, validate_resume_config

    import stochastic_llm.rl as likelihood

    root = Path(root)
    if cfg.get("search_plan_file") != "grid-search-plan.json":
        raise ValueError("Need the frozen search plan")
    plan = read(root / cfg["search_plan_file"])
    if digest(plan) != cfg["search_plan_sha256"]:
        raise ValueError("Search plan changed")
    if sha(likelihood.__file__) != plan["likelihood_implementation_sha256"]:
        raise ValueError("Installed sequence-likelihood implementation changed")
    expected = trial_config(plan, cfg["search_trial_id"], cfg["search_rung"])
    validate_resume_config(cfg, expected)
    for key in (
        "stop_after_steps",
        "final_evaluation_split",
        "requested_allocation_seconds",
        "train_budget_seconds",
        "evaluation_reserve_seconds",
        "wandb_group",
    ):
        if cfg.get(key) != expected[key]:
            raise ValueError("Search runtime mismatch: " + key)
    for name, value in plan["data_sha256"].items():
        if sha(root / name) != value:
            raise ValueError("Search data changed: " + name)
    for name, value in plan["training_code_sha256"].items():
        if training_sha(root / name) != value:
            raise ValueError("Training implementation changed: " + name)
    rung = cfg["search_rung"]
    if rung == RUNGS[0]:
        if cfg.get("resume_checkpoint"):
            raise ValueError("First rung must start fresh")
    else:
        previous = RUNGS[RUNGS.index(rung) - 1]
        if cfg.get("resume_step") != previous:
            raise ValueError("Promotions must resume the preceding rung")
        validate_resume_checkpoint(cfg, "quality-unigram", world=4)
    return plan


def _prepare_configs(plan, configs, output, stage_runtimes):
    from stage_runtime import stage

    output = Path(output).resolve()
    output.mkdir(parents=True, exist_ok=False)
    write(output / "grid-search-plan.json", plan)
    for trial, cfg in configs.items():
        path = output / f"{trial}-rung{cfg['search_rung']}.json"
        write(path, cfg)
        if stage_runtimes:
            stage(output / path.stem, path)
    manifest = {
        "plan_sha256": digest(plan),
        "rung": next(iter(configs.values()))["search_rung"],
        "configurations": list(configs),
        "jobs_submitted": 0,
        "gpu_submission_authorized": False,
        "maximum_stage_gpu_hours_if_separately_approved": sum(
            c["requested_gpus"] * c["requested_allocation_seconds"] / 3600 for c in configs.values()
        ),
    }
    write(output / "preparation.json", manifest)
    return manifest


def prepare(output, stage_runtimes=False, quality_tolerance=0.01, compliance_floor=0.7):
    plan = make_plan(quality_tolerance, compliance_floor)
    return _prepare_configs(
        plan,
        {trial: trial_config(plan, trial, RUNGS[0]) for trial in trials()},
        output,
        stage_runtimes,
    )


def _verify_inputs(report):
    for name, expected in report["input_sha256"].items():
        if sha(name) != expected:
            raise ValueError("Decision input changed: " + name)


def _verify_decision(plan, decision):
    _verify_inputs(decision)
    if decision["plan_sha256"] != digest(plan):
        raise ValueError("Decision uses a different plan")
    expected_count = {150: 9, 400: 3, 1000: 2}.get(decision["rung"])
    if (
        len(decision["records"]) != expected_count
        or CONTROL not in decision["records"]
        or set(read(decision["index_file"])) != set(decision["records"])
    ):
        raise ValueError("Decision does not include the complete rung")
    for trial, record in decision["records"].items():
        checked = _load_trial(
            plan, trial, decision["rung"], record["output"], record["initial_output"], {}
        )
        if checked != record:
            raise ValueError("Decision metrics or provenance changed")
    if choose(plan, decision["records"], decision["rung"]) != decision["selection"]:
        raise ValueError("Decision selection changed")


def _sample_metrics(path, expected_ids):
    from common import read_rows, words

    rows = read_rows(path)
    if len({r["id"] for r in rows}) != len(rows) or Counter(r["case_id"] for r in rows) != Counter(
        {k: 4 for k in expected_ids}
    ):
        raise ValueError("Wrong validation prompts, duplicates or sample budget")
    if {r["id"] for r in rows} != {f"{k}-{i}" for k in expected_ids for i in range(4)}:
        raise ValueError("Wrong validation sample identities")
    quality = [r["normalized_reward"] for r in rows]
    if not all(isinstance(q, (int, float)) and math.isfinite(q) and 0 <= q <= 1 for q in quality):
        raise ValueError("Invalid raw judge scores")
    return (
        {
            "judge_quality_mean": statistics.mean(quality),
            "mean_word_count": statistics.mean(len(words(r["passage"])) for r in rows),
            "length_in_range_rate": statistics.mean(
                250 <= len(words(r["passage"])) <= 350 for r in rows
            ),
            "passages": len(rows),
        },
        digest(sorted((r["id"], r["passage"], r["normalized_reward"]) for r in rows)),
        rows,
    )


def _load_trial(plan, trial, rung, output, initial, input_hashes):
    from common import ROOT, make_distribution, read_rows
    from resume_policy import validate_resume_config
    from train_policy import evaluation_subset

    output, initial = Path(output).resolve(), Path(initial).resolve()
    source = output.parents[1] / "run-config.json"
    cfg = read(source)
    validate_search_config(cfg, source.parent)
    expected = trial_config(plan, trial, rung)
    validate_resume_config(cfg, expected)
    for key in ("search_rung", "stop_after_steps", "final_evaluation_split"):
        if cfg.get(key) != expected[key]:
            raise ValueError("Wrong rung or evaluation split")
    logged = read(output / "run_config.json")
    if {k: v for k, v in logged.items() if k not in TRACKING_FIELDS} != cfg:
        raise ValueError("Recorded runtime configuration differs from deployed configuration")
    done = read(output / "comparison.json")
    if (
        done["training_updates"] != rung
        or done["training_passages"] != rung * 16
        or done["final_evaluation_split"] != "validation"
        or done["stop_reason"] != "step_limit"
    ):
        raise ValueError("Trial did not finish the required validation-only rung")
    cases = read_rows(ROOT / "data/policy_validation.jsonl")
    ids = {r["id"] for r in evaluation_subset(cases, 16, cfg["seed"] + 11)}
    if len(ids) != 16:
        raise ValueError("Need sixteen distinct validation prompts")
    for name, expected_hash in plan["data_sha256"].items():
        if sha(ROOT / name) != expected_hash:
            raise ValueError("Ranking data changed")
    baseline_path = initial / "validation-0-samples.jsonl"
    baseline, fingerprint, _ = _sample_metrics(baseline_path, ids)
    metrics_path = output / f"validation-{rung}-metrics.json"
    samples_path = output / f"validation-{rung}-samples.jsonl"
    actual, _, samples = _sample_metrics(samples_path, ids)
    metrics = read(metrics_path)
    for key, value in actual.items():
        if not math.isclose(metrics[key], value, rel_tol=1e-8, abs_tol=1e-8):
            raise ValueError("Validation metric does not match samples: " + key)
    target = make_distribution(
        cfg, [r["response"] for r in read_rows(ROOT / "data/reference.jsonl")]
    )
    kl = target.kl(target.counts([r["passage"] for r in samples]).sum(0))
    if not math.isfinite(kl) or not math.isclose(metrics["distribution_kl"], kl, abs_tol=1e-8):
        raise ValueError("Validation raw KL does not match samples")
    for path in (
        source,
        output / "run_config.json",
        output / "comparison.json",
        baseline_path,
        metrics_path,
        samples_path,
    ):
        input_hashes[str(path)] = sha(path)
    return {
        "trial_id": trial,
        "output": str(output),
        "initial_output": str(initial),
        "source_config": str(source),
        "baseline_fingerprint": fingerprint,
        "baseline_quality": baseline["judge_quality_mean"],
        "raw_quality": actual["judge_quality_mean"],
        "length_compliance": actual["length_in_range_rate"],
        "mean_word_count": actual["mean_word_count"],
        "raw_word_kl": kl,
    }


def choose(plan, records, rung):
    """Fixed constraints; fallback is explicit, comparable and independent of reward weights."""
    resource_limits(rung)
    ordered = []
    constraints = {}
    for trial, record in records.items():
        quality_floor = record["baseline_quality"] - plan["quality_tolerance"]
        qgap = max(0.0, quality_floor - record["raw_quality"])
        lgap = max(0.0, plan["compliance_floor"] - record["length_compliance"])
        violation = qgap / plan["violation_quality_unit"] + lgap / plan["violation_compliance_unit"]
        constraints[trial] = {
            "quality_floor": quality_floor,
            "feasible": qgap == 0 and lgap == 0,
            "normalized_violation": violation,
        }
        ordered.append(
            (
                qgap > 0 or lgap > 0,
                violation,
                record["raw_word_kl"],
                -record["raw_quality"],
                -record["length_compliance"],
                trial,
            )
        )
    order = [entry[-1] for entry in sorted(ordered)]
    feasible = [entry[-1] for entry in sorted(ordered) if not entry[0]]
    if rung in SURVIVORS:
        selected = [CONTROL] + [t for t in order if t != CONTROL][: SURVIVORS[rung] - 1]
    else:
        selected = []
    return {
        "ranking": order,
        "feasible_trials": feasible,
        "selected_for_next_rung": selected,
        "constraints": constraints,
        "selected_infeasible": [trial for trial in selected if trial not in feasible],
        "winner": feasible[0] if rung == RUNGS[-1] and feasible else None,
        "status": "feasible_candidates" if feasible else "no_feasible_trial",
        "fallback": "normalized constraint violation, then raw KL" if not feasible else None,
    }


def rank(plan_path, index_path, rung, output, previous_decision=None):
    plan = read(plan_path)
    validate_plan(plan)
    resource_limits(rung)
    index = read(index_path)
    hashes = {
        str(Path(plan_path).resolve()): sha(plan_path),
        str(Path(index_path).resolve()): sha(index_path),
    }
    previous = None
    if rung == RUNGS[0]:
        expected_trials = set(trials())
    else:
        if previous_decision is None:
            raise ValueError("Need the preceding promotion decision")
        previous = read(previous_decision)
        _verify_decision(plan, previous)
        if (
            previous["plan_sha256"] != digest(plan)
            or previous["rung"] != RUNGS[RUNGS.index(rung) - 1]
        ):
            raise ValueError("Wrong preceding decision")
        expected_trials = set(previous["selection"]["selected_for_next_rung"])
        hashes[str(Path(previous_decision).resolve())] = sha(previous_decision)
    if set(index) != expected_trials:
        raise ValueError("Need every expected trial exactly once; incomplete rungs are not ranked")
    records = {}
    for trial, path in index.items():
        initial = previous["records"][trial]["initial_output"] if previous else path
        records[trial] = _load_trial(plan, trial, rung, path, initial, hashes)
    if len({r["baseline_fingerprint"] for r in records.values()}) != 1:
        raise ValueError("Initial validation generations differ between trials")
    report = {
        "plan_sha256": digest(plan),
        "rung": rung,
        "records": records,
        "selection": choose(plan, records, rung),
        "input_sha256": hashes,
        "index_file": str(Path(index_path).resolve()),
        "jobs_submitted": 0,
    }
    if Path(output).exists():
        raise FileExistsError(output)
    write(output, report)
    return report


def promote(plan_path, decision_path, output, stage_runtimes=False):
    from resume_policy import validate_resume_checkpoint

    plan, decision = read(plan_path), read(decision_path)
    validate_plan(plan)
    _verify_decision(plan, decision)
    rung = decision["rung"]
    if decision["plan_sha256"] != digest(plan) or rung not in SURVIVORS:
        raise ValueError("Wrong plan or no further rung")
    selection = choose(plan, decision["records"], rung)
    if selection != decision["selection"]:
        raise ValueError("Promotion decision changed")
    configs = {}
    next_rung = RUNGS[RUNGS.index(rung) + 1]
    for trial in selection["selected_for_next_rung"]:
        record = decision["records"][trial]
        checkpoint = Path(record["output"]) / f"checkpoint-{rung}"
        cfg = trial_config(plan, trial, next_rung)
        cfg.update(
            policy_initialization=f"Resume this grid trial's complete checkpoint {rung}",
            resume_checkpoint=str(checkpoint),
            resume_source_config=record["source_config"],
            resume_source_config_sha256=sha(record["source_config"]),
            resume_checkpoint_sha256={
                str(f.relative_to(checkpoint)): sha(f) for f in checkpoint.rglob("*") if f.is_file()
            },
            resume_step=rung,
        )
        validate_resume_checkpoint(cfg, "quality-unigram", world=4)
        configs[trial] = cfg
    return _prepare_configs(plan, configs, output, stage_runtimes)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    commands = parser.add_subparsers(dest="command", required=True)
    prep = commands.add_parser("prepare")
    prep.add_argument("--output", type=Path, required=True)
    prep.add_argument("--stage", action="store_true")
    prep.add_argument("--quality-tolerance", type=float, default=0.01)
    prep.add_argument("--compliance-floor", type=float, default=0.7)
    ranking = commands.add_parser("rank")
    ranking.add_argument("--plan", type=Path, required=True)
    ranking.add_argument("--index", type=Path, required=True)
    ranking.add_argument("--rung", type=int, choices=RUNGS, required=True)
    ranking.add_argument("--output", type=Path, required=True)
    ranking.add_argument("--previous-decision", type=Path)
    promotion = commands.add_parser("promote")
    promotion.add_argument("--plan", type=Path, required=True)
    promotion.add_argument("--decision", type=Path, required=True)
    promotion.add_argument("--output", type=Path, required=True)
    promotion.add_argument("--stage", action="store_true")
    args = parser.parse_args()
    if args.command == "prepare":
        result = prepare(args.output, args.stage, args.quality_tolerance, args.compliance_floor)
    elif args.command == "rank":
        result = rank(args.plan, args.index, args.rung, args.output, args.previous_decision)
    else:
        result = promote(args.plan, args.decision, args.output, args.stage)
    print(json.dumps(result, indent=2))


if __name__ == "__main__":
    main()
