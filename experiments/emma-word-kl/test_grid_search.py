"""CPU integration checks for search selection, data isolation and exact promotion state."""

import contextlib
import copy
import hashlib
import io
import json
import os
import random
import subprocess
import tempfile
import unittest
from pathlib import Path
from unittest.mock import patch

import common
import grid_search as search
import preflight_gpu
import run
import torch
from resume_policy import validate_resume_checkpoint
from stage_runtime import stage
from train_policy import evaluation_subset
from transformers import get_cosine_schedule_with_warmup


def fixture(plan, folder, trial, rung=150):
    """Synthetic validation records, not model results; no model/GPU is loaded."""
    folder = Path(folder)
    search.write(folder / "grid-search-plan.json", plan)
    cfg = search.trial_config(plan, trial, rung)
    path = folder / f"{trial}-rung{rung}.json"
    search.write(path, cfg)
    runtime = folder / path.stem
    with contextlib.redirect_stdout(io.StringIO()):
        stage(runtime, path)
    return runtime, populate_output(runtime, rung)


def populate_output(runtime, rung):
    cfg = search.read(runtime / "run-config.json")
    output = runtime / "outputs/cpu-fixture"
    output.mkdir()
    cases = evaluation_subset(
        common.read_rows(common.ROOT / "data/policy_validation.jsonl"), 16, 53
    )
    rows = [
        {
            "id": f"{case['id']}-{sample}",
            "case_id": case["id"],
            "passage": "the " * 300,
            "normalized_reward": 0.8,
            "quality_reward": 0.8,
            "terminated": True,
        }
        for case in cases
        for sample in range(4)
    ]
    common.write_rows(output / "validation-0-samples.jsonl", rows)
    common.write_rows(output / f"validation-{rung}-samples.jsonl", rows)
    target = common.make_distribution(
        cfg, [r["response"] for r in common.read_rows(common.ROOT / "data/reference.jsonl")]
    )
    metrics = {
        "judge_quality_mean": 0.8,
        "length_in_range_rate": 1.0,
        "passages": 64,
        "mean_word_count": 300.0,
        "distribution_kl": target.kl(target.counts([r["passage"] for r in rows]).sum(0)),
        "joint_reward": -1000000.0,
    }
    search.write(output / f"validation-{rung}-metrics.json", metrics)
    search.write(
        output / "run_config.json",
        {
            **cfg,
            "human_validated": False,
            "distribution_weight_actual": cfg["distribution_weight"],
            "reward_model_wandb_url": "https://example.invalid/cpu-fixture",
        },
    )
    search.write(
        output / "comparison.json",
        {
            "training_updates": rung,
            "training_passages": rung * 16,
            "final_evaluation_split": "validation",
            "stop_reason": "step_limit",
        },
    )
    return output


def checkpoint(runtime, output, step=150):
    """Real tiny AdamW/scheduler state, with synthetic adapter and four RNG records."""
    cfg = search.read(runtime / "run-config.json")
    parameter = torch.nn.Parameter(torch.ones(1))
    optimizer = torch.optim.AdamW([parameter], lr=cfg["learning_rate"])
    scheduler = get_cosine_schedule_with_warmup(optimizer, 50, 1000)
    for _ in range(step):
        parameter.grad = torch.ones_like(parameter)
        optimizer.step()
        scheduler.step()
    p = output / f"checkpoint-{step}"
    (p / "adapter").mkdir(parents=True)
    (p / "adapter/adapter_model.safetensors").write_bytes(b"synthetic-CPU-adapter")
    state = {
        "step": step,
        "optimizer": optimizer.state_dict(),
        "scheduler": scheduler.state_dict(),
        "prompt_rng": random.Random(42).getstate(),
        "rank_rng": [
            {"torch": torch.get_rng_state(), "python": random.getstate(), "cuda": None}
            for _ in range(4)
        ],
        "config_hash": hashlib.sha256(json.dumps(cfg, sort_keys=True).encode()).hexdigest(),
        "manifest": {"arm": "quality-unigram"},
    }
    torch.save(state, p / "state.pt")
    search.write(p / "complete.json", {"step": step, "world_size": 4, "complete": True})
    return state


class GridSearchTests(unittest.TestCase):
    def test_slurm_launch_finds_shared_environment_for_nested_trials(self):
        with tempfile.TemporaryDirectory() as folder:
            root = Path(folder)
            python = root / ".venv-h200/bin/python"
            python.parent.mkdir(parents=True)
            python.write_text('#!/bin/sh\nprintf "%s\\n" "$HF_HOME" "$PWD" "$*"\n')
            python.chmod(0o755)
            source = common.ROOT / "policy.sbatch"
            environment = {k: v for k, v in os.environ.items() if k != "STOCH_PROJECT_DIR"}
            for relative in ("experiments/old-runtime", "experiments/grid/trial-rung150"):
                runtime = root / relative
                runtime.mkdir(parents=True)
                script = runtime / "policy.sbatch"
                script.write_bytes(source.read_bytes())
                env = {**environment, "SLURM_SUBMIT_DIR": str(runtime)}
                result = subprocess.run(
                    ["bash", str(script)], env=env, capture_output=True, text=True, check=True
                )
                self.assertEqual(
                    result.stdout.splitlines(),
                    [str(root / ".cache/huggingface"), str(runtime), "run.py"],
                )
            bad = subprocess.run(
                ["bash", str(script)],
                env={**env, "STOCH_PROJECT_DIR": str(root / "missing")},
                capture_output=True,
                text=True,
            )
            self.assertNotEqual(bad.returncode, 0)

    def test_nine_profiles_caps_and_no_inherited_gpu_approval(self):
        plan = search.make_plan()
        self.assertEqual(len(search.trials()), 9)
        self.assertEqual(search.trials()[search.CONTROL], (0.1, 0.2))
        for trial in search.trials():
            cfg = search.trial_config(plan, trial, 150)
            self.assertEqual(cfg["steps"], 1000)
            self.assertEqual(cfg["stop_after_steps"], 150)
            self.assertEqual(cfg["validation_prompt_count"], 16)
            self.assertEqual(cfg["final_evaluation_split"], "validation")
            self.assertFalse(cfg["gpu_submission_authorized"])
            self.assertEqual(cfg["approved_maximum_gpu_hours"], 0)
        self.assertEqual(
            sum(
                4 * search.WALL_SECONDS[s] / 3600 * count
                for s, count in [(150, 9), (400, 3), (1000, 2)]
            ),
            118,
        )
        for value in [-0.1, float("nan"), True]:
            self.assertRaises(ValueError, search.make_plan, quality_tolerance=value)

    def test_runtime_preflight_rejects_launch_and_plan_or_source_tampering(self):
        plan = search.make_plan()
        with tempfile.TemporaryDirectory() as folder:
            runtime, _ = fixture(plan, folder, search.CONTROL)
            with patch.object(preflight_gpu, "ROOT", runtime):
                report = preflight_gpu.preflight_gpu(False)
                self.assertEqual(report["gpus"], 4)
                self.assertEqual(report["gpu_wall_seconds"], 5400)
                with patch.object(run.subprocess, "Popen") as launch:
                    self.assertRaises(PermissionError, run.run)
                    launch.assert_not_called()
            cfg = search.read(runtime / "run-config.json")
            for change in [
                {"distribution_weight": 0.2},
                {"length_penalty_weight": 0.1},
                {"final_evaluation_split": "test"},
                {"stop_after_steps": 1000},
                {"requested_allocation_seconds": 20000},
            ]:
                self.assertRaises(
                    ValueError, search.validate_search_config, {**cfg, **change}, runtime
                )
            changed = copy.deepcopy(plan)
            changed["compliance_floor"] = 0.1
            search.write(runtime / "grid-search-plan.json", changed)
            self.assertRaises(ValueError, search.validate_search_config, cfg, runtime)
            search.write(runtime / "grid-search-plan.json", plan)
            (runtime / "length_reward.py").write_text("# changed reward\n")
            self.assertRaises(ValueError, search.validate_search_config, cfg, runtime)

    def test_common_constraints_rank_raw_kl_and_keep_control(self):
        plan = search.make_plan()
        records = {
            trial: {
                "baseline_quality": 0.8,
                "raw_quality": 0.9,
                "length_compliance": 0.9,
                "raw_word_kl": float(i + 1),
                "joint_reward": float(100 - i),
            }
            for i, trial in enumerate(search.trials())
        }
        records[search.CONTROL]["raw_word_kl"] = 999
        selected = search.choose(plan, records, 150)
        self.assertEqual(selected["selected_for_next_rung"][0], search.CONTROL)
        self.assertEqual(selected["ranking"][0], "kl010-len000")
        records["kl010-len000"]["raw_quality"] = 0.1
        records["kl010-len040"]["length_compliance"] = 0.1
        selected = search.choose(plan, records, 150)
        self.assertNotIn("kl010-len000", selected["selected_for_next_rung"])
        self.assertNotIn("kl010-len040", selected["selected_for_next_rung"])
        for record in records.values():
            record["length_compliance"] = 0
        none = search.choose(plan, records, 1000)
        self.assertEqual(none["status"], "no_feasible_trial")
        self.assertIsNone(none["winner"])

    def test_result_loader_checks_actual_validation_not_weighted_training_rewards(self):
        plan = search.make_plan()
        with tempfile.TemporaryDirectory() as folder:
            runtime, output = fixture(plan, folder, search.CONTROL)
            record = search._load_trial(plan, search.CONTROL, 150, output, output, {})
            self.assertEqual(record["raw_quality"], 0.8)
            self.assertEqual(record["length_compliance"], 1.0)
            self.assertNotIn("joint_reward", record)
            metrics = search.read(output / "validation-150-metrics.json")
            search.write(output / "validation-150-metrics.json", {**metrics, "distribution_kl": 0})
            self.assertRaises(
                ValueError, search._load_trial, plan, search.CONTROL, 150, output, output, {}
            )
            search.write(output / "validation-150-metrics.json", metrics)
            done = search.read(output / "comparison.json")
            for change in [{"training_updates": 149}, {"final_evaluation_split": "test"}]:
                search.write(output / "comparison.json", {**done, **change})
                self.assertRaises(
                    ValueError, search._load_trial, plan, search.CONTROL, 150, output, output, {}
                )

    def test_complete_rung_ranking_and_real_checkpoint_bound_promotion(self):
        plan = search.make_plan()
        with tempfile.TemporaryDirectory() as folder:
            root = Path(folder)
            search.write(root / "plan.json", plan)
            index = {}
            for trial in search.trials():
                runtime, output = fixture(plan, root, trial)
                index[trial] = str(output)
            search.write(root / "index.json", index)
            decision = search.rank(
                root / "plan.json", root / "index.json", 150, root / "decision.json"
            )
            selected = decision["selection"]["selected_for_next_rung"]
            self.assertEqual(len(selected), 3)
            self.assertIn(search.CONTROL, selected)
            self.assertRaises(
                FileNotFoundError,
                search.promote,
                root / "plan.json",
                root / "decision.json",
                root / "missing-checkpoint",
            )
            self.assertFalse((root / "missing-checkpoint").exists())
            saved = {}
            for trial in selected:
                output = Path(index[trial])
                saved[trial] = checkpoint(output.parents[1], output)
            with contextlib.redirect_stdout(io.StringIO()):
                report = search.promote(
                    root / "plan.json",
                    root / "decision.json",
                    root / "promoted",
                    stage_runtimes=True,
                )
            self.assertEqual(report["jobs_submitted"], 0)
            for trial in selected:
                cfg = search.read(root / "promoted" / f"{trial}-rung400.json")
                restored = validate_resume_checkpoint(cfg, "quality-unigram", world=4)
                self.assertEqual(restored["scheduler"], saved[trial]["scheduler"])
                self.assertEqual(restored["prompt_rng"], saved[trial]["prompt_rng"])
                self.assertEqual(restored["scheduler"]["last_epoch"], 150)
                self.assertEqual(cfg["steps"], 1000)
                self.assertEqual(cfg["resume_step"], 150)
                self.assertEqual(cfg["stop_after_steps"], 400)
                self.assertFalse(cfg["gpu_submission_authorized"])
                with patch.object(preflight_gpu, "ROOT", root / "promoted" / f"{trial}-rung400"):
                    self.assertTrue(preflight_gpu.preflight_gpu(False)["passed"])
            # Exercise both subsequent rungs and final selection on CPU fixtures.
            index400 = {}
            for trial in selected:
                runtime = root / "promoted" / f"{trial}-rung400"
                output = populate_output(runtime, 400)
                checkpoint(runtime, output, step=400)
                index400[trial] = str(output)
            search.write(root / "index400.json", index400)
            second = search.rank(
                root / "plan.json",
                root / "index400.json",
                400,
                root / "decision400.json",
                root / "decision.json",
            )
            self.assertEqual(len(second["selection"]["selected_for_next_rung"]), 2)
            with contextlib.redirect_stdout(io.StringIO()):
                search.promote(
                    root / "plan.json",
                    root / "decision400.json",
                    root / "finalists",
                    stage_runtimes=True,
                )
            index1000 = {}
            for trial in second["selection"]["selected_for_next_rung"]:
                runtime = root / "finalists" / f"{trial}-rung1000"
                cfg = search.read(runtime / "run-config.json")
                self.assertEqual(cfg["resume_step"], 400)
                self.assertEqual(cfg["stop_after_steps"], 1000)
                with patch.object(preflight_gpu, "ROOT", runtime):
                    self.assertEqual(preflight_gpu.preflight_gpu(False)["gpu_wall_seconds"], 18000)
                index1000[trial] = str(populate_output(runtime, 1000))
            search.write(root / "index1000.json", index1000)
            final = search.rank(
                root / "plan.json",
                root / "index1000.json",
                1000,
                root / "decision1000.json",
                root / "decision400.json",
            )
            self.assertIn(final["selection"]["winner"], index1000)
            self.assertEqual(final["selection"]["selected_for_next_rung"], [])
            self.assertRaises(
                ValueError,
                search.promote,
                root / "plan.json",
                root / "decision1000.json",
                root / "beyond-final",
            )
            changed = copy.deepcopy(decision)
            changed["records"][search.CONTROL]["raw_quality"] = 1.0
            search.write(root / "changed-decision.json", changed)
            self.assertRaises(
                ValueError,
                search.promote,
                root / "plan.json",
                root / "changed-decision.json",
                root / "bad",
            )
            search.write(
                root / "index-missing.json", {k: v for k, v in index.items() if k != search.CONTROL}
            )
            self.assertRaises(
                ValueError,
                search.rank,
                root / "plan.json",
                root / "index-missing.json",
                150,
                root / "bad-rank.json",
            )
            source = Path(index[search.CONTROL]) / "validation-150-metrics.json"
            source.write_text(source.read_text() + " ")
            self.assertRaises(
                ValueError,
                search.promote,
                root / "plan.json",
                root / "decision.json",
                root / "bad-input",
            )


if __name__ == "__main__":
    unittest.main()
