"""Audit the completed KL-reward RL experiment against its SFT starting point."""

import argparse
import json
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

from stochastic_llm.io import read_jsonl, write_json
from stochastic_llm.kl_reward import HistogramTarget, histogram_kl


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--recorded", action="store_true", help="Read the committed results/rl-v1 snapshot."
    )
    args = parser.parse_args()
    output = Path("outputs/rl-results")
    output.mkdir(parents=True, exist_ok=True)
    snapshot = Path("results/rl-v1")
    cases = read_jsonl(
        snapshot / "test-cases.jsonl" if args.recorded else "data/numeric-v1/test.jsonl"
    )
    results, summaries = {}, {}
    for label, folder, recorded in (
        ("sft", "eval-sft", "sft-evaluation"),
        ("rl", "eval-batch-kl-rl", "evaluation"),
    ):
        root = snapshot / recorded if args.recorded else Path("outputs") / folder
        results[label] = {r["id"]: r for r in read_jsonl_results(root / "results.json")}
        summaries[label] = json.loads((root / "summary.json").read_text())
        for case in cases:
            rows = read_jsonl(root / "samples" / f"{case['id']}.jsonl")
            target = HistogramTarget.from_case(case, 32, 5)
            bins = [target.assign(r["value"], r["terminated"]) for r in rows]
            counts = np.bincount(bins, minlength=len(target.probabilities))
            results[label][case["id"]]["target_histogram_kl"] = histogram_kl(
                counts, target.probabilities, 0.1
            )
    rng = np.random.default_rng(90210)
    indices = rng.integers(0, len(cases), size=(10000, len(cases)))
    statistics = {}
    for key in ("normalized_wasserstein", "ks_statistic", "target_histogram_kl"):
        before = np.array([results["sft"][c["id"]][key] for c in cases])
        after = np.array([results["rl"][c["id"]][key] for c in cases])
        delta = np.median(after[indices], axis=1) - np.median(before[indices], axis=1)
        mean_delta = (after - before)[indices].mean(1)
        statistics[key] = {
            "sft_median": float(np.median(before)),
            "rl_median": float(np.median(after)),
            "relative_median_reduction_percent": float(
                100 * (1 - np.median(after) / np.median(before))
            ),
            "sft_mean": float(before.mean()),
            "rl_mean": float(after.mean()),
            "relative_mean_reduction_percent": float(100 * (1 - after.mean() / before.mean())),
            "prompts_improved": int(np.sum(after < before)),
            "prompts_worsened": int(np.sum(after > before)),
            "prompts_tied": int(np.sum(after == before)),
            "paired_bootstrap_median_difference_95pct": np.quantile(delta, [0.025, 0.975]).tolist(),
            "paired_bootstrap_mean_difference_95pct": np.quantile(
                mean_delta, [0.025, 0.975]
            ).tolist(),
        }
    write_json(output / "comparison-statistics.json", statistics)
    write_json(output / "per-prompt-results.json", results)
    write_json(output / "evaluation-summaries.json", summaries)
    kl, ks, w = (
        statistics[key] for key in ("target_histogram_kl", "ks_statistic", "normalized_wasserstein")
    )
    validity = [summaries[name]["all"]["numeric_valid_rate"] for name in ("sft", "rl")]
    report = f"""# Completed KL-reward RL experiment

All jobs completed successfully. The final 250-update RL adapter was evaluated,
not the best validation checkpoint at update 225. The starting point was the
completed SFT adapter. RL generated 128,000 training responses and optimized a
negative smoothed forward histogram-KL reward with a leave-one-out group baseline.
No supervised answer labels were used during the RL stage.

The comparison uses the same 100 test prompts, 1,000 generations per prompt,
base model revision, sampling seed, temperature 1, top-p 1, and top-k disabled.
The CLI comparison checks these evaluation identities before accepting the runs.

| Metric | SFT | SFT + KL-reward RL |
|---|---:|---:|
| Mean test histogram KL | {kl["sft_mean"]:.6f} | {kl["rl_mean"]:.6f} |
| Median test histogram KL | {kl["sft_median"]:.6f} | {kl["rl_median"]:.6f} |
| Median normalized Wasserstein | {w["sft_median"]:.6f} | {w["rl_median"]:.6f} |
| Median KS | {ks["sft_median"]:.6f} | {ks["rl_median"]:.6f} |
| Numeric-only responses with EOS | {validity[0]:.3%} | {validity[1]:.3%} |

Mean test histogram KL improved {kl["relative_mean_reduction_percent"]:.2f}%.
Its median improved {kl["relative_median_reduction_percent"]:.2f}%.
KS and Wasserstein medians improved {ks["relative_median_reduction_percent"]:.2f}%
and {w["relative_median_reduction_percent"]:.2f}%, respectively.
KL improved on {kl["prompts_improved"]} of 100 prompts;
Wasserstein improved on {w["prompts_improved"]} of 100.

Histogram KL was computed post hoc from all test draws using the same 32-bin,
alpha=0.1 reward definition. Invalid and out-of-support draws remain in the
denominator as an invalid category. Unlike the training/validation batches of
256 draws, test histograms use 1,000 draws; their absolute KL values should not
be compared directly. The test cases also include harder parameter shifts,
unseen families, and unfamiliar wording. KS and Wasserstein are conditional on
valid numeric responses and retain numeric support violations.

Validation mean histogram KL changed from 0.070539 to 0.065246 (7.50% lower).
The held-out Weibull/Poisson aggregate did not improve on median KS or Wasserstein:
KS changed from 0.169606 to 0.175837; Wasserstein from 0.141 to 0.142.
Thus transfer to unseen families is mixed rather than uniformly beneficial.

The paired bootstrap resamples the 100 test prompts 10,000 times (seed 90210).
The mean KL difference has 95% interval {kl["paired_bootstrap_mean_difference_95pct"]}.
This exploratory interval reflects prompt variability in this one experiment,
not variability across training seeds. The median KL and median Wasserstein
intervals include zero. There was one training seed, and no control receiving
the same additional compute as supervised fine-tuning. These results support a
small incremental improvement in this run, not general superiority of RL.

Training job time: 22m34s. Evaluation: 5m43s. Benchmark: 2m20s.
The model remains well above the prior direct numerical-sampler reference
(median KS about 0.025 and normalized Wasserstein about 0.011).

![Comparison](rl-vs-sft.png)

- [RL training](https://forge.coreweave.com/wandb/yash2007-university-of-washington/stochastic-llm/runs/tx91cho0)
- [Final held-out evaluation](https://forge.coreweave.com/wandb/yash2007-university-of-washington/stochastic-llm/runs/n9d9aegn)
"""
    (output / "RESULTS.md").write_text(report)
    plt.rcParams.update(
        {
            "font.family": "DejaVu Sans",
            "font.size": 10,
            "axes.spines.top": False,
            "axes.spines.right": False,
        }
    )
    fig, axes = plt.subplots(1, 2, figsize=(11, 4.8), layout="constrained")
    changes = [
        kl["relative_mean_reduction_percent"],
        kl["relative_median_reduction_percent"],
        ks["relative_median_reduction_percent"],
        w["relative_median_reduction_percent"],
    ]
    ax = axes[0]
    bars = ax.barh(
        ["Mean histogram KL", "Median histogram KL", "Median KS", "Median Wasserstein"],
        changes,
        color="#187d9b",
        height=0.6,
    )
    ax.bar_label(bars, labels=[f"{v:.1f}%" for v in changes], padding=4)
    ax.set_xlim(0, 15)
    ax.invert_yaxis()
    ax.set_xlabel("Error reduction relative to SFT (%)")
    ax.set_title("Overall test improvement", loc="left", fontweight="bold")
    ax = axes[1]
    families = [
        "normal",
        "uniform",
        "exponential",
        "beta",
        "gamma",
        "binomial",
        "weibull",
        "poisson",
    ]
    delta = [
        100
        * (
            summaries["rl"][f"family:{f}"]["median_ks_statistic"]
            / summaries["sft"][f"family:{f}"]["median_ks_statistic"]
            - 1
        )
        for f in families
    ]
    bars = ax.barh(
        [f.title() + ("*" if f in ("weibull", "poisson") else "") for f in families],
        delta,
        color=["#187d9b" if d < 0 else "#a45c70" for d in delta],
    )
    ax.axvline(0, color="#444444", lw=1)
    ax.bar_label(bars, labels=[f"{v:+.1f}%" for v in delta], padding=3, fontsize=9)
    ax.set_xlim(-22, 12)
    ax.invert_yaxis()
    ax.set_xlabel("Change in median KS (%) — negative is better")
    ax.set_title("Gains vary by family", loc="left", fontweight="bold")
    fig.suptitle(
        "KL-reward RL adds a modest improvement over SFT\n"
        "100 prompts × 1,000 generations per model; * held-out families",
        fontsize=14,
    )
    fig.savefig(output / "rl-vs-sft.png", dpi=160)
    plt.close(fig)
    print(json.dumps(statistics, indent=2))


def read_jsonl_results(path):
    return json.loads(Path(path).read_text())


if __name__ == "__main__":
    main()
