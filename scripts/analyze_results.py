"""Reproduce the first-run report and a matched-size numerical-sampler reference.

This reads completed evaluations only; it does not train or allocate any GPUs.
"""

import argparse
import json
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

from stochastic_llm.distributions import distribution, format_number
from stochastic_llm.io import read_jsonl, write_json
from stochastic_llm.metrics import score_samples


def analyze(root, repeats=20):
    root = Path(root)
    output = root / "outputs/results-analysis"
    output.mkdir(parents=True, exist_ok=True)
    cases = read_jsonl(root / "results/sft-v1/test-cases.jsonl")
    summaries = {}
    results = {}
    for name in ("base", "sft"):
        folder = root / f"results/sft-v1/{name}"
        summaries[name] = json.loads((folder / "summary.json").read_text())
        results[name] = {r["id"]: r for r in json.loads((folder / "results.json").read_text())}
    references = []
    for index, case in enumerate(cases):
        count = results["sft"][case["id"]]["valid"]
        assert count > 0
        for repeat in range(repeats):
            rng = np.random.default_rng(np.random.SeedSequence([7391, index, repeat]))
            values = distribution(case).rvs(size=count, random_state=rng)
            text = [format_number(x, case["family"], 5) for x in values]
            rows = [{"value": float(x), "text": x, "valid": True, "terminated": True} for x in text]
            score = score_samples(case, rows, decimals=5)
            references.append(
                {"id": case["id"], "family": case["family"], "repeat": repeat, **score}
            )
    reference_summary = {
        "repeats_per_case": repeats,
        "seed": 7391,
        "sample_sizes": "Matched to each SFT case's valid response count",
        "median_ks": float(np.median([r["ks_statistic"] for r in references])),
        "median_normalized_wasserstein": float(
            np.median([r["normalized_wasserstein"] for r in references])
        ),
    }
    report = {
        "model": "Qwen/Qwen2.5-3B",
        "training_objective": "sampled-target SFT",
        "training_examples": 64000,
        "training_updates": 2000,
        "test_prompts": 100,
        "generations_per_prompt_per_model": 1000,
        "base": summaries["base"],
        "sft": summaries["sft"],
        "scipy_reference": reference_summary,
    }
    for name in ("base", "sft"):
        rows = list(results[name].values())
        report[name]["all"]["support_violation_rate_among_valid"] = sum(
            r["support_violation_rate"] * r["valid"]
            for r in rows
            if r["support_violation_rate"] is not None
        ) / sum(r["valid"] for r in rows)
        report[name]["all"]["truncated_rate"] = sum(
            r["truncated_rate"] * r["requested"] for r in rows
        ) / sum(r["requested"] for r in rows)
    write_json(output / "report.json", report)
    write_json(output / "scipy-reference.json", references)

    plt.rcParams.update(
        {
            "font.size": 10,
            "axes.spines.top": False,
            "axes.spines.right": False,
            "figure.facecolor": "white",
        }
    )
    fig, axes = plt.subplots(2, 2, figsize=(13, 9))
    colors = {"base": "#9b6571", "sft": "#187d9b", "reference": "#4d9660"}
    ax = axes[0, 0]
    validity = [100 * summaries[name]["all"]["numeric_valid_rate"] for name in ("base", "sft")]
    bars = ax.bar(
        ["Pretrained base", "After SFT"],
        validity,
        color=[colors["base"], colors["sft"]],
        width=0.55,
    )
    ax.bar_label(bars, labels=[f"{v:.2f}%" for v in validity], padding=4, fontsize=12)
    ax.set_ylim(0, 110)
    ax.set_ylabel("Numeric-only responses, with EOS (%)")
    ax.set_title("Formatting / response validity", loc="left", fontweight="bold")
    ax.text(
        0.03,
        0.78,
        "Base: 88.63% hit the token limit\nSFT: 0.65% hit the token limit",
        transform=ax.transAxes,
        fontsize=10,
    )

    ax = axes[0, 1]
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
    x = np.arange(len(families))
    for offset, name, label in ((-0.27, "base", "Base"), (0, "sft", "SFT")):
        values = [summaries[name][f"family:{family}"]["median_ks_statistic"] for family in families]
        ax.bar(x + offset, values, width=0.25, color=colors[name], label=label)
    reference_values = [
        float(np.median([r["ks_statistic"] for r in references if r["family"] == family]))
        for family in families
    ]
    ax.bar(
        x + 0.27, reference_values, width=0.25, color=colors["reference"], label="SciPy reference"
    )
    ax.set_xticks(
        x,
        [f.title() + ("*" if f in ("weibull", "poisson") else "") for f in families],
        rotation=35,
        ha="right",
    )
    ax.set_ylabel("Median KS distance (lower is better)")
    ax.set_title("Distribution calibration among valid responses", loc="left", fontweight="bold")
    ax.legend(frameon=False, fontsize=9)

    for ax, family in zip(axes[1], ("normal", "poisson")):
        # First test case in each family, chosen before inspecting its performance.
        case = next(case for case in cases if case["family"] == family)
        law = distribution(case)
        low, high = law.ppf([0.001, 0.999])
        margin = max(0.05 * (high - low), 0.1)
        grid = np.linspace(low - margin, high + margin, 2000)
        ax.plot(grid, law.cdf(grid), color="#202c3c", lw=2, label="Requested CDF")
        for name, label in (("base", "Base"), ("sft", "SFT")):
            rows = read_jsonl(root / f"results/sft-v1/{name}/samples/{case['id']}.jsonl")
            values = np.sort([r["value"] for r in rows if r["valid"]])
            empirical = np.searchsorted(values, grid, side="right") / len(values)
            ax.plot(
                grid,
                empirical,
                color=colors[name],
                lw=1.7,
                label=f"{label} (n={len(values)} valid)",
            )
        ax.set_xlim(grid[0], grid[-1])
        ax.set_ylim(-0.03, 1.03)
        ax.set_xlabel("Generated value")
        ax.set_ylabel("Cumulative probability")
        ax.set_title(f"{family.title()} example: {case['parameters']}", loc="left", fontsize=10)
        ax.legend(frameon=False, fontsize=9)
        ax.grid(alpha=0.12)
    fig.suptitle(
        "Qwen2.5-3B: first sampling experiment", x=0.07, ha="left", fontsize=18, fontweight="bold"
    )
    fig.text(
        0.07,
        0.935,
        "64,000 training examples | 2,000 LoRA updates | "
        "100 held-out prompts × 1,000 generations per model",
        fontsize=11,
        color="#465364",
    )
    fig.text(
        0.07,
        0.025,
        "* Held-out families. Distances use valid responses only; "
        "4 base prompts had no valid outputs.\n"
        f"SciPy reference: {repeats} seeds/case, matched SFT valid sample counts. "
        "Single training seed; no independence test.",
        fontsize=9,
        color="#465364",
    )
    fig.subplots_adjust(top=0.88, bottom=0.13, hspace=0.5, wspace=0.27)
    fig.savefig(output / "sampling-results.png", dpi=170)
    fig.savefig(output / "sampling-results.pdf")
    plt.close(fig)
    print(
        json.dumps(
            {"reference": reference_summary, "figure": str(output / "sampling-results.png")},
            indent=2,
        )
    )


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--root", default=".")
    parser.add_argument("--reference-repeats", type=int, default=20)
    args = parser.parse_args()
    analyze(args.root, args.reference_repeats)
