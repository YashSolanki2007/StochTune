"""Generate manuscript tables and vector figures from committed experiment records."""

import json
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

from stochastic_llm.distributions import distribution
from stochastic_llm.io import read_jsonl
from stochastic_llm.metrics import cdf_sides

PAPER = Path(__file__).resolve().parent
ROOT = PAPER.parent
RECORDS = ROOT / "results/rl-v1"
TABLES = PAPER / "tables"
FIGURES = PAPER / "figures"


def write_table(name, rows):
    (TABLES / name).write_text(
        "\n".join(" & ".join(row) + r" \\" for row in rows) + "\n\\bottomrule\n"
    )


def main():
    TABLES.mkdir(parents=True, exist_ok=True)
    FIGURES.mkdir(parents=True, exist_ok=True)
    stats = json.loads((RECORDS / "analysis/comparison-statistics.json").read_text())
    summaries = json.loads((RECORDS / "analysis/evaluation-summaries.json").read_text())
    per_prompt = json.loads((RECORDS / "analysis/per-prompt-results.json").read_text())
    baseline = json.loads((ROOT / "results/sft-v1/report.json").read_text())
    rows = []
    labels = {
        "target_histogram_kl": "Histogram KL",
        "ks_statistic": "KS distance",
        "normalized_wasserstein": "Normalized Wasserstein",
    }
    for key, label in labels.items():
        s = stats[key]
        for agg in ("mean", "median"):
            rows.append(
                [
                    label,
                    agg.title(),
                    f"{s[f'sft_{agg}']:.6f}",
                    f"{s[f'rl_{agg}']:.6f}",
                    f"{s[f'relative_{agg}_reduction_percent']:.2f}\\%",
                ]
            )
    write_table("overall.tex", rows)
    rows = []
    for family in [
        "normal",
        "uniform",
        "exponential",
        "beta",
        "gamma",
        "binomial",
        "weibull",
        "poisson",
    ]:
        a, b = (summaries[m][f"family:{family}"] for m in ("sft", "rl"))
        rows.append(
            [
                family.title() + (r"$^\star$" if family in ("weibull", "poisson") else ""),
                str(a["prompts"]),
                f"{a['median_ks_statistic']:.4f}",
                f"{b['median_ks_statistic']:.4f}",
                f"{a['median_normalized_wasserstein']:.4f}",
                f"{b['median_normalized_wasserstein']:.4f}",
            ]
        )
    write_table("families.tex", rows)
    rows, group_kl = [], {}
    for category, label in [
        ("unseen_parameters", "Unseen parameters"),
        ("unseen_family", "Unseen families"),
        ("paraphrase", "New wording"),
    ]:
        a, b = (summaries[m][f"category:{category}"] for m in ("sft", "rl"))
        means = [
            np.mean(
                [
                    r["target_histogram_kl"]
                    for r in per_prompt[m].values()
                    if r["category"] == category
                ]
            )
            for m in ("sft", "rl")
        ]
        group_kl[category] = dict(zip(("sft_mean_kl", "rl_mean_kl"), means))
        rows.append(
            [
                label,
                str(a["prompts"]),
                *[f"{x:.4f}" for x in means],
                f"{a['median_ks_statistic']:.4f}",
                f"{b['median_ks_statistic']:.4f}",
                f"{a['median_normalized_wasserstein']:.4f}",
                f"{b['median_normalized_wasserstein']:.4f}",
            ]
        )
    write_table("categories.tex", rows)
    rows = []
    for key, label in labels.items():
        s = stats[key]
        for agg in ("mean", "median"):
            lo, hi = s[f"paired_bootstrap_{agg}_difference_95pct"]
            rows.append(
                [
                    label,
                    agg.title(),
                    f"{s[f'rl_{agg}'] - s[f'sft_{agg}']:+.6f}",
                    f"[{lo:+.6f}, {hi:+.6f}]",
                ]
            )
    write_table("bootstrap.tex", rows)
    history = read_jsonl(RECORDS / "training/history.jsonl")
    training = [r for r in history if "rl/target_histogram_kl" in r]
    validation = [r for r in history if "validation/mean_histogram_kl" in r]
    write_table(
        "validation.tex",
        [
            [
                str(r["rl/update"]),
                f"{r['validation/mean_histogram_kl']:.6f}",
                f"{r['validation/median_ks']:.6f}",
                f"{r['validation/median_normalized_wasserstein']:.6f}",
                f"{100 * r['validation/numeric_valid_rate']:.2f}\\%",
            ]
            for r in validation
        ],
    )
    quality = []
    for label, records in [
        ("Pretrained", ROOT / "results/sft-v1/base/results.json"),
        ("SFT", RECORDS / "sft-evaluation/results.json"),
        ("SFT + RL", RECORDS / "evaluation/results.json"),
    ]:
        r = json.loads(records.read_text())
        valid = sum(x["valid"] for x in r)
        violations = sum(
            round(x["support_violation_rate"] * x["valid"])
            for x in r
            if x["support_violation_rate"] is not None
        )
        requested = sum(x["requested"] for x in r)
        trunc = sum(round(x["truncated_rate"] * x["requested"]) for x in r)
        quality.append(
            [
                label,
                f"{valid:,}",
                f"{100 * valid / requested:.3f}\\%",
                f"{violations:,}",
                f"{100 * violations / valid:.3f}\\%",
                f"{100 * trunc / requested:.3f}\\%",
            ]
        )
    write_table("quality.tex", quality)
    plt.rcParams.update(
        {
            "font.family": "DejaVu Serif",
            "font.size": 9,
            "axes.spines.top": False,
            "axes.spines.right": False,
            "pdf.fonttype": 42,
            "savefig.bbox": "tight",
        }
    )
    colors = {"sft": "#31658a", "rl": "#b55335", "target": "#222222"}
    fig, axes = plt.subplots(1, 2, figsize=(6.6, 2.7), layout="constrained")
    step = np.array([r["rl/update"] for r in training])
    cost = np.array([r["rl/target_histogram_kl"] for r in training])
    axes[0].plot(step, cost, color="#bbc8d2", lw=0.7, label="Training (2 prompts/update)")
    window = 20
    axes[0].plot(
        step[window - 1 :],
        np.convolve(cost, np.ones(window) / window, "valid"),
        color=colors["sft"],
        lw=1.5,
        label="Training (20-update mean)",
    )
    axes[0].plot(
        [r["rl/update"] for r in validation],
        [r["validation/mean_histogram_kl"] for r in validation],
        "o-",
        color=colors["rl"],
        ms=3,
        lw=1.2,
        label="Fixed 12-case validation",
    )
    axes[0].set(xlabel="RL update", ylabel="Mean histogram KL", title="(a) Target mismatch")
    axes[0].legend(fontsize=6.3, loc="upper right", frameon=False)
    ratio = np.array([r["rl/reference_sequence_kl_estimate"] for r in training])
    axes[1].plot(step, ratio, color="#d8b7aa", lw=0.7, label="Sampled estimate")
    axes[1].plot(
        step[window - 1 :],
        np.convolve(ratio, np.ones(window) / window, "valid"),
        color=colors["rl"],
        lw=1.5,
        label="20-update mean",
    )
    axes[1].axhline(0, color="#777777", lw=0.5)
    axes[1].set(
        xlabel="RL update",
        ylabel="Mean sequence log-ratio (nats)",
        title="(b) Drift from the SFT reference",
    )
    axes[1].legend(fontsize=6.5, loc="upper left", frameon=False)
    fig.savefig(FIGURES / "training.pdf")
    plt.close(fig)
    cases = {c["id"]: c for c in read_jsonl(RECORDS / "test-cases.jsonl")}
    fig, axes = plt.subplots(2, 2, figsize=(6.6, 4.7), layout="constrained")
    selected = ["test-00000", "test-00003", "test-00001", "test-00004"]
    for ax, case_id in zip(axes.flat, selected):
        case = cases[case_id]
        law = distribution(case)
        lo, hi = law.ppf([0.001, 0.999])
        grid = np.linspace(lo, hi, 1400)
        ax.plot(
            grid, cdf_sides(case, grid)[1], color=colors["target"], lw=1.4, ls="--", label="Target"
        )
        for method, folder in [("sft", "sft-evaluation"), ("rl", "evaluation")]:
            rows = read_jsonl(RECORDS / folder / "samples" / f"{case_id}.jsonl")
            values = np.sort([r["value"] for r in rows if r["valid"]])
            ax.plot(
                grid,
                np.searchsorted(values, grid, side="right") / len(values),
                color=colors[method],
                lw=1.2,
                label="SFT" if method == "sft" else "SFT + RL",
            )
        ax.set(
            title=f"{case['family'].title()} ({case_id})",
            xlabel="Value",
            ylabel="CDF",
            xlim=(lo, hi),
            ylim=(-0.02, 1.02),
        )
        ax.legend(fontsize=6.5, loc="lower right", frameon=False)
    fig.savefig(FIGURES / "ecdf.pdf")
    plt.close(fig)
    summary = {
        "statistics": stats,
        "category_mean_kl": group_kl,
        "scipy_reference": baseline["scipy_reference"],
        "illustrative_case_ids": selected,
    }
    (PAPER / "numerical-summary.json").write_text(json.dumps(summary, indent=2) + "\n")
    print("Generated six numerical tables, quality table, and two vector figures.")


if __name__ == "__main__":
    main()
