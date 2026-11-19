"""Visualize recorded word-bigram distributions; no model or GPU calls."""

import csv
import json
import sys
from pathlib import Path

import matplotlib
import numpy as np

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
from matplotlib.ticker import PercentFormatter  # noqa: E402

REPO = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO / "experiments/shakespeare-quality-pilot"))
from passage_reward import PassageDistribution  # noqa: E402
from pilot import pooled_counts, read_rows  # noqa: E402

OUTPUT = REPO / "outputs/bigram-fit-plots"
OUTPUT.mkdir(parents=True, exist_ok=True)
RESULTS = REPO / "results/passage-rl-v1"
COLORS = {"reference": "#263448", "baseline": "#CC8A32", "final": "#167D9B"}
LABELS = {"reference": "Shakespeare reference", "baseline": "Before RL", "final": "After RL"}
plt.rcParams.update(
    {
        "font.family": "DejaVu Sans",
        "font.size": 10,
        "axes.spines.top": False,
        "axes.spines.right": False,
        "figure.facecolor": "white",
        "savefig.facecolor": "white",
    }
)


def save(fig, name):
    fig.savefig(OUTPUT / f"{name}.png", dpi=180, bbox_inches="tight")
    fig.savefig(OUTPUT / f"{name}.pdf", bbox_inches="tight")
    plt.close(fig)


reference_text = (RESULTS / "corpus/reference.txt").read_text()
reference = pooled_counts(reference_text.split("\n\n"))
ranked = [pair for pair, _ in reference.most_common()]
model_counts = {}
sample_sizes = {}
for label in ("baseline", "final"):
    rows = read_rows(RESULTS / f"ddp/{label}-samples.jsonl")
    sample_sizes[label] = len(rows)
    model_counts[label] = pooled_counts([row["passage"] for row in rows])
counts = {"reference": reference, **model_counts}
totals = {label: sum(c.values()) for label, c in counts.items()}
pmf = {label: np.array([c[pair] / totals[label] for pair in ranked]) for label, c in counts.items()}
unknown = {label: 1 - pmf[label].sum() for label in pmf}
for label in pmf:
    assert abs(pmf[label].sum() + unknown[label] - 1) < 1e-12

# Export the complete PMF comparison in target-frequency order, including UNKNOWN.
with (OUTPUT / "bigram-probabilities.csv").open("w", newline="") as handle:
    writer = csv.writer(handle)
    writer.writerow(
        [
            "reference_rank",
            "word_1",
            "word_2",
            "reference_count",
            "baseline_count",
            "final_count",
            "reference_probability",
            "baseline_probability",
            "final_probability",
        ]
    )
    for i, pair in enumerate(ranked):
        writer.writerow(
            [
                i + 1,
                *pair,
                *[counts[label][pair] for label in counts],
                *[pmf[label][i] for label in pmf],
            ]
        )
    writer.writerow(
        [
            "UNKNOWN",
            "",
            "",
            0,
            sum(n for pair, n in model_counts["baseline"].items() if pair not in reference),
            sum(n for pair, n in model_counts["final"].items() if pair not in reference),
            0,
            unknown["baseline"],
            unknown["final"],
        ]
    )

# Common-pair PMF: categories selected only from the fixed target, no renormalization.
top = 20
fig, ax = plt.subplots(figsize=(10.5, 8.3), layout="constrained")
y = np.arange(top)
for shift, label in zip((-0.25, 0, 0.25), ("reference", "baseline", "final")):
    ax.barh(
        y + shift,
        pmf[label][:top] * 100,
        height=0.23,
        label=LABELS[label],
        color=COLORS[label],
        alpha=0.95,
    )
ax.set_yticks(y, [" ".join(pair) for pair in ranked[:top]])
ax.invert_yaxis()
ax.set_xlabel("Probability mass (% of all word bigrams)")
ax.xaxis.set_major_formatter(PercentFormatter(xmax=100, decimals=2))
ax.set_xlim(0, max(pmf[label][:top].max() for label in pmf) * 100 * 1.12)
ax.grid(axis="x", alpha=0.15)
ax.set_axisbelow(True)
ax.legend(loc="lower right", frameon=False)
ax.set_title(
    "The 20 most common reference bigrams\n"
    "Same categories for every policy; probabilities are not renormalized",
    pad=14,
)
fig.supxlabel(
    f"Only {pmf['reference'][:top].sum():.2%} of reference mass is shown here. "
    "Model frequencies are estimates from 64 passages each.",
    fontsize=9,
)
save(fig, "common-bigram-pmf")

# Full support: target-frequency ordering is fixed for both models. UNKNOWN remains visible.
fig, axes = plt.subplots(
    1, 2, figsize=(12, 5), layout="constrained", gridspec_kw={"width_ratios": [1.65, 1]}
)
ax = axes[0]
rank_points = np.unique(np.r_[1, np.geomspace(1, len(ranked), 2200).astype(int), len(ranked)])
for label in ("reference", "baseline", "final"):
    cdf = np.cumsum(pmf[label])
    ax.plot(rank_points, cdf[rank_points - 1] * 100, color=COLORS[label], label=LABELS[label], lw=2)
    ax.scatter([len(ranked)], [cdf[-1] * 100], color=COLORS[label], s=22)
    ax.annotate(
        f"{cdf[-1]:.1%}",
        (len(ranked), cdf[-1] * 100),
        xytext=(6, -2),
        textcoords="offset points",
        color=COLORS[label],
        fontsize=9,
    )
ax.set_xscale("log")
ax.set_xlim(1, len(ranked) * 4)
ax.set_ylim(0, 105)
ax.set_xlabel("Reference bigram rank (most frequent → rarest; log scale)")
ax.set_ylabel("Cumulative probability mass (%)")
ax.set_title("Cumulative mass over all 83,454 reference pairs", fontsize=11)
ax.grid(alpha=0.15)
ax.legend(loc="upper left", frameon=False)
ax = axes[1]
x = np.arange(3)
known = np.array([1 - unknown[label] for label in pmf]) * 100
unseen = np.array([unknown[label] for label in pmf]) * 100
ax.bar(x, known, color="#91BCCD", label="Pair occurs in reference")
ax.bar(x, unseen, bottom=known, color="#BC7382", label="Pair absent from reference")
for i in range(3):
    ax.text(i, known[i] / 2, f"{known[i]:.1f}%", ha="center", va="center", color="#182B37")
    if unseen[i] > 1:
        ax.text(
            i,
            known[i] + unseen[i] / 2,
            f"{unseen[i]:.1f}%",
            ha="center",
            va="center",
            color="white",
        )
ax.set_xticks(x, ["Reference", "Before RL", "After RL"])
ax.set_ylim(0, 105)
ax.set_title("Where the empirical probability mass goes", fontsize=11)
ax.set_ylabel("Probability mass (%)")
ax.legend(loc="upper center", bbox_to_anchor=(0.5, -0.09), frameon=False, fontsize=9)
fig.suptitle("Whole-distribution fit: improvement, but a large mismatch remains", fontsize=13)
fig.supxlabel(
    "UNKNOWN means an unseen word pair in this reference, not invalid English. "
    "Bigrams are counted within passages.",
    fontsize=9,
)
save(fig, "full-bigram-fit")

# Match pair counts before comparing the sparse, smoothed histogram-KL estimator.
target = PassageDistribution(reference_text.split("\n\n"))
recorded = json.loads((RESULTS / "ddp/comparison.json").read_text())
vectors = {
    label: target.counts([r["passage"] for r in read_rows(RESULTS / f"ddp/{label}-samples.jsonl")])
    .sum(0)
    .astype(np.int64)
    for label in ("baseline", "final")
}
rng = np.random.default_rng(20261119)
n = min(int(c.sum()) for c in vectors.values())
matched = {}
for label, count in vectors.items():
    nonzero = np.flatnonzero(count)
    values = []
    for _ in range(200):
        subsampled = np.zeros_like(count)
        subsampled[nonzero] = rng.multivariate_hypergeometric(count[nonzero], n)
        values.append(target.kl(subsampled))
    matched[label] = values
    assert np.isclose(target.kl(count), recorded[label]["bigram_kl"])
fig, ax = plt.subplots(figsize=(7.7, 4.1), layout="constrained")
x = np.arange(2)
for shift, label in zip((-0.18, 0.18), ("baseline", "final")):
    values = [recorded[label]["bigram_kl"], float(np.mean(matched[label]))]
    bars = ax.bar(x + shift, values, width=0.34, color=COLORS[label], label=LABELS[label])
    ax.bar_label(bars, labels=[f"{value:.3f}" for value in values], padding=4)
ax.set_xticks(x, ["Original evaluation samples", f"Count-matched ({n:,} pairs each)"])
ax.set_ylabel("Smoothed empirical forward KL (lower is better)")
ax.set_ylim(0, 9.4)
ax.legend(frameon=False, loc="upper center", ncol=2)
ax.set_title("The KL improvement remains small after matching pair counts", fontsize=12)
ax.grid(axis="y", alpha=0.15)
ax.set_axisbelow(True)
fig.supxlabel(
    "Count-matched bars average 200 subsamples of the saved data. "
    "This diagnoses count sensitivity; it does not remove estimator bias.",
    fontsize=9,
)
save(fig, "count-matched-kl")
summary = {
    "reference_bigram_observations": totals["reference"],
    "reference_distinct_pairs": len(ranked),
    "model_bigram_observations": {k: totals[k] for k in model_counts},
    "model_passages": sample_sizes,
    "top20_reference_mass": float(pmf["reference"][:top].sum()),
    "unknown_rates": {k: float(v) for k, v in unknown.items()},
    "matched_count": n,
    "matched_kl": {k: float(np.mean(v)) for k, v in matched.items()},
    "notes": [
        "All plots show empirical marginals, not exact model probabilities.",
        "Reference order is fixed across policies.",
        "UNKNOWN pairs are not necessarily incorrect English.",
        "The target is the selected Tiny Shakespeare reference, not all Shakespeare.",
    ],
}
(OUTPUT / "plot-summary.json").write_text(json.dumps(summary, indent=2) + "\n")
print(json.dumps(summary, indent=2))
