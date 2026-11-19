"""Reproduce passage results from committed samples; no models, GPU, or network."""

import argparse
import json
import math
import statistics
from collections import defaultdict
from pathlib import Path

import matplotlib
import numpy as np

matplotlib.use("Agg")
import matplotlib.pyplot as plt
from passage_reward import PassageDistribution
from pilot import duplicate_window_rate, read_rows, words, write_json

REPOSITORY = Path(__file__).resolve().parents[2]
METRICS = {
    "bigram_kl": "Pooled bigram KL",
    "coherence": "Judge coherence /5",
    "fluency": "Judge fluency /5",
    "prompt_fulfillment": "Judge prompt fulfillment /5",
    "quality_reward_mean": "Combined quality reward /1",
    "mean_word_count": "Mean words per passage",
    "unknown_bigram_rate": "Out-of-reference bigram rate",
}


def analyze(results, output):
    source = results / "ddp"
    corpus = results / "corpus"
    recorded = json.loads((source / "comparison.json").read_text())
    target = PassageDistribution((corpus / "reference.txt").read_text().split("\n\n"))
    corpus_tokens = words((corpus / "tinyshakespeare.txt").read_text())
    windows = {tuple(corpus_tokens[i : i + 12]) for i in range(len(corpus_tokens) - 11)}
    rows, counts, metrics = {}, {}, {}
    for label in ("baseline", "final"):
        rows[label] = read_rows(source / f"{label}-samples.jsonl")
        samples = rows[label]
        counts[label] = target.counts([r["passage"] for r in samples]).sum(0).astype(np.int64)
        quality = [
            (statistics.mean(r["judge"]["scores"].values()) - 1) / 4 * r["terminated"]
            for r in samples
        ]
        for row, reward in zip(samples, quality):
            if not math.isclose(row["quality_reward"], reward, abs_tol=1e-8):
                raise ValueError(f"Saved reward disagrees with rubric arithmetic: {row['id']}")
        metric = {
            "bigram_kl": target.kl(counts[label]),
            "quality_reward_mean": statistics.mean(quality),
            "unknown_bigram_rate": float(counts[label][-1] / counts[label].sum()),
            "passages": len(samples),
            "terminated_rate": statistics.mean(r["terminated"] for r in samples),
            "mean_word_count": statistics.mean(len(words(r["passage"])) for r in samples),
            "corpus_12word_overlap": statistics.mean(
                duplicate_window_rate(r["passage"], windows) or 0.0 for r in samples
            ),
        }
        for key in ("coherence", "fluency", "prompt_fulfillment"):
            metric[key] = statistics.mean(r["judge"]["scores"][key] for r in samples)
        for key, value in metric.items():
            if not math.isclose(value, recorded[label][key], rel_tol=1e-9, abs_tol=1e-9):
                raise ValueError(f"Recorded aggregate mismatch: {label}/{key}")
        metrics[label] = metric
    if {r["id"] for r in rows["baseline"]} != {r["id"] for r in rows["final"]}:
        raise ValueError("Baseline/final sample identities differ")
    rng = np.random.default_rng(20261119)
    matched_count = int(min(c.sum() for c in counts.values()))
    matched = {}
    for label, count in counts.items():
        indices = np.flatnonzero(count)
        samples = []
        for _ in range(200):
            chosen = rng.multivariate_hypergeometric(count[indices], matched_count)
            subsampled = np.zeros_like(count)
            subsampled[indices] = chosen
            samples.append(target.kl(subsampled))
        matched[label] = statistics.mean(samples)
    case_means = {}
    for label, samples in rows.items():
        grouped = defaultdict(list)
        for row in samples:
            grouped[row["case_id"]].append(row["quality_reward"])
        case_means[label] = {key: statistics.mean(value) for key, value in grouped.items()}
    case_ids = sorted(case_means["baseline"])
    delta = np.array([case_means["final"][key] - case_means["baseline"][key] for key in case_ids])
    indices = rng.integers(0, len(case_ids), size=(10000, len(case_ids)))
    diagnostics = {
        "seed": 20261119,
        "matched_bigram_count": matched_count,
        "original_bigram_counts": {key: int(value.sum()) for key, value in counts.items()},
        "resamples": 200,
        "matched_count_mean_kl": matched,
        "judge_quality_prompts_improved": int((delta > 0).sum()),
        "judge_quality_prompts_worsened": int((delta < 0).sum()),
        "judge_quality_prompt_count": len(case_ids),
        "paired_prompt_bootstrap_judge_quality_delta_95pct": np.quantile(
            delta[indices].mean(1), [0.025, 0.975]
        ).tolist(),
        "caveats": [
            "Matched-count subsampling ignores passage structure and does not remove KL bias.",
            "Bootstrap reflects prompt variation within one training seed and one judge.",
            "Judge scores are not human validation.",
        ],
    }
    output.mkdir(parents=True, exist_ok=True)
    write_json(output / "recomputed-metrics.json", metrics)
    write_json(output / "posthoc-diagnostics.json", diagnostics)
    validation = sorted(
        (
            (int(path.name.split("-")[1]), json.loads(path.read_text()))
            for path in source.glob("validation-*-metrics.json")
        ),
        key=lambda item: item[0],
    )
    plot_validation(validation, output)
    before, after = metrics["baseline"], metrics["final"]
    reduction = 100 * (1 - after["bigram_kl"] / before["bigram_kl"])
    lines = [
        "# Completed two-H200 passage RL results",
        "",
        "The run completed 1,200 updates and 19,200 training passages in 1h41m21s",
        "on two H200s (approximately 3.38 GPU-hours). It stopped at its update limit.",
        "Both policies were evaluated on the same 16 held-out scenarios, four draws",
        "per scenario, for 64 passages per policy. Decoding and per-rank seeds match.",
        "",
        "| Metric | Before RL | After RL |",
        "|---|---:|---:|",
    ]
    for key, label in METRICS.items():
        lines.append(f"| {label} | {before[key]:.4f} | {after[key]:.4f} |")
    lines += [
        "",
        f"Raw pooled histogram KL fell {reduction:.2f}%. Mean judge quality improved",
        f"on {diagnostics['judge_quality_prompts_improved']}/{len(case_ids)} prompts.",
        "Both models terminated all 64 test responses; this quality gain is not the",
        "short experiment's truncation artifact. Source-overlapping 12-word windows",
        "were absent from both sets; this limited check does not rule out memorization.",
        "",
        "## Count sensitivity and uncertainty",
        "",
        "The trained model produces about 19.2% more words. Subsampling the pooled",
        f"bigrams to {matched_count:,} observations without replacement (200 repeats)",
        f"gives mean KL {matched['baseline']:.4f} → {matched['final']:.4f}.",
        "This is a post hoc count-sensitivity diagnostic, not a bias correction.",
        "",
        "A paired bootstrap over the 16 prompt-mean quality rewards gives an",
        "exploratory 95% mean-improvement interval of "
        f"{diagnostics['paired_prompt_bootstrap_judge_quality_delta_95pct']}.",
        "It reflects prompt variation within this one seed and judge, not training-seed",
        "variability or human writing quality.",
        "",
        "## Interpretation",
        "",
        "This is a modest positive result under the frozen judge and a small reduction",
        "in empirical word-bigram mismatch. One seed, one shared reward/evaluation judge,",
        "16 templated held-out scenarios, longer outputs, and sparse estimated KL limit",
        "the conclusion. Some passages remain awkward. It does not establish exact",
        "Shakespeare sampling, independent human-rated improvement, or superiority",
        "to continued SFT or MaD-RL. The broader prompt set also prevents attributing",
        "differences from the short pilot solely to additional training time.",
        "",
        "## Validation trajectories",
        "",
        "Validation uses 32 passages; its absolute KL is not directly comparable",
        "to the 64-passage final test.",
        "",
        "![Validation trajectories](validation-trends.png)",
        "",
        "## First saved matched example",
        "",
        "Prompt: " + rows["baseline"][0]["prompt"],
        "",
    ]
    for label in ("baseline", "final"):
        lines += [f"### {label.title()}", "", rows[label][0]["passage"], ""]
    run_url = json.loads((source / "wandb_run.json").read_text())["url"]
    lines += [f"[Training and final evaluation on W&B]({run_url})", ""]
    (output / "RESULTS.md").write_text("\n".join(lines))
    print(
        f"Verified recorded metrics. KL reduction: {reduction:.2f}%. "
        f"Report: {output / 'RESULTS.md'}"
    )


def plot_validation(validation, output):
    fig, axes = plt.subplots(1, 2, figsize=(10, 3.7), layout="constrained")
    for ax, key, title in zip(
        axes,
        ["quality_reward_mean", "bigram_kl"],
        [
            "Validation judge quality — higher is better",
            "Validation histogram KL — lower is better",
        ],
    ):
        ax.plot(
            [step for step, _ in validation],
            [value[key] for _, value in validation],
            marker="o",
            color="#16758f",
            lw=1.8,
        )
        ax.set_xlabel("RL update")
        ax.set_title(title, fontsize=10)
        ax.spines[["top", "right"]].set_visible(False)
        ax.grid(alpha=0.15)
    fig.suptitle("Two H200s · 1,200 updates · 19,200 training passages", fontsize=12)
    fig.savefig(output / "validation-trends.png", dpi=160)
    plt.close(fig)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--results", type=Path, default=REPOSITORY / "results/passage-rl-v1")
    parser.add_argument("--output", type=Path, default=REPOSITORY / "outputs/passage-analysis")
    args = parser.parse_args()
    analyze(args.results.resolve(), args.output.resolve())
