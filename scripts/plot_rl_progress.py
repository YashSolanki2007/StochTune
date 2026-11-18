"""Plot an interim RL log and simulate its finite-batch KL reference on CPU."""

import json
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

from stochastic_llm.io import read_config, read_jsonl, write_json
from stochastic_llm.kl_reward import HistogramTarget


def main():
    output = Path("outputs/rl-progress")
    config = read_config("configs/batch-kl-rl-h200.yaml")
    cases = read_jsonl("data/numeric-v1/validation_configs.jsonl")[: config["validation_cases"]]
    rng = np.random.default_rng(912431)
    reference = []
    for case in cases:
        target = HistogramTarget.from_case(case, config["reward_bins"], config["decimals"])
        p = target.probabilities
        n, alpha = config["validation_samples_per_case"], config["pseudocount"]
        counts = rng.multinomial(n, p, size=2000)
        q = (counts + alpha) / (n + len(p) * alpha)
        positive = p > 0
        reference.append((p[positive] * np.log(p[positive] / q[:, positive])).sum(1))
    means = np.array(reference).mean(0)
    rows = read_jsonl(output / "history.jsonl")
    train = [r for r in rows if "rl/target_histogram_kl" in r]
    val = [r for r in rows if "validation/mean_histogram_kl" in r]
    summary = {
        "latest_update": train[-1]["rl/update"],
        "validation": val,
        "perfect_sampler_reference_mean_kl": float(means.mean()),
        "perfect_sampler_reference_95pct_range": np.quantile(means, [0.025, 0.975]).tolist(),
    }
    for key in ("mean_histogram_kl", "median_ks", "median_normalized_wasserstein"):
        field = f"validation/{key}"
        summary[f"{key}_change_percent"] = 100 * (val[-1][field] / val[0][field] - 1)
    write_json(output / "summary.json", summary)
    x = np.array([r["rl/update"] for r in train])
    vx = np.array([r["rl/update"] for r in val])
    plt.rcParams.update({"font.size": 10, "axes.spines.top": False, "axes.spines.right": False})
    fig, axes = plt.subplots(2, 2, figsize=(12, 7.8))
    window = min(20, len(train))
    ax = axes[0, 0]
    y = np.array([r["rl/target_histogram_kl"] for r in train])
    ax.plot(x, y, color="#bed3e1", lw=1, label="Fresh batches")
    ax.plot(
        x[window - 1 :],
        np.convolve(y, np.ones(window) / window, mode="valid"),
        color="#176a91",
        lw=2,
        label=f"{window}-update mean",
    )
    ax.set_title("Training KL", loc="left", fontweight="bold")
    ax.set_ylabel("Smoothed histogram KL")
    ax.legend(frameon=False)
    ax = axes[0, 1]
    lo, hi = summary["perfect_sampler_reference_95pct_range"]
    ax.axhspan(lo, hi, color="#e5eee5", label="Perfect-sampler 95% simulation range")
    ax.axhline(means.mean(), color="#609263", ls="--", label="Perfect-sampler mean")
    ax.plot(
        vx,
        [r["validation/mean_histogram_kl"] for r in val],
        "o-",
        color="#176a91",
        lw=2,
        label="Model validation KL",
    )
    ax.set_title("Validation KL", loc="left", fontweight="bold")
    ax.set_ylabel("Mean histogram KL across 12 prompts")
    ax.legend(frameon=False, fontsize=8.5, loc="lower left")
    ax = axes[1, 0]
    for field, label, color in (
        ("median_ks", "KS error", "#95516c"),
        ("median_normalized_wasserstein", "Wasserstein error", "#176a91"),
    ):
        y = np.array([r[f"validation/{field}"] for r in val])
        ax.plot(vx, 100 * y / y[0], "o-", label=label, color=color, lw=2)
    ax.axhline(100, color="#888888", lw=1, ls="--")
    ax.set_ylabel("Error as % of its starting value (lower is better)")
    ax.set_title("Validation distances", loc="left", fontweight="bold")
    ax.legend(frameon=False)
    ax = axes[1, 1]
    y = np.array([r["rl/gradient_norm"] for r in train])
    ax.plot(x, y, color="#d3b8c3", lw=1, label="Gradient norm before clipping")
    ax.plot(
        x[window - 1 :],
        np.convolve(y, np.ones(window) / window, mode="valid"),
        color="#95516c",
        lw=2,
        label=f"{window}-update mean",
    )
    ax.axhline(1, color="#777777", ls="--", label="Clipping threshold")
    ax.set_title("Gradient stability", loc="left", fontweight="bold")
    ax.set_ylabel("Gradient norm")
    ax.legend(frameon=False, fontsize=8.5)
    for ax in axes.flat:
        ax.set_xlabel("RL update")
        ax.grid(alpha=0.12)
    fig.suptitle(
        f"KL-reward RL: curves through update {int(x[-1])}",
        x=0.075,
        ha="left",
        fontsize=17,
        fontweight="bold",
    )
    fig.text(
        0.075,
        0.922,
        f"Validation through update {int(vx[-1])}; full budget 250 updates. "
        "These are validation curves, not final test results.",
        fontsize=10,
        color="#465364",
    )
    fig.text(
        0.075,
        0.025,
        "Reference: exact same bins, 256 draws/prompt, alpha=0.1, 12 validation "
        "prompts, 2,000 repetitions.\nThe simulation band measures finite-batch variability; "
        "it is not a confidence interval on RL improvement.",
        fontsize=8.5,
        color="#465364",
    )
    fig.subplots_adjust(top=0.86, bottom=0.14, hspace=0.42, wspace=0.25)
    fig.savefig(output / "training-curves.png", dpi=160)
    plt.close(fig)
    print(
        json.dumps({key: value for key, value in summary.items() if key != "validation"}, indent=2)
    )


if __name__ == "__main__":
    main()
