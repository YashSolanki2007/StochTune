"""Recompute unigram/bigram comparisons from recorded samples without a GPU."""

import json
import sys
from collections import defaultdict
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "experiments/shakespeare-quality-pilot"))
from passage_reward import PassageDistribution  # noqa: E402
from pilot import read_rows  # noqa: E402


def grouped_quality(records):
    grouped = defaultdict(list)
    for row in records:
        grouped[row["case_id"]].append(row["quality_reward"])
    return {key: np.mean(values) for key, values in grouped.items()}


def main():
    previous = ROOT / "results/passage-rl-v1"
    current = ROOT / "results/passage-unigram-v1"
    reference = (previous / "corpus/reference.txt").read_text().split("\n\n")
    rows = {
        "baseline": read_rows(current / "baseline-samples.jsonl"),
        "bigram_trained": read_rows(previous / "ddp/final-samples.jsonl"),
        "unigram_trained": read_rows(current / "final-samples.jsonl"),
    }
    assert [r["passage"] for r in rows["baseline"]] == [
        r["passage"] for r in read_rows(previous / "ddp/baseline-samples.jsonl")
    ]
    recorded = json.loads((current / "final-diagnostics.json").read_text())
    report = {
        "resamples": 200,
        "seed": 20261119,
        "distribution_metrics": {},
        "quality_comparisons": {},
    }
    rng = np.random.default_rng(report["seed"])
    for order in (1, 2):
        target = PassageDistribution(reference, order=order)
        counts = {
            key: target.counts([r["passage"] for r in records]).sum(0).astype(np.int64)
            for key, records in rows.items()
        }
        n = min(int(c.sum()) for c in counts.values())
        result = {"matched_count": n, "policies": {}}
        for key, count in counts.items():
            nonzero = np.flatnonzero(count)
            values = []
            for _ in range(report["resamples"]):
                subsampled = np.zeros_like(count)
                subsampled[nonzero] = rng.multivariate_hypergeometric(count[nonzero], n)
                values.append(target.kl(subsampled))
            result["policies"][key] = {
                "observations": int(count.sum()),
                "raw_kl": target.kl(count),
                "matched_mean_kl": float(np.mean(values)),
                "unknown_rate": float(count[-1] / count.sum()),
            }
        report["distribution_metrics"][target.name] = result
    for a, b in (("unigram_trained", "baseline"), ("unigram_trained", "bigram_trained")):
        ga, gb = grouped_quality(rows[a]), grouped_quality(rows[b])
        assert set(ga) == set(gb)
        delta = np.array([ga[key] - gb[key] for key in sorted(ga)])
        sampled = rng.choice(delta, size=(10000, len(delta)), replace=True).mean(1)
        report["quality_comparisons"][a + "_minus_" + b] = {
            "mean_delta": float(delta.mean()),
            "prompts_improved": int((delta > 0).sum()),
            "prompts_worsened": int((delta < 0).sum()),
            "prompt_count": len(delta),
            "exploratory_prompt_bootstrap_95pct": np.quantile(sampled, [0.025, 0.975]).tolist(),
        }
    report["limitations"] = recorded["limitations"]
    assert report == recorded, "Recomputed diagnostics differ from the archive"
    output = ROOT / "outputs/unigram-analysis"
    output.mkdir(parents=True, exist_ok=True)
    path = output / "final-diagnostics.json"
    path.write_text(json.dumps(report, indent=2) + "\n")
    print(f"Reproduced recorded diagnostics: {path}")


if __name__ == "__main__":
    main()
