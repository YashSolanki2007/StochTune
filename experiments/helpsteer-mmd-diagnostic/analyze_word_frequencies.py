"""CPU audit of actual saved samples; no new model generation or GPU allocation."""

import argparse
import hashlib
import json
import sys
from collections import Counter, defaultdict
from pathlib import Path

import matplotlib
import numpy as np
from scipy.spatial.distance import jensenshannon

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402

REPO = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(REPO / "experiments/shakespeare-quality-pilot"))
sys.path.insert(0, str(REPO / "experiments/helpsteer-rl-v2"))
from passage_reward import PassageDistribution  # noqa: E402
from pilot import words  # noqa: E402
from word_distribution import BucketedWordDistribution  # noqa: E402

# Display selection only: ALL words participate in every numerical metric.
STOP = set(
    "a an the and or but if then to of for in on at by with from as is are was were be been "
    "being it its this that these those you your yours i we they their our can could should "
    "would will may might must do does did not no yes have has had also some any each all "
    "more most other such than both which what how when where there here into about through "
    "using use used like example examples including following need one two make".split()
)


def read_rows(path):
    return [json.loads(line) for line in path.read_text().splitlines() if line.strip()]


def count(passages):
    return Counter(w for passage in passages for w in words(passage))


def compare(target_counts, generated):
    vocabulary = sorted(target_counts)
    p = np.array([target_counts[w] for w in vocabulary] + [0], dtype=float)
    p /= p.sum()
    counts = np.array(
        [generated[w] for w in vocabulary]
        + [sum(n for w, n in generated.items() if w not in target_counts)],
        dtype=float,
    )
    if counts.sum() == 0:
        counts[-1] = 1
    q = counts / counts.sum()
    # Same total pseudocount mass as the historical word-frequency KL metric.
    smooth = (counts + 1 / len(counts)) / (counts.sum() + 1)
    mask = p > 0
    return {
        "forward_kl": float(np.sum(p[mask] * np.log(p[mask] / smooth[mask]))),
        "jensen_shannon": float(jensenshannon(p, q) ** 2),
        "total_variation": float(0.5 * np.abs(p - q).sum()),
        "outside_reference_word_mass": float(q[-1]),
    }


def mean_question_pmf(groups):
    result = Counter()
    for passages in groups.values():
        counts = count(passages)
        total = sum(counts.values())
        for w, n in counts.items():
            result[w] += n / total / len(groups)
    return result


def analyze(source, output):
    output.mkdir(parents=True, exist_ok=True)
    data = source / ("data" if (source / "data").exists() else "reference-inputs")
    samples = source / "samples" if (source / "samples").exists() else source
    refs = {r["case_id"]: r for r in read_rows(data / "semantic-references.jsonl")}
    broad_refs = [r["response"] for r in read_rows(data / "reference.jsonl")]
    pooled_target = BucketedWordDistribution(broad_refs, max_words=10000, min_count=5)
    full_target = PassageDistribution(broad_refs, order=1)
    report = {
        "source_job_id": "510187",
        "fresh_generation": False,
        "gpu_hours_used_for_audit": 0,
        "source_file_sha256": {},
        "scopes": {},
        "protocol": "All words lowercased using the original project tokenizer; punctuation "
        "excluded. Exact-question target counts pool its two reference answers; question "
        "metrics are averaged equally. Forward KL smooths generated counts with total "
        "pseudocount mass 1. JS/TV use unsmoothed PMFs and an OTHER category. The plot "
        "averages normalized word PMFs equally across questions; stopwords omitted from "
        "display selection only. Reference pooling weights answers by their word counts, "
        "rather than the equal-answer weighting of the embedding objective.",
        "limitations": [
            "Only four seen and four unseen questions, one training seed.",
            "Only two references per question; absent words are not necessarily bad.",
            "These saved before/after evaluations are not fresh independent replications.",
            "Small changes can reflect sample noise and KL smoothing.",
        ],
    }
    fig, axes = plt.subplots(1, 2, figsize=(13, 7.2), sharex=False)
    snippets = []
    for ax, scope in zip(axes, ("seen", "unseen")):
        groups = {}
        records = {}
        for label in ("baseline", "final"):
            path = samples / f"{label}-{scope}-samples.jsonl"
            report["source_file_sha256"][str(path.relative_to(source))] = hashlib.sha256(
                path.read_bytes()
            ).hexdigest()
            records[label] = read_rows(path)
            grouped = defaultdict(list)
            for r in records[label]:
                grouped[r["case_id"]].append(r["passage"])
            groups[label] = grouped
        ids = sorted(groups["baseline"])
        assert set(ids) == set(groups["final"])
        assert all(len(groups["baseline"][q]) == len(groups["final"][q]) for q in ids)
        questions = []
        for q in ids:
            target_counts = count(refs[q]["responses"])
            questions.append(
                {
                    "case_id": q,
                    "prompt": refs[q]["prompt"],
                    **{
                        label: compare(target_counts, count(groups[label][q]))
                        for label in ("baseline", "final")
                    },
                }
            )
        means = {
            label: {
                key: float(np.mean([r[label][key] for r in questions]))
                for key in questions[0][label]
            }
            for label in ("baseline", "final")
        }
        historical = {}
        for label in ("baseline", "final"):
            passages = [r["passage"] for r in records[label]]
            historical[label] = {
                "bucketed_unigram_kl": pooled_target.kl(pooled_target.counts(passages).sum(0)),
                "full_vocabulary_unigram_kl": full_target.kl(full_target.counts(passages).sum(0)),
            }
            saved = json.loads((samples / f"{label}-{scope}-metrics.json").read_text())
            for key, value in historical[label].items():
                if abs(value - saved[key]) > 1e-9:
                    raise ValueError("Historical metric replay mismatch")
        p = mean_question_pmf({q: refs[q]["responses"] for q in ids})
        pmfs = {
            "reference": p,
            **{label: mean_question_pmf(groups[label]) for label in ("baseline", "final")},
        }
        top = sorted((w for w in p if w not in STOP), key=lambda w: (-p[w], w))[:12]
        frequencies = {w: {label: pmf[w] for label, pmf in pmfs.items()} for w in top}
        result = {
            "questions": len(ids),
            "answers_per_model": len(records["final"]),
            "matched_question_mean_metrics": means,
            "per_question": questions,
            "historical_pooled_training_corpus_metrics": historical,
            "displayed_word_frequencies": frequencies,
        }
        report["scopes"][scope] = result
        y = np.arange(len(top))
        for j, (label, title, color) in enumerate(
            (
                ("reference", "Reference", "#334155"),
                ("baseline", "Base model", "#df8b3f"),
                ("final", "After RL", "#168a83"),
            )
        ):
            ax.barh(
                y + (j - 1) * 0.24,
                [100 * pmfs[label][w] for w in top],
                height=0.22,
                label=title,
                color=color,
            )
        ax.set_yticks(y, top)
        ax.invert_yaxis()
        ax.set_title(
            ("Training questions" if scope == "seen" else "Unseen questions")
            + f"\n{len(records['final'])} actual answers per model",
            loc="left",
            fontsize=12,
        )
        ax.set_xlabel("Share of all words (%)")
        ax.grid(axis="x", alpha=0.18)
        ax.set_axisbelow(True)
        ax.spines[["top", "right"]].set_visible(False)
        # Deterministic examples: first question ID, sample 0, no quality-based selection.
        q = ids[0]
        snippets.append(f"SCOPE: {scope}\nPROMPT:\n{refs[q]['prompt']}\n")
        for label in ("baseline", "final"):
            candidates = sorted(
                [r for r in records[label] if r["case_id"] == q],
                key=lambda r: int(r["id"].rsplit("-", 1)[1]),
            )
            sample = candidates[0]
            snippets.append(
                f"{label.upper()} SAMPLE (id={sample['id']}; "
                f"terminated={sample['terminated']}):\n{sample['passage']}\n"
            )
    fig.suptitle(
        "Word frequencies before and after distribution-only RL", fontsize=16, x=0.065, ha="left"
    )
    handles, labels = axes[0].get_legend_handles_labels()
    fig.legend(
        handles, labels, loc="upper right", bbox_to_anchor=(0.98, 0.98), ncol=3, frameon=False
    )
    fig.text(
        0.065,
        0.025,
        "Top 12 reference content words in each scope; fixed stopword list affects display only.\n"
        "Question PMFs receive equal weight. No fresh generations or GPU compute used.",
        fontsize=9,
        color="#475569",
    )
    fig.tight_layout(rect=(0.035, 0.09, 0.99, 0.91))
    fig.savefig(output / "word-frequency-comparison.png", dpi=180, facecolor="white")
    plt.close(fig)
    (output / "word-frequency-report.json").write_text(json.dumps(report, indent=2) + "\n")
    (output / "sampled-responses.txt").write_text("\n".join(snippets))
    print(
        json.dumps(
            {
                scope: report["scopes"][scope]["matched_question_mean_metrics"]
                for scope in ("seen", "unseen")
            },
            indent=2,
        )
    )
    print("plot", output / "word-frequency-comparison.png")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--source", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    analyze(args.source, args.output)
