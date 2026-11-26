"""CPU-only character unigram analysis of the same saved before/after generations."""

import argparse
import hashlib
import json
import string
from collections import Counter, defaultdict
from pathlib import Path

import matplotlib
import numpy as np

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
from analyze_word_frequencies import compare, read_rows  # noqa: E402


def count_characters(passages, protocol):
    result = Counter()
    for text in passages:
        if protocol == "raw":
            result.update(text)
        elif protocol == "letters":
            result.update(c for c in text.lower() if c in string.ascii_lowercase)
        else:
            raise ValueError("Unknown character counting protocol")
    return result


def averaged_pmf(groups, protocol):
    result = Counter()
    for texts in groups.values():
        counts = count_characters(texts, protocol)
        n = sum(counts.values())
        if not n:
            raise ValueError("Empty character sequence after filtering")
        for char, value in counts.items():
            result[char] += value / n / len(groups)
    return result


def character_metrics(target, generated):
    result = compare(target, generated)
    result["outside_reference_character_mass"] = result.pop("outside_reference_word_mass")
    return result


def analyze(source, output):
    output.mkdir(parents=True, exist_ok=True)
    refs = {r["case_id"]: r for r in read_rows(source / "data/semantic-references.jsonl")}
    report = {
        "source_job_id": "510187",
        "fresh_generation": False,
        "gpu_hours_used": 0,
        "protocols": {
            "raw": "Exact Unicode code points, case sensitive, including spaces, "
            "newlines, punctuation and code.",
            "letters": "Lowercase English letters a-z only; all other characters excluded.",
        },
        "metric_protocol": "Compare each question's pooled character counts with its own "
        "two reference answers. Average question metrics equally. Forward KL uses total "
        "pseudocount mass 1 on the generated PMF; JS and TV are unsmoothed with OTHER for "
        "characters absent from that question's references. Units are nats. This audit uses "
        "character unigrams only. References are weighted by character counts.",
        "limitations": [
            "Four seen and four unseen questions, one training seed.",
            "English character frequencies can agree without matching vocabulary or meaning.",
            "Two reference answers do not establish the true character distribution.",
            "These are the saved evaluations, not fresh independent generation.",
        ],
        "source_file_sha256": {},
        "scopes": {},
    }
    fig, axes = plt.subplots(1, 2, figsize=(13, 6.5))
    for scope, ax in zip(("seen", "unseen"), axes):
        grouped = {}
        for label in ("baseline", "final"):
            path = source / f"samples/{label}-{scope}-samples.jsonl"
            report["source_file_sha256"][str(path.relative_to(source))] = hashlib.sha256(
                path.read_bytes()
            ).hexdigest()
            groups = defaultdict(list)
            for row in read_rows(path):
                groups[row["case_id"]].append(row["passage"])
            grouped[label] = groups
        ids = sorted(grouped["baseline"])
        assert set(ids) == set(grouped["final"])
        targets = {q: refs[q]["responses"] for q in ids}
        scope_report = {
            "questions": len(ids),
            "answers_per_model": sum(len(v) for v in grouped["final"].values()),
            "protocols": {},
        }
        for protocol in ("raw", "letters"):
            rows = []
            for q in ids:
                target = count_characters(targets[q], protocol)
                rows.append(
                    {
                        "case_id": q,
                        "prompt": refs[q]["prompt"],
                        **{
                            label: character_metrics(
                                target, count_characters(grouped[label][q], protocol)
                            )
                            for label in ("baseline", "final")
                        },
                    }
                )
            means = {
                label: {
                    key: float(np.mean([r[label][key] for r in rows])) for key in rows[0][label]
                }
                for label in ("baseline", "final")
            }
            differences = np.array(
                [r["final"]["forward_kl"] - r["baseline"]["forward_kl"] for r in rows]
            )
            # Exact enumeration of the four-question empirical bootstrap distribution.
            indices = np.indices((len(ids),) * len(ids)).reshape(len(ids), -1).T
            bootstrap = differences[indices].mean(1)
            scope_report["protocols"][protocol] = {
                "question_mean_metrics": means,
                "per_question": rows,
                "questions_kl_improved": int((differences < 0).sum()),
                "question_bootstrap_kl_change_ci95": np.quantile(
                    bootstrap, [0.025, 0.975]
                ).tolist(),
                "bootstrap_note": "Across four fixed question-level estimates only; excludes "
                "generation, reference-sampling or training-seed uncertainty.",
            }
        pmfs = {
            "reference": averaged_pmf(targets, "letters"),
            **{label: averaged_pmf(grouped[label], "letters") for label in ("baseline", "final")},
        }
        top = sorted(pmfs["reference"], key=lambda c: (-pmfs["reference"][c], c))[:15]
        scope_report["displayed_letter_frequencies"] = {
            char: {label: pmf[char] for label, pmf in pmfs.items()} for char in top
        }
        report["scopes"][scope] = scope_report
        x = np.arange(len(top))
        for j, (label, title, color) in enumerate(
            (
                ("reference", "Reference", "#334155"),
                ("baseline", "Base model", "#df8b3f"),
                ("final", "After RL", "#168a83"),
            )
        ):
            ax.bar(
                x + (j - 1) * 0.25,
                [100 * pmfs[label][c] for c in top],
                width=0.23,
                color=color,
                label=title,
            )
        ax.set_xticks(x, top)
        ax.set_title(
            ("Training questions" if scope == "seen" else "Unseen questions")
            + f"\n{scope_report['answers_per_model']} actual answers per model",
            loc="left",
            fontsize=12,
        )
        ax.set_xlabel("Most frequent reference letters")
        ax.set_ylabel("Share of English letters (%)")
        ax.grid(axis="y", alpha=0.18)
        ax.set_axisbelow(True)
        ax.spines[["top", "right"]].set_visible(False)
    fig.suptitle(
        "Character frequencies before and after distribution-only RL",
        fontsize=16,
        x=0.065,
        ha="left",
    )
    handles, labels = axes[0].get_legend_handles_labels()
    fig.legend(
        handles, labels, loc="upper right", bbox_to_anchor=(0.98, 0.89), ncol=3, frameon=False
    )
    fig.text(
        0.065,
        0.035,
        "Display: lowercase a-z only; question PMFs receive equal weight.\n"
        "The report also includes case-sensitive Unicode, whitespace and punctuation. "
        "No GPU compute used.",
        fontsize=9,
        color="#475569",
    )
    fig.tight_layout(rect=(0.035, 0.12, 0.99, 0.83))
    fig.savefig(output / "character-frequency-comparison.png", dpi=180, facecolor="white")
    plt.close(fig)
    (output / "character-frequency-report.json").write_text(json.dumps(report, indent=2) + "\n")
    print(
        json.dumps(
            {
                scope: {
                    protocol: record["question_mean_metrics"]
                    for protocol, record in data["protocols"].items()
                }
                for scope, data in report["scopes"].items()
            },
            indent=2,
        )
    )


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--source", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    analyze(args.source, args.output)
