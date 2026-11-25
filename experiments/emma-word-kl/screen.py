"""Equal-word CPU diagnostics: genuine chapter holdout and comparable corpus sections."""

import json
import random
from collections import Counter

import numpy as np
from common import REPO, ROOT, BucketedWordDistribution, words
from prepare import write_json


def read_rows(path):
    return [json.loads(line) for line in path.read_text().splitlines() if line.strip()]


def chunks(tokens, width):
    return [" ".join(tokens[i : i + width]) for i in range(0, len(tokens) - width + 1, width)]


def summary(values):
    values = np.asarray(values)
    return {
        "mean": float(values.mean()),
        "median": float(np.median(values)),
        "p05_p95": np.percentile(values, [5, 95]).tolist(),
    }


def js(p, counts):
    q = counts / counts.sum()
    midpoint = (p + q) / 2
    a, b = p > 0, q > 0
    return float(
        0.5 * np.sum(p[a] * np.log(p[a] / midpoint[a]))
        + 0.5 * np.sum(q[b] * np.log(q[b] / midpoint[b]))
    )


def batch_diagnostic(reference_groups, observed_groups, cfg, fixed_vocabulary=None):
    target = BucketedWordDistribution(
        reference_groups, cfg["vocabulary_max_words"], cfg["vocabulary_min_count"]
    )
    if fixed_vocabulary is not None:
        target.index = {w: i for i, w in enumerate(fixed_vocabulary)}
        target.unknown = len(target.index)
        reference = Counter(words(" ".join(reference_groups)))
        counts = np.array(
            [reference[w] for w in fixed_vocabulary]
            + [sum(c for w, c in reference.items() if w not in target.index)],
            dtype=np.float64,
        )
        target.probabilities = counts / counts.sum()
        target.alpha = target.smoothing_mass / len(counts)
    blocks = [chunks(words(g), cfg["screen_passage_words"]) for g in observed_groups]
    blocks = [g for g in blocks if g]
    all_blocks = [p for group in blocks for p in group]
    if len(all_blocks) < 16:
        raise ValueError("Need at least sixteen distinct observed passages")
    counts = target.counts(all_blocks)
    grouped, offset = [], 0
    for group in blocks:
        if len(group) >= 4:
            grouped.append(np.arange(offset, offset + len(group)))
        offset += len(group)
    rng = np.random.default_rng(cfg["seed"])
    result = {
        "source_passages": len(all_blocks),
        "source_groups": len(blocks),
        "groups_with_at_least_four_passages": len(grouped),
        "word_categories": len(target.probabilities),
        "reference_other_mass": float(target.probabilities[-1]),
        "words_per_batch": cfg["screen_passage_words"] * 16,
    }
    for mode in ("random_16_passages", "four_groups_four_passages"):
        if mode.startswith("four") and len(grouped) < 4:
            result[mode] = {"unavailable": "Fewer than four eligible groups"}
            continue
        kls, jss = [], []
        for _ in range(cfg["screen_resamples"]):
            if mode.startswith("four"):
                ix = np.concatenate(
                    [
                        rng.choice(grouped[i], 4, replace=False)
                        for i in rng.choice(len(grouped), 4, replace=False)
                    ]
                )
            else:
                ix = rng.choice(len(counts), 16, replace=False)
            total = counts[ix].sum(0)
            kls.append(target.kl(total))
            jss.append(js(target.probabilities, total))
        result[mode] = {"kl_nats": summary(kls), "js_nats": summary(jss)}
    return result


def section_split(text, cfg):
    sections = chunks(words(text), cfg["screen_section_words"])
    ids = list(range(len(sections)))
    random.Random(cfg["seed"]).shuffle(ids)
    train = set(ids[: int(0.7 * len(ids))])
    return (
        [s for i, s in enumerate(sections) if i in train],
        [s for i, s in enumerate(sections) if i not in train],
    )


def screen():
    cfg = json.loads((ROOT / "config.json").read_text())
    data = ROOT / "data"
    actual = {
        name: read_rows(data / f"chapters_{name}.jsonl") for name in ("train", "validation", "test")
    }
    # This first diagnostic has a different target from the training continuation PMF.
    # It measures whole-chapter variation, not training or model performance.
    chapter_report = {
        name: batch_diagnostic(
            [r["text"] for r in actual["train"]], [r["text"] for r in actual[name]], cfg
        )
        for name in ("train", "validation", "test")
    }
    # Restore original chapter ordering; don't concatenate shuffled partitions.
    from prepare import chapters

    emma_text = "\n\n".join(r["text"] for r in chapters((data / "emma.txt").read_text()))
    shakespeare = REPO / "experiments/shakespeare-quality-pilot/data/tinyshakespeare.txt"
    if not shakespeare.exists():
        raise FileNotFoundError("Existing Shakespeare baseline required; no GPU data generation")
    raw = {"emma": emma_text, "tiny_shakespeare": shakespeare.read_text()}
    sets = {name: section_split(text, cfg) for name, text in raw.items()}
    pooled_training = Counter(w for train, _ in sets.values() for s in train for w in words(s))
    common_vocabulary = [
        w for w, _ in sorted(pooled_training.items(), key=lambda item: (-item[1], item[0]))[:1000]
    ]
    comparison = {}
    for name, (train, heldout) in sets.items():
        comparison[name] = {
            "training_sections": len(train),
            "heldout_sections": len(heldout),
            "own_capped_vocabulary": batch_diagnostic(train, heldout, cfg),
            "shared_1000_words_plus_other": batch_diagnostic(
                train, heldout, cfg, common_vocabulary
            ),
        }
    reference = read_rows(data / "reference.jsonl")
    target = BucketedWordDistribution(
        [r["response"] for r in reference], cfg["vocabulary_max_words"], cfg["vocabulary_min_count"]
    )
    continuation_report = {}
    for name in ("validation", "test"):
        rows = read_rows(data / f"policy_{name}.jsonl")
        counts = target.counts([r["response"] for r in rows])
        rng = np.random.default_rng(cfg["seed"])
        values, lengths = [], []
        for _ in range(cfg["screen_resamples"]):
            ix = rng.choice(len(rows), 16, replace=False)
            total = counts[ix].sum(0)
            values.append(target.kl(total))
            lengths.append(float(total.sum()))
        continuation_report[name] = {
            "source_batch_kl_nats": summary(values),
            "source_batch_word_count": summary(lengths),
        }
    report = {
        "resamples_per_condition": cfg["screen_resamples"],
        "gpu_used": False,
        "trained_model_results": False,
        "smoothing_total_mass": 1.0,
        "actual_chapter_split": chapter_report,
        "matched_section_comparison": comparison,
        "training_objective_human_continuation_baselines": continuation_report,
        "limitations": [
            "Descriptive source-text diagnostics, not evidence RL will improve.",
            "Resampled batches reuse a finite corpus; "
            "intervals describe batches, not population confidence.",
            "Own vocabularies differ; the shared 1000-word comparison "
            "aggregates much of the lexical tail.",
            "Equal sections are not semantic topics; "
            "chapter splits and section splits answer different questions.",
            "Classic-book pretraining memorization is possible "
            "and must be checked on generated text.",
        ],
    }
    write_json(ROOT / "screening-results.json", report)
    print(json.dumps(report, indent=2))


if __name__ == "__main__":
    screen()
