"""CPU-only corpus preparation and diagnostics; never starts model inference."""

import argparse
import hashlib
import json
import math
import random
import re
import urllib.request
from collections import Counter, defaultdict
from pathlib import Path

ROOT = Path(__file__).resolve().parent
CORPUS_REVISION = "370cbcd448eb7daf32f21a6be560b70e0b33c4e3"
CORPUS_URL = (
    f"https://raw.githubusercontent.com/karpathy/char-rnn/{CORPUS_REVISION}"
    "/data/tinyshakespeare/input.txt"
)
WORD = re.compile(r"[A-Za-z]+(?:['’][A-Za-z]+)*")


def words(text):
    return [w.lower().replace("’", "'") for w in WORD.findall(text)]


def bigrams(text):
    tokens = words(text)
    return Counter(zip(tokens, tokens[1:]))


def pooled_counts(passages):
    result = Counter()
    for passage in passages:
        result.update(bigrams(passage))
    return result


def distribution_metrics(reference, observed, smoothing_mass=1.0):
    """Full reference support + UNKNOWN; total pseudocount mass is fixed.

    Pool only within passages. Vocabulary comes only from reference counts.
    Missing support is retained in the denominator; no cross-passage bigrams.
    """
    if not reference or not observed or smoothing_mass <= 0:
        raise ValueError("Need nonempty counts and positive total smoothing mass")
    support = sorted(reference)
    n, m = sum(reference.values()), sum(observed.values())
    unknown = sum(count for pair, count in observed.items() if pair not in reference)
    alpha = smoothing_mass / (len(support) + 1)
    p = [reference[pair] / n for pair in support] + [0.0]
    q = [(observed[pair] + alpha) / (m + smoothing_mass) for pair in support]
    q.append((unknown + alpha) / (m + smoothing_mass))
    kl = sum(x * math.log(x / y) for x, y in zip(p, q) if x > 0)
    js = 0.0
    for x, y in zip(p, q):
        mid = (x + y) / 2
        js += (0.5 * x * math.log(x / mid) if x else 0) + 0.5 * y * math.log(y / mid)
    return {
        "forward_kl": kl,
        "js_divergence": js,
        "observed_bigrams": m,
        "unknown_bigram_rate": unknown / m,
        "smoothing_total_mass": smoothing_mass,
    }


def duplicate_window_rate(text, corpus_windows, width=12):
    tokens = words(text)
    windows = [tuple(tokens[i : i + width]) for i in range(len(tokens) - width + 1)]
    return sum(w in corpus_windows for w in windows) / len(windows) if windows else None


def write_json(path, value):
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(value, indent=2, ensure_ascii=False) + "\n")


def read_rows(path):
    return [json.loads(line) for line in path.read_text().splitlines() if line.strip()]


def prepare():
    data = ROOT / "data"
    data.mkdir(exist_ok=True)
    target = data / "tinyshakespeare.txt"
    if not target.exists():
        with urllib.request.urlopen(CORPUS_URL, timeout=30) as response:
            target.write_bytes(response.read())
    text = target.read_text()
    # Contiguous block holdout, NOT a play-disjoint or pretraining-clean test set.
    paragraphs = [p for p in text.split("\n\n") if len(words(p)) >= 10]
    cut = int(len(paragraphs) * 0.8)
    training, heldout = paragraphs[:cut], paragraphs[cut:]
    (data / "reference.txt").write_text("\n\n".join(training))
    (data / "heldout.txt").write_text("\n\n".join(heldout))
    reference = pooled_counts(training)
    transitions = defaultdict(list)
    for (a, b), count in reference.items():
        transitions[a].append((b, count))
    rng = random.Random(20261118)
    anchors = json.loads((ROOT / "anchors.json").read_text())
    rows = []
    for index, anchor in enumerate(anchors):
        tokens = anchor["passage"].split()
        shuffled = list(tokens)
        rng.shuffle(shuffled)
        repeated = (tokens[:12] * (len(tokens) // 12 + 1))[: len(tokens)]
        first = rng.choices(list(reference), weights=list(reference.values()), k=1)[0][0]
        sampled = [first]
        while len(sampled) < len(tokens):
            choices = transitions.get(sampled[-1])
            if not choices:
                sampled.append(rng.choice(list(transitions)))
            else:
                sampled.append(rng.choices([a for a, _ in choices], [b for _, b in choices])[0])
        variants = {
            "authored_anchor": anchor["passage"],
            "shuffled_words": " ".join(shuffled),
            "repetition": " ".join(repeated),
            "off_prompt": anchors[(index + 1) % len(anchors)]["passage"],
            "markov_bigram": " ".join(sampled),
        }
        for kind, passage in variants.items():
            rows.append(
                {
                    "id": f"{anchor['id']}-{kind}",
                    "prompt_id": anchor["id"],
                    "kind": kind,
                    "prompt": anchor["prompt"],
                    "passage": passage,
                }
            )
        eligible = [p for p in heldout if 65 <= len(words(p)) <= 170]
        passage = eligible[index * len(eligible) // len(anchors)]
        rows.append(
            {
                "id": f"source-{index}",
                "prompt_id": f"source-{index}",
                "kind": "source_excerpt",
                "prompt": None,
                "passage": passage,
            }
        )
    (data / "candidates.jsonl").write_text(
        "".join(json.dumps(r, ensure_ascii=False) + "\n" for r in rows)
    )
    write_json(
        data / "manifest.json",
        {
            "corpus_url": CORPUS_URL,
            "corpus_revision": CORPUS_REVISION,
            "corpus_sha256": hashlib.sha256(target.read_bytes()).hexdigest(),
            "candidate_sha256": hashlib.sha256(
                (data / "candidates.jsonl").read_bytes()
            ).hexdigest(),
            "seed": 20261118,
            "candidates": len(rows),
            "reference_bigrams": sum(reference.values()),
            "reference_distinct_bigrams": len(reference),
            "split": "80/20 contiguous paragraphs",
            "caveats": [
                "Diagnostic controls, not a held-out writing benchmark.",
                "Authored anchors are assistant-written, not human quality labels.",
                "Public Shakespeare may be present in model pretraining.",
                "Corruption labels are hypotheses, not ground-truth rubric scores.",
            ],
        },
    )
    report = summarize(rows, ROOT / "outputs" / "preflight")
    print(json.dumps({"manifest": str(data / "manifest.json"), "groups": report}, indent=2))


def summarize(rows, output):
    reference_text = (ROOT / "data/reference.txt").read_text()
    reference = pooled_counts(reference_text.split("\n\n"))
    corpus_tokens = words((ROOT / "data/tinyshakespeare.txt").read_text())
    windows = {tuple(corpus_tokens[i : i + 12]) for i in range(len(corpus_tokens) - 11)}
    groups = defaultdict(list)
    for row in rows:
        groups[row["kind"]].append(row)
    summaries = {}
    for kind, samples in groups.items():
        counts = pooled_counts([r["passage"] for r in samples])
        copying = [duplicate_window_rate(r["passage"], windows) for r in samples]
        summaries[kind] = {
            "passages": len(samples),
            **distribution_metrics(reference, counts),
            "mean_word_count": sum(len(words(r["passage"])) for r in samples) / len(samples),
            "mean_repeated_bigram_fraction": sum(
                1 - len(bigrams(r["passage"])) / sum(bigrams(r["passage"]).values())
                for r in samples
            )
            / len(samples),
            "mean_corpus_12word_window_overlap": sum(x for x in copying if x is not None)
            / sum(x is not None for x in copying),
        }
    write_json(output / "distribution-diagnostics.json", summaries)
    return summaries


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("command", choices=["prepare"])
    parser.parse_args()
    prepare()
