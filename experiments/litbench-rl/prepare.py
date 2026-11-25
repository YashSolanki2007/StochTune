"""Pinned CPU-only preparation with prompt AND shared-story component disjoint splits."""

import hashlib
import html
import json
import random
import re
from collections import Counter
from pathlib import Path

from common import (
    ROOT,
    make_distribution,
    words,
    write_json,
    write_rows,
)
from huggingface_hub import hf_hub_download
from transformers import AutoTokenizer

FANTASY = re.compile(
    r"\b(dragon|dragons|wizard|wizards|magic|magical|witch|witches|elf|elves|fairy|fairies|sorcerer|sorceress|necromancer|enchanted)\b",
    re.I,
)


def clean(text):
    return html.unescape(text).replace("\r\n", "\n").strip()


def fingerprint(text):
    return hashlib.sha256(" ".join(clean(text).casefold().split()).encode()).hexdigest()


def split_components(rows):
    parent = {}

    def find(x):
        parent.setdefault(x, x)
        while parent[x] != x:
            parent[x] = parent[parent[x]]
            x = parent[x]
        return x

    def union(a, b):
        a, b = find(a), find(b)
        if a != b:
            parent[max(a, b)] = min(a, b)

    for r in rows:
        q = "q:" + fingerprint(r["prompt"])
        union(q, "s:" + fingerprint(r["chosen_story"]))
        union(q, "s:" + fingerprint(r["rejected_story"]))
    result = {}
    for r in rows:
        key = fingerprint(r["prompt"])
        component = find("q:" + key)
        fold = int(hashlib.sha256(("42:" + component).encode()).hexdigest(), 16) % 100
        name = (
            "reward_train"
            if fold < 70
            else "reward_validation"
            if fold < 75
            else "reward_audit"
            if fold < 80
            else "policy_train"
            if fold < 90
            else "policy_validation"
            if fold < 95
            else "policy_test"
        )
        result[key] = (component, name)
    return result


def balanced_subset(rows, limit, seed):
    rows = sorted(rows, key=lambda r: r["pair_id"])
    rng = random.Random(seed)
    matched = [
        r
        for r in rows
        if min(r["response_tokens_1"], r["response_tokens_2"])
        / max(r["response_tokens_1"], r["response_tokens_2"])
        >= 0.8
    ]
    rng.shuffle(matched)
    first = matched[: min(len(matched), limit // 2)]
    chosen = {r["pair_id"] for r in first}
    rest = [r for r in rows if r["pair_id"] not in chosen]
    rng.shuffle(rest)
    return first + rest[: max(0, limit - len(first))]


def request(prompt):
    return (
        "Write an original, self-contained short story responding to this premise. "
        "Return only the story.\n\n"
        + re.sub(r"^\s*\[[A-Z ]+\]\s*", "", clean(prompt))
    )


def prepare():
    import pyarrow
    import pyarrow.parquet as pq

    cfg = json.loads((ROOT / "config.json").read_text())
    raw = Path(
        hf_hub_download(
            cfg["dataset"],
            "data/train-00000-of-00001.parquet",
            repo_type="dataset",
            revision=cfg["dataset_revision"],
        )
    )
    original = pq.read_table(raw).to_pylist()
    assignment = split_components(original)
    tokenizer = AutoTokenizer.from_pretrained(
        cfg["model"], revision=cfg["revision"], local_files_only=True
    )
    eligible = []
    excluded = Counter()
    for i, r in enumerate(original):
        texts = [clean(r["chosen_story"]), clean(r["rejected_story"])]
        lengths = [len(words(t)) for t in texts]
        if min(lengths) < cfg["story_min_words"] or max(lengths) > cfg["story_max_words"]:
            excluded["story_word_length"] += 1
            continue
        if texts[0] == texts[1] or any(t in ("[deleted]", "[removed]") for t in texts):
            excluded["invalid_pair"] += 1
            continue
        q = fingerprint(r["prompt"])
        group, split = assignment[q]
        prompt = request(r["prompt"])
        eligible.append(
            {
                "pair_id": hashlib.sha256(
                    (q + fingerprint(texts[0]) + fingerprint(texts[1])).encode()
                ).hexdigest(),
                "id": q,
                "group_id": group,
                "split": split,
                "prompt": prompt,
                "original_prompt": clean(r["prompt"]),
                "chosen": texts[0],
                "rejected": texts[1],
                "fantasy": bool(FANTASY.search(r["prompt"])),
            }
        )
    # Tokenize batches in Rust; no response is truncated to satisfy a length limit.
    prepared = []
    for start in range(0, len(eligible), 128):
        batch = eligible[start : start + 128]
        conversations = [
            [{"role": "user", "content": r["prompt"]}, {"role": "assistant", "content": r[k]}]
            for r in batch
            for k in ("chosen", "rejected")
        ]
        sequences = tokenizer.apply_chat_template(
            conversations, tokenize=True, add_generation_prompt=False
        )
        prompts = tokenizer.apply_chat_template(
            [[{"role": "user", "content": r["prompt"]}] for r in batch],
            tokenize=True,
            add_generation_prompt=True,
        )
        response_tokens = tokenizer(
            [r[k] for r in batch for k in ("chosen", "rejected")], add_special_tokens=False
        )["input_ids"]
        for j, r in enumerate(batch):
            ids = sequences[2 * j : 2 * j + 2]
            lengths = [len(s) for s in response_tokens[2 * j : 2 * j + 2]]
            if (
                len(prompts[j]) > cfg["max_prompt_tokens"]
                or max(map(len, ids)) > cfg["reward_max_sequence_length"]
            ):
                excluded["token_context_length"] += 1
                continue
            flip = int(r["pair_id"][-1], 16) % 2
            r.update(
                prompt_ids=prompts[j],
                response_1=r["rejected"] if flip else r["chosen"],
                response_2=r["chosen"] if flip else r["rejected"],
                ids_1=ids[1] if flip else ids[0],
                ids_2=ids[0] if flip else ids[1],
                response_tokens_1=lengths[1] if flip else lengths[0],
                response_tokens_2=lengths[0] if flip else lengths[1],
                preference_strength=1 if flip else -1,
            )
            prepared.append(r)
    unique = {r["pair_id"]: r for r in prepared}
    prepared = list(unique.values())
    splits = {
        name: [r for r in prepared if r["split"] == name]
        for name in (
            "reward_train",
            "reward_validation",
            "reward_audit",
            "policy_train",
            "policy_validation",
            "policy_test",
        )
    }
    train = sorted(splits["reward_train"], key=lambda r: r["pair_id"])
    random.Random(cfg["seed"]).shuffle(train)
    splits["reward_train"] = train[: cfg["reward_train_pair_limit"]]
    for i, name in enumerate(("reward_validation", "reward_audit")):
        # One pair per prompt/story component avoids repeated validation units.
        by_prompt = {}
        for r in sorted(splits[name], key=lambda r: r["pair_id"]):
            by_prompt.setdefault(r["group_id"], r)
        splits[name] = balanced_subset(
            list(by_prompt.values()), cfg[name + "_pairs"], cfg["seed"] + i
        )
    for name, limit in (
        ("policy_train", None),
        ("policy_validation", cfg["validation_prompt_count"]),
        ("policy_test", cfg["test_prompt_count"]),
    ):
        by_prompt = {}
        for r in sorted(splits[name], key=lambda r: r["pair_id"]):
            if r["fantasy"]:
                by_prompt.setdefault(r["id"], r)
        values = sorted(by_prompt.values(), key=lambda r: r["id"])
        random.Random(cfg["seed"]).shuffle(values)
        splits[name] = values if limit is None else values[:limit]
        if len(splits[name]) < (32 if limit is None else limit):
            raise ValueError("Insufficient fantasy policy prompts")
    if min(len(splits[k]) for k in ("reward_validation", "reward_audit")) < 128:
        raise ValueError("Insufficient held-out literary preference pairs")
    refs = {}
    for r in prepared:
        if r["fantasy"] and r["split"] in ("reward_train", "policy_train"):
            refs.setdefault(
                fingerprint(r["chosen"]),
                {
                    "id": r["id"],
                    "group_id": r["group_id"],
                    "story_id": fingerprint(r["chosen"]),
                    "response": r["chosen"],
                },
            )
    data = ROOT / "data"
    data.mkdir(exist_ok=True)
    for name, rows in splits.items():
        if name.startswith("policy"):
            rows = [{k: r[k] for k in ("id", "group_id", "prompt", "prompt_ids")} for r in rows]
        write_rows(data / f"{name}.jsonl", rows)
    write_rows(data / "reference.jsonl", list(refs.values()))
    target = make_distribution(cfg, [r["response"] for r in refs.values()])
    counts = Counter(w for r in refs.values() for w in words(r["response"]))
    write_json(
        data / "vocabulary.json",
        {
            "categories": [
                {"word": w, "count": counts[w], "probability": float(target.probabilities[i])}
                for w, i in target.index.items()
            ]
            + [{"word": None, "label": "OTHER", "probability": float(target.probabilities[-1])}],
            "deduplicated_reference_stories": len(refs),
            "training_word_count": sum(counts.values()),
        },
    )
    sets = {k: {r["group_id"] for r in v} for k, v in splits.items()}
    for i, a in enumerate(sets):
        for b in list(sets)[i + 1 :]:
            if sets[a] & sets[b]:
                raise ValueError("Prompt/story component leakage")
    manifest = {
        "dataset": cfg["dataset"],
        "revision": cfg["dataset_revision"],
        "raw_parquet_sha256": hashlib.sha256(raw.read_bytes()).hexdigest(),
        "configuration_sha256": hashlib.sha256((ROOT / "config.json").read_bytes()).hexdigest(),
        "counts": {k: len(v) for k, v in splits.items()},
        "excluded": dict(excluded),
        "deduplicated_reference_stories": len(refs),
        "vocabulary_words": target.unknown,
        "reference_other_mass": float(target.probabilities[-1]),
        "official_test_used": False,
        "pyarrow_version": pyarrow.__version__,
        "split_protocol": (
            "Connected components over canonical prompts and shared story texts, before filtering; "
            "hashed component assignment with seed 42."
        ),
        "fantasy_prompt_regex": FANTASY.pattern,
        "reference_protocol": (
            "Unique preferred stories from eligible fantasy prompts in original internal "
            "training partitions only. No held-out references."
        ),
        "prepared_sha256": {
            p.name: hashlib.sha256(p.read_bytes()).hexdigest()
            for p in data.iterdir()
            if p.name != "manifest.json" and p.is_file()
        },
    }
    write_json(data / "manifest.json", manifest)
    print(json.dumps(manifest, indent=2))


if __name__ == "__main__":
    prepare()
