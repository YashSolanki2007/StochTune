"""Pinned Emma text, chapter-disjoint splits and genuine continuation pairs; CPU only."""

import hashlib
import io
import json
import random
import re
import urllib.request
import zipfile

from common import ROOT, BucketedWordDistribution, words

SOURCE = (
    "https://raw.githubusercontent.com/nltk/nltk_data/{revision}/packages/corpora/gutenberg.zip"
)
HEADING = re.compile(r"^(VOLUME [IVX]+|CHAPTER [IVX]+)\s*$", re.M)


def digest(data):
    return hashlib.sha256(data).hexdigest()


def write_json(path, value):
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(value, indent=2, ensure_ascii=False) + "\n")


def write_rows(path, rows):
    path.write_text("".join(json.dumps(r, ensure_ascii=False) + "\n" for r in rows))


def chapters(text):
    headings = list(HEADING.finditer(text))
    result, volume = [], None
    for i, match in enumerate(headings):
        title = match.group().strip()
        if title.startswith("VOLUME"):
            volume = title
            continue
        end = headings[i + 1].start() if i + 1 < len(headings) else len(text)
        body = text[match.end() : end].strip()
        result.append({"id": f"{volume}-{title}", "volume": volume, "chapter": title, "text": body})
    if len(result) != 55 or len({r["id"] for r in result}) != 55:
        raise ValueError("Expected Emma's 55 distinct chapters in three volumes")
    return result


def split_chapters(rows, cfg):
    names = sorted(r["id"] for r in rows)
    random.Random(cfg["seed"]).shuffle(names)
    a, b = cfg["train_chapters"], cfg["validation_chapters"]
    if a + b + cfg["test_chapters"] != len(names):
        raise ValueError("Every chapter must belong to exactly one partition")
    return {
        name: [r for r in rows if r["id"] in set(ids)]
        for name, ids in (
            ("train", names[:a]),
            ("validation", names[a : a + b]),
            ("test", names[a + b :]),
        )
    }


def continuation_pairs(chapter, cfg):
    paragraphs = [" ".join(p.split()) for p in re.split(r"\n\s*\n", chapter["text"]) if p.strip()]
    pairs, cursor = [], 0
    while cursor < len(paragraphs):
        start = cursor
        prefix = []
        while cursor < len(paragraphs) and len(words(" ".join(prefix))) < cfg["prefix_min_words"]:
            prefix.append(paragraphs[cursor])
            cursor += 1
        response = []
        while (
            cursor < len(paragraphs) and len(words(" ".join(response))) < cfg["response_min_words"]
        ):
            response.append(paragraphs[cursor])
            cursor += 1
        prefix, response = "\n\n".join(prefix), "\n\n".join(response)
        if not cfg["prefix_min_words"] <= len(words(prefix)) <= cfg["prefix_max_words"]:
            continue
        if not cfg["response_min_words"] <= len(words(response)) <= cfg["response_max_words"]:
            continue
        prompt = (
            "Continue this passage from Jane Austen's Emma with original prose in the same "
            "narrative setting and voice. Keep characters and events consistent. "
            "Write about 250 to 350 words. Return only the continuation.\n\nPASSAGE:\n" + prefix
        )
        pairs.append(
            {
                "id": f"{chapter['id']}-paragraph-{start}",
                "group_id": chapter["id"],
                "prompt": prompt,
                "prefix": prefix,
                "response": response,
                "source_paragraph_start": start,
                "source_paragraph_end": cursor,
            }
        )
    return pairs


def prepare():
    from transformers import AutoTokenizer

    cfg = json.loads((ROOT / "config.json").read_text())
    data = ROOT / "data"
    data.mkdir(exist_ok=True)
    url = SOURCE.format(revision=cfg["source_revision"])
    archive = data / "source-gutenberg.zip"
    if not archive.exists():
        with urllib.request.urlopen(url, timeout=30) as response:
            archive.write_bytes(response.read())
    with zipfile.ZipFile(io.BytesIO(archive.read_bytes())) as source:
        raw = source.read(cfg["source_member"])
        (data / "source-README.txt").write_bytes(source.read("gutenberg/README"))
    text = raw.decode("utf-8")
    (data / "emma.txt").write_text(text)
    all_chapters = chapters(text)
    splits = split_chapters(all_chapters, cfg)
    tokenizer = AutoTokenizer.from_pretrained(
        cfg["model"], revision=cfg["revision"], local_files_only=True
    )
    prepared = {}
    exclusions = {}
    for name, group in splits.items():
        write_rows(data / f"chapters_{name}.jsonl", group)
        cases = [case for chapter in group for case in continuation_pairs(chapter, cfg)]
        ids = tokenizer.apply_chat_template(
            [[{"role": "user", "content": case["prompt"]}] for case in cases],
            tokenize=True,
            add_generation_prompt=True,
        )
        kept = [
            {**r, "prompt_ids": tokens}
            for r, tokens in zip(cases, ids)
            if len(tokens) <= cfg["max_prompt_tokens"]
        ]
        exclusions[name] = len(cases) - len(kept)
        prepared[name] = kept
        write_rows(data / f"policy_{name}.jsonl", kept)
    references = [
        {"id": r["id"], "group_id": r["group_id"], "response": r["response"]}
        for r in prepared["train"]
    ]
    write_rows(data / "reference.jsonl", references)
    target = BucketedWordDistribution(
        [r["response"] for r in references],
        cfg["vocabulary_max_words"],
        cfg["vocabulary_min_count"],
    )
    write_json(
        data / "vocabulary.json",
        {
            "categories": [
                {"word": word, "probability": float(target.probabilities[index])}
                for word, index in target.index.items()
            ]
            + [{"word": None, "label": "OTHER", "probability": float(target.probabilities[-1])}],
            "reference_words": sum(len(words(r["response"])) for r in references),
        },
    )
    manifest = {
        "source_url": url,
        "source_revision": cfg["source_revision"],
        "source_member": cfg["source_member"],
        "source_text_sha256": digest(raw),
        "source_archive_sha256": digest(archive.read_bytes()),
        "configuration_sha256": digest((ROOT / "config.json").read_bytes()),
        "corpus_words_without_headings": sum(len(words(r["text"])) for r in all_chapters),
        "chapter_ids": {name: [r["id"] for r in rows] for name, rows in splits.items()},
        "continuation_counts": {name: len(rows) for name, rows in prepared.items()},
        "prompt_token_exclusions": exclusions,
        "reference_passages": len(references),
        "word_categories": len(target.probabilities),
        "reference_other_mass": float(target.probabilities[-1]),
        "target_protocol": (
            "Source continuations for the exact training-prompt population only; "
            "prefixes and every validation/test chapter excluded."
        ),
        "human_preference_labels": False,
        "gpu_used": False,
        "pretraining_contamination_unknown": True,
        "prepared_sha256": {
            p.name: digest(p.read_bytes())
            for p in sorted(data.iterdir())
            if p.is_file() and p.name != "manifest.json"
        },
    }
    write_json(data / "manifest.json", manifest)
    print(json.dumps(manifest, indent=2))


if __name__ == "__main__":
    prepare()
