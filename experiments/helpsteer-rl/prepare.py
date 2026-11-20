"""Pin HelpSteer2 and create prompt-disjoint reward/policy datasets on CPU."""

import argparse
import gzip
import hashlib
import json
import random
import urllib.request
from pathlib import Path

from common import ROOT, config, encoded_response, messages, prompt_id, write_json, write_rows
from transformers import AutoTokenizer


def mostly_ascii_letters(text):
    letters = [c for c in text if c.isalpha()]
    return bool(letters) and sum(c.isascii() for c in letters) / len(letters) >= 0.9


def prepare(raw):
    cfg = config()
    raw.mkdir(parents=True, exist_ok=True)
    names = ["train.jsonl.gz", "validation.jsonl.gz", "preference/preference.jsonl.gz"]
    source_hashes = {}
    loaded = {}
    for name in names:
        path = raw / Path(name).name
        if not path.exists():
            url = (
                f"https://huggingface.co/datasets/{cfg['dataset']}/resolve/"
                f"{cfg['dataset_revision']}/{name}"
            )
            urllib.request.urlretrieve(url, path)
        source_hashes[name] = hashlib.sha256(path.read_bytes()).hexdigest()
        with gzip.open(path, "rt") as stream:
            loaded[name] = [json.loads(line) for line in stream]
    attributes = {
        (prompt_id(r["prompt"]), r["response"]): r for name in names[:2] for r in loaded[name]
    }
    tokenizer = AutoTokenizer.from_pretrained(cfg["model"], revision=cfg["revision"])
    groups = {}
    excluded = {}
    for row in loaded[names[2]]:
        key = prompt_id(row["prompt"])
        try:
            conversation = messages(row["prompt"])
            pids = tokenizer.apply_chat_template(
                conversation, tokenize=True, add_generation_prompt=True
            )
        except ValueError:
            excluded["invalid_conversation"] = excluded.get("invalid_conversation", 0) + 1
            continue
        if len(pids) > cfg["max_prompt_tokens"]:
            excluded["long_prompt"] = excluded.get("long_prompt", 0) + 1
            continue
        if not all(mostly_ascii_letters(row[f"response_{i}"]) for i in (1, 2)):
            excluded["non_ascii_letter_language"] = excluded.get("non_ascii_letter_language", 0) + 1
            continue
        ids = [encoded_response(tokenizer, row["prompt"], row[f"response_{i}"]) for i in (1, 2)]
        if max(map(len, ids)) > cfg["reward_max_sequence_length"]:
            excluded["long_reward_response"] = excluded.get("long_reward_response", 0) + 1
            continue
        entry = {
            k: row[k]
            for k in ("prompt", "response_1", "response_2", "preference_strength", "split")
        }
        entry.update(id=key, prompt_ids=pids, ids_1=ids[0], ids_2=ids[1])
        # Keep one pair per canonical prompt; official validation takes priority.
        if key not in groups or row["split"] == "val":
            groups[key] = entry
    train = [r for r in groups.values() if r["split"] == "train"]
    official_val = [r for r in groups.values() if r["split"] == "val"]

    def preferred(row):
        strength = row["preference_strength"]
        if strength == 0:
            return None
        text = row["response_2"] if strength > 0 else row["response_1"]
        attr = attributes.get((row["id"], text))
        if attr is None or min(attr[k] for k in ("coherence", "helpfulness", "correctness")) < 3:
            return None
        if (
            len(tokenizer.encode(text, add_special_tokens=False))
            > cfg["reference_response_max_tokens"]
        ):
            return None
        return text

    reward_train = [r for r in train if int(r["id"], 16) % 4 != 0]
    policy_train = [r for r in train if int(r["id"], 16) % 4 == 0 and preferred(r)]
    eligible_val = sorted([r for r in official_val if preferred(r)], key=lambda r: r["id"])
    random.Random(cfg["seed"]).shuffle(eligible_val)
    nv, nt = cfg["validation_prompt_count"], cfg["test_prompt_count"]
    if len(eligible_val) < nv + nt:
        raise ValueError(f"Only {len(eligible_val)} eligible validation prompts; need {nv + nt}")
    validation, test = eligible_val[:nv], eligible_val[nv : nv + nt]
    policy_eval_ids = {r["id"] for r in validation + test}
    reward_val = [r for r in official_val if r["id"] not in policy_eval_ids]
    splits = {
        "reward_train": reward_train,
        "reward_validation": reward_val,
        "policy_train": policy_train,
        "policy_validation": validation,
        "policy_test": test,
    }
    sets = {key: {r["id"] for r in rows} for key, rows in splits.items()}
    assert all(
        not a.intersection(b)
        for i, a in enumerate(sets.values())
        for b in list(sets.values())[i + 1 :]
    )
    if min(len(reward_train), len(policy_train)) < 100 or len(reward_val) < 50:
        raise ValueError("Too few eligible training/validation examples")
    data = ROOT / "data"
    for name, rows in splits.items():
        if name.startswith("policy"):
            rows = [{k: r[k] for k in ("id", "prompt", "prompt_ids")} for r in rows]
        write_rows(data / f"{name}.jsonl", rows)
    references = [{"id": r["id"], "response": preferred(r)} for r in policy_train]
    write_rows(data / "reference.jsonl", references)
    hashes = {p.name: hashlib.sha256(p.read_bytes()).hexdigest() for p in data.glob("*.jsonl")}
    manifest = {
        "dataset": cfg["dataset"],
        "revision": cfg["dataset_revision"],
        "raw_sha256": source_hashes,
        "prepared_sha256": hashes,
        "counts": {name: len(rows) for name, rows in splits.items()},
        "reference_responses": len(references),
        "excluded": excluded,
        "prompt_ids": {key: sorted(value) for key, value in sets.items()},
        "selection": "ASCII-letter ratio >=0.9; prompt <=384 tokens; complete reward input <=1024. "
        "Policy/reference uses non-tied preferred response <=256 tokens with human "
        "coherence, helpfulness, correctness all >=3/4. No text is truncated.",
        "splitting": "Canonical prompt hash; 75% of original train for reward training, "
        "25% eligible prompts for policy/PMF; official validation divided into "
        "disjoint reward validation, 16 policy validation, 32 policy test prompts.",
    }
    write_json(data / "manifest.json", manifest)
    print(json.dumps({"counts": manifest["counts"], "reference_responses": len(references)}))


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--raw", type=Path, default=ROOT / "data/raw")
    prepare(parser.parse_args().raw.resolve())
