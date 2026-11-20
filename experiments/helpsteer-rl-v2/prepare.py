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
from validation import baselines


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
        missing_attributes = False
        for i in (1, 2):
            attr = attributes.get((key, row[f"response_{i}"]))
            if attr is None:
                missing_attributes = True
                break
            for name in ("coherence", "helpfulness", "correctness"):
                entry[f"{name}_{i}"] = attr[name]
            entry[f"response_tokens_{i}"] = len(
                tokenizer.encode(row[f"response_{i}"], add_special_tokens=False)
            )
        if missing_attributes:
            excluded["missing_human_attributes"] = excluded.get("missing_human_attributes", 0) + 1
            continue
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
        return text

    reward_train = [r for r in train if int(r["id"], 16) % 4 != 0]
    # The old reward model never saw >1024-token inputs. Reserve explicit human
    # long-input examples for both its audit and any future reward retraining.
    long_candidates = sorted(
        [r for r in reward_train if max(len(r["ids_1"]), len(r["ids_2"])) > 1024],
        key=lambda r: r["id"],
    )
    random.Random(cfg["seed"]).shuffle(long_candidates)
    long_audit = long_candidates[:64]
    long_ids = {r["id"] for r in long_audit}
    if len(long_ids) < 32:
        raise ValueError("Insufficient held-out long-input human pairs")
    reward_train = [r for r in reward_train if r["id"] not in long_ids]
    candidates = [r for r in train if int(r["id"], 16) % 4 == 0 and preferred(r)]
    candidates.sort(
        key=lambda r: r["response_tokens_2"]
        if r["preference_strength"] > 0
        else r["response_tokens_1"]
    )
    rng = random.Random(cfg["seed"])
    # Equal-probability length strata preserve coverage in a small policy subset.
    strata = [
        candidates[i * len(candidates) // 4 : (i + 1) * len(candidates) // 4] for i in range(4)
    ]
    for stratum in strata:
        rng.shuffle(stratum)
    limit = cfg["policy_train_prompt_limit"]
    policy_train = [strata[i % 4].pop() for i in range(limit)]
    fixed = json.loads((ROOT / "evaluation-splits.json").read_text())
    by_id = {r["id"]: r for r in official_val}
    validation = [by_id[key] for key in fixed["validation"]]
    test = [by_id[key] for key in fixed["test"]]
    assert len(validation) == cfg["validation_prompt_count"]
    assert len(test) == cfg["test_prompt_count"]
    policy_eval_ids = {r["id"] for r in validation + test}
    reward_val = [r for r in official_val if r["id"] not in policy_eval_ids] + long_audit
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
    # The PMF uses ALL eligible original-training references, independently of
    # the 128-prompt policy subset. Official-validation answers are never included.
    references = [
        {"id": r["id"], "response": preferred(r)}
        for r in train
        if r["id"] not in long_ids and preferred(r)
    ]
    write_rows(data / "reference.jsonl", references)
    hashes = {p.name: hashlib.sha256(p.read_bytes()).hexdigest() for p in data.glob("*.jsonl")}
    manifest = {
        "dataset": cfg["dataset"],
        "revision": cfg["dataset_revision"],
        "raw_sha256": source_hashes,
        "prepared_sha256": hashes,
        "counts": {name: len(rows) for name, rows in splits.items()},
        "reference_responses": len(references),
        "reference_prompt_ids": sorted(r["id"] for r in references),
        "reference_scope": "All eligible preferred original-training responses; includes "
        "reward-training prompts; excludes every official-validation prompt "
        "and the reserved long-input reward-audit prompts.",
        "long_reward_audit_ids": sorted(long_ids),
        "excluded": excluded,
        "prompt_ids": {key: sorted(value) for key, value in sets.items()},
        "selection": "ASCII-letter ratio >=0.9; prompt <=384 tokens; complete reward input <=2048. "
        "Reference uses non-tied preferred responses with human coherence, helpfulness, "
        "correctness all >=3/4. No 256-token response filter; no text is truncated.",
        "splitting": "Canonical prompt hash; 75% of original train for reward training, "
        "128 length-stratified policy prompts from the other 25%; preserve original held-out "
        "policy prompt IDs; reward validation excludes those policy evaluation prompts.",
    }
    write_json(data / "manifest.json", manifest)
    strengths = [r["preference_strength"] for r in reward_train if r["preference_strength"]]
    majority = 1 if sum(s > 0 for s in strengths) >= len(strengths) / 2 else -1
    write_json(data / "validation-baselines.json", baselines(reward_val, majority))
    print(json.dumps({"counts": manifest["counts"], "reference_responses": len(references)}))


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--raw", type=Path, default=ROOT / "data/raw")
    prepare(parser.parse_args().raw.resolve())
