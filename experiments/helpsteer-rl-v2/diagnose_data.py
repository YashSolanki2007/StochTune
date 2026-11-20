"""Compare reference coverage and human-label baselines; CPU only, no model calls."""

import argparse
import gzip
import json
from collections import Counter
from pathlib import Path

import numpy as np
from common import ROOT, config, prompt_id, read_rows, words, write_json


def run(raw):
    cfg = config()
    data = ROOT / "data"
    references = read_rows(data / "reference.jsonl")
    counts = Counter(w for r in references for w in words(r["response"]))
    validation = read_rows(data / "reward_validation.jsonl")
    source = {}
    with gzip.open(raw / "preference.jsonl.gz", "rt") as stream:
        for line in stream:
            row = json.loads(line)
            if row["split"] == "val":
                source[prompt_id(row["prompt"])] = row
    preferred = []
    for case in read_rows(data / "policy_test.jsonl"):
        row = source[case["id"]]
        preferred.append(row["response_2"] if row["preference_strength"] > 0 else row["response_1"])
    observed = Counter(w for text in preferred for w in words(text))
    old = ROOT.parents[1] / "outputs/helpsteer-subset-deploy/data/reference.jsonl"
    old_counts = Counter(w for r in read_rows(old) for w in words(r["response"]))
    report = {
        "cpu_only": True,
        "model_predictions_computed": False,
        "policy_training_prompts": len(read_rows(data / "policy_train.jsonl")),
        "reference_answers": len(references),
        "reference_word_types": len(counts),
        "reference_words": sum(counts.values()),
        "old_reference_answers": 128,
        "old_reference_words": sum(old_counts.values()),
        "same_human_test_answers": len(preferred),
        "human_test_unknown_word_rate_old_reference": sum(
            n for w, n in observed.items() if w not in old_counts
        )
        / sum(observed.values()),
        "human_test_unknown_word_rate_broad_reference": sum(
            n for w, n in observed.items() if w not in counts
        )
        / sum(observed.values()),
        "reward_validation_pairs": len(validation),
        "long_reward_validation_pairs": sum(
            max(len(r["ids_1"]), len(r["ids_2"])) > 1024 for r in validation
        ),
        "length_matched_non_tie_pairs": sum(
            r["preference_strength"] != 0
            and min(r["response_tokens_1"], r["response_tokens_2"])
            / max(r["response_tokens_1"], r["response_tokens_2"])
            >= 0.8
            for r in validation
        ),
        "human_coherence_distinguishable_pairs": sum(
            r["coherence_1"] != r["coherence_2"] for r in validation
        ),
        "human_preference_baselines": json.loads((data / "validation-baselines.json").read_text()),
        "output_token_cap": cfg["max_new_tokens"],
        "reward_total_input_cap": cfg["reward_max_sequence_length"],
        "independent_audit_input_cap": cfg["audit_max_sequence_length"],
        "legacy_reward_eligibility": (
            "Not approved: its reported accuracy matched the longer-answer baseline "
            "on the old validation set; no v2 per-pair audit exists."
        ),
        "caveats": [
            "Coverage improvement describes human reference answers, not new model results.",
            "Broader word coverage does not guarantee unbiased KL or coherent generation.",
            "The fixed mixed-domain prompt pool does not guarantee matching topic proportions.",
        ],
    }
    # Explicitly describe, rather than silently excluding, references exceeding rollout length.
    lengths = []
    from transformers import AutoTokenizer

    tok = AutoTokenizer.from_pretrained(
        cfg["model"], revision=cfg["revision"], local_files_only=True
    )
    for row in references:
        lengths.append(len(tok.encode(row["response"], add_special_tokens=False)))
    report["reference_response_token_quantiles"] = np.quantile(lengths, [0.5, 0.9, 0.99]).tolist()
    report["references_longer_than_output_cap"] = sum(n > cfg["max_new_tokens"] for n in lengths)
    output = ROOT / "outputs/data-audit.json"
    write_json(output, report)
    print(json.dumps(report, indent=2))


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--raw", type=Path, required=True)
    run(parser.parse_args().raw.resolve())
