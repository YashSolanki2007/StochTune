"""Existing DDP quality+word-KL updater, with Emma reward and copying diagnostics."""

from functools import lru_cache
from pathlib import Path


def evaluation_subset(rows, count, seed):
    import random

    rng = random.Random(seed)
    values = sorted(rows, key=lambda r: r["id"])
    rng.shuffle(values)
    representatives = {}
    for r in values:
        representatives.setdefault(r["group_id"], r)
    chosen = list(representatives.values())[:count]
    seen = {r["id"] for r in chosen}
    return chosen + [r for r in values if r["id"] not in seen][: max(0, count - len(chosen))]


_here = Path(__file__).resolve().parent
_impl = _here / "_policy_impl.py"
if not _impl.exists():
    _impl = _here.parent / "helpsteer-rl-v2/train_policy.py"
_source = _impl.read_text()
_replacements = {
    'audit = IndependentAudit(local_rank, cfg["audit_max_sequence_length"])': (
        "audit = None  # Quality evaluation uses the same frozen training judge."
    ),
    '"reward_trained_on_human_preferences": True': '"reward_trained_on_human_preferences": False',
    '"Auditor is a separate same-family 7B model, not human ground truth."': (
        '"Quality evaluation uses the training judge; not independent or human ground truth."'
    ),
    "# This auditor is separate from the learned reward and is never optimized.": (
        "# No extra same-model auditor is counted as independent evaluation."
    ),
    '        references = [r["response"]': (
        '        cases["validation"] = evaluation_subset(cases["validation"], '
        'cfg["validation_prompt_count"], cfg["seed"]+11)\n'
        '        cases["test"] = evaluation_subset(cases["test"], '
        'cfg["test_prompt_count"], cfg["seed"]+12)\n'
        '        references = [r["response"]'
    ),
    "        dist.destroy_process_group()": '        if __import__("sys").exc_info()[0] is None:\n'
    "            dist.destroy_process_group()",
}
for _old, _new in _replacements.items():
    if _source.count(_old) != 1:
        raise RuntimeError("Shared policy source changed; review Emma integration: " + _old)
    _source = _source.replace(_old, _new)
_source, _separator, _cli = _source.partition('\nif __name__ == "__main__":')
if not _separator:
    raise RuntimeError("Shared policy CLI boundary changed")
exec(compile(_source, str(_impl), "exec"), globals())

_evaluate = globals()["evaluate"]


@lru_cache(maxsize=1)
def corpus_windows():
    from common import ROOT, words

    tokens = words((ROOT / "data/emma.txt").read_text())
    return {tuple(tokens[i : i + 12]) for i in range(len(tokens) - 11)}


def evaluate(
    policy, tokenizer, reward, target, cases, cfg, output, label, device, audit=None, initial=False
):
    import statistics

    import torch.distributed as dist
    from common import read_rows, words, write_json
    from ddp_passage_rl import broadcast_object
    from pilot import duplicate_window_rate

    metrics = _evaluate(
        policy,
        tokenizer,
        reward,
        target,
        cases,
        cfg,
        output,
        label,
        device,
        audit=None,
        initial=initial,
    )
    if dist.get_rank() == 0:
        rows = read_rows(output / f"{label}-samples.jsonl")
        metrics["corpus_12word_overlap"] = statistics.mean(
            duplicate_window_rate(r["passage"], corpus_windows()) or 0.0 for r in rows
        )
        metrics["judge_quality_mean"] = statistics.mean(r["normalized_reward"] for r in rows)
        metrics["quality_before_length_penalty_mean"] = statistics.mean(
            r.get("quality_reward_before_length_penalty", r["quality_reward"]) for r in rows
        )
        metrics["length_penalty_mean"] = statistics.mean(r.get("length_penalty", 0.0) for r in rows)
        from length_reward import settings

        spec = settings(cfg)
        lengths = [len(words(r["passage"])) for r in rows]
        metrics["length_in_range_rate"] = statistics.mean(
            spec["min_words"] <= length <= spec["max_words"] for length in lengths
        )
        metrics["length_underflow_rate"] = statistics.mean(
            length < spec["min_words"] for length in lengths
        )
        metrics["length_overflow_rate"] = statistics.mean(
            length > spec["max_words"] for length in lengths
        )
        for key in ("coherence", "readability", "prompt_fulfillment"):
            metrics[f"training_judge_{key}"] = statistics.mean(
                r["rubric_scores"][key] for r in rows
            )
        write_json(output / f"{label}-metrics.json", metrics)
    return broadcast_object(metrics)


if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--arm", choices=["quality-unigram"], required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    globals()["train"](args.arm, args.output.resolve())
