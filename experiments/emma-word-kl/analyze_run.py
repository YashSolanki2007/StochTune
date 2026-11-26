"""Recover saved Emma validation diagnostics without generation or GPU use.

Raw files are never rewritten. Physical-line parsing preserves Unicode
separators. All outcomes remain validation measurements, not final test results.
"""

import argparse
import hashlib
import json
import statistics
from collections import defaultdict
from pathlib import Path

import numpy as np

from common import ROOT, config, make_distribution, read_rows, words, write_json
from pilot import duplicate_window_rate


def analyze(source, destination, resamples=2000, seed=20261124):
    cfg = config()
    references = read_rows(ROOT / "data/reference.jsonl")
    target = make_distribution(cfg, [r["response"] for r in references])
    corpus = words((ROOT / "data/emma.txt").read_text())
    windows = {tuple(corpus[i:i+12]) for i in range(len(corpus)-11)}
    baseline = read_rows(source / "validation-0-samples.jsonl")
    ids = [r["id"] for r in baseline]
    assert len(ids) == len(set(ids)) == 16
    baseline_by_id = {r["id"]:r for r in baseline}
    base_counts = target.counts([r["passage"] for r in baseline]).sum(0).astype(np.int64)
    count = int(base_counts.sum())
    hashes = {}
    results = {}
    for metrics_path in sorted(source.glob("validation-*-metrics.json"),
                               key=lambda p:int(p.name.split("-")[1])):
        step = int(metrics_path.name.split("-")[1])
        samples_path = source / f"validation-{step}-samples.jsonl"
        rows = read_rows(samples_path)
        by_id = {r["id"]:r for r in rows}
        assert len(rows) == len(by_id) == 16 and set(by_id) == set(ids)
        rows = [by_id[i] for i in ids]
        for path in [metrics_path, samples_path]:
            hashes[path.name] = hashlib.sha256(path.read_bytes()).hexdigest()
        recorded = json.loads(metrics_path.read_text())
        counts = target.counts([r["passage"] for r in rows]).sum(0).astype(np.int64)
        for key, value in {
            "quality_reward_mean":statistics.mean(r["quality_reward"] for r in rows),
            "mean_word_count":statistics.mean(len(words(r["passage"])) for r in rows),
            "terminated_rate":statistics.mean(r["terminated"] for r in rows),
            "distribution_kl":target.kl(counts),
        }.items():
            assert np.isclose(recorded[key], value), (step, key)
        recovered = {**recorded,
            "corpus_12word_overlap":statistics.mean(duplicate_window_rate(r["passage"], windows) or 0.0 for r in rows),
            **{f"training_judge_{k}":statistics.mean(r["rubric_scores"][k] for r in rows)
               for k in ["coherence","readability","prompt_fulfillment"]}}
        write_json(destination/f"validation-{step}-recovered-metrics.json",recovered)
        # Count matching is a sensitivity diagnostic, not a correction for KL bias.
        matched_n=min(count,int(counts.sum()))
        matched={}
        for name, c in [("baseline",base_counts),("checkpoint",counts)]:
            rng=np.random.default_rng(seed)
            values=[target.kl(rng.multivariate_hypergeometric(c,matched_n)) for _ in range(resamples)]
            matched[name]={"mean_kl":float(np.mean(values)), "observed_words":int(c.sum())}
        groups=defaultdict(list)
        for r in rows:
            groups[r["case_id"]].append(r["quality_reward"]-baseline_by_id[r["id"]]["quality_reward"])
        delta=np.array([np.mean(groups[k]) for k in sorted(groups)])
        rng=np.random.default_rng(seed)
        draws=rng.integers(0,len(delta),size=(10000,len(delta)))
        interval=np.quantile(delta[draws].mean(1),[.025,.975]).tolist()
        results[str(step)]={"metrics":recovered,"count_matched_words":matched_n,
            "count_matched":matched,"paired_question_quality_delta":float(delta.mean()),
            "paired_question_quality_ci95":interval,"questions":len(groups)}
    report={"source_job_id":"519712","training_steps_completed":80,
        "training_passages":1280,"latest_saved_checkpoint":70,
        "scheduler_state":"FAILED","failure":"str.splitlines split a valid U+2028 inside a JSON string",
        "validation_samples_all_recovered":True,"final_test_available":False,
        "fresh_generations":0,"gpu_hours_used_for_recovery":0,
        "seed":seed,"count_resamples":resamples,"quality_bootstrap_resamples":10000,
        "source_sha256":hashes,"checkpoints":results,
        "limitations":["Validation only; eight prompts with two answers each, one seed.",
            "Quality uses the training judge, not human evaluation.",
            "Count matching ignores passage dependence and does not remove histogram estimator bias.",
            "Question bootstrap conditions on saved generations; no multiple-comparison correction.",
            "Step-80 validation samples survive, but step-80 weights were not checkpointed."]}
    write_json(destination/"recovery-analysis.json",report)
    return report


if __name__ == "__main__":
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--input",type=Path,required=True)
    parser.add_argument("--output",type=Path,required=True)
    args=parser.parse_args()
    report=analyze(args.input,args.output)
    print(json.dumps({s:{k:v for k,v in x.items() if k!="metrics"}
                      for s,x in report["checkpoints"].items()},indent=2))
