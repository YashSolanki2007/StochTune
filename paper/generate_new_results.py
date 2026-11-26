"""Rebuild completed text-experiment tables/figures from saved records, CPU only.

Run from a full repository checkout. The source archive includes the generated
assets, compact source records, and question-level statistics for inspection.
"""
from collections import defaultdict
from pathlib import Path
import csv
import hashlib
import json
import shutil

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
from scipy.spatial.distance import cdist

ROOT = Path(__file__).resolve().parents[1]
PAPER = ROOT / "paper"
SUP = PAPER / "supplementary" / "new-results"
SUP.mkdir(parents=True, exist_ok=True)
sources = {}


def record(path):
    path = Path(path)
    data = (ROOT / path).read_bytes()
    sources[str(path)] = hashlib.sha256(data).hexdigest()
    target = SUP / path
    target.parent.mkdir(parents=True, exist_ok=True)
    target.write_bytes(data)
    return json.loads(data)


def source_rows(path):
    data = (ROOT / path).read_bytes()
    sources[str(path)] = hashlib.sha256(data).hexdigest()
    return [json.loads(line) for line in data.splitlines() if line]


def table(name, caption, label, headers, rows, align=None, size="small"):
    columns = align or ("l" + "r" * (len(headers) - 1))
    text = [r"\begin{table}[htbp]", rf"\centering\{size}",
            rf"\caption{{{caption}}}\label{{{label}}}",
            rf"\begin{{tabular}}{{@{{}}{columns}@{{}}}}", r"\toprule",
            " & ".join(headers) + r"\\", r"\midrule"]
    text += [" & ".join(str(x) for x in row) + r"\\" for row in rows]
    text += [r"\bottomrule", r"\end{tabular}", r"\end{table}", ""]
    (PAPER / "tables" / f"{name}.tex").write_text("\n".join(text))


def f(value):
    return f"{value:.4f}"


def percent(value):
    return f"{100 * value:.1f}"


def bootstrap(delta):
    delta = np.asarray(delta, dtype=float)
    # Reinitialize deliberately: every metric uses the same question draws.
    rng = np.random.default_rng(20261122)
    draws = rng.integers(0, len(delta), size=(10000, len(delta)))
    lo, hi = np.quantile(delta[draws].mean(axis=1), [.025, .975])
    return {"mean_delta": float(delta.mean()), "ci95": [float(lo), float(hi)],
            "questions": len(delta)}


help_root = "results/helpsteer-v1/"
run_names = ["subset-quality-only", "subset-quality-unigram", "subset-lambda05",
             "broad-v2", "programming-unigram", "semantic-mmd",
             "kcgm-gpu-comparison/rbf", "kcgm-gpu-comparison/energy", "energy-long"]
runs = {name: record(help_root + name + "/comparison.json") for name in run_names}
for name in run_names:
    for filename in ["run_config.json", "source-manifest.json", "wandb_run.json"]:
        path = help_root + name + "/" + filename
        if (ROOT / path).exists():
            record(path)
diag = record(help_root + "distribution-only-diagnostic/comparison.json")
for name in ["run_config.json", "diagnostic-selection.json", "source-manifest.json",
             "wandb_run.json", "controller-summary.json"]:
    record(help_root + "distribution-only-diagnostic/" + name)
word = record(help_root + "word-frequency-audit/word-frequency-report.json")
char_path = "outputs/mmd-diagnostic-character-audit/character-frequency-report.json"
char = record(char_path)
for path in char["source_file_sha256"]:
    filename = Path(path).name
    actual = ROOT / help_root / "distribution-only-diagnostic" / filename
    assert hashlib.sha256(actual.read_bytes()).hexdigest() == char["source_file_sha256"][path]
char_script = ROOT / "experiments/helpsteer-mmd-diagnostic/analyze_character_frequencies.py"
shutil.copy2(char_script, SUP / "analyze_character_frequencies.py")
sources[str(char_script.relative_to(ROOT))] = hashlib.sha256(char_script.read_bytes()).hexdigest()
calibration = record(help_root + "kcgm-cpu-audit/calibration.json")
replay = record(help_root + "kcgm-cpu-audit/report.json")
audits = {name: record(help_root + f"reward-audit-{name}/checks.json")
          for name in ["legacy", "skywork"]}
passage = record("results/passage-rl-v1/ddp/comparison.json")
pilot = record("results/passage-rl-v1/pilot/comparison.json")
unigram = record("results/passage-unigram-v1/comparison.json")
count = record("results/passage-unigram-v1/final-diagnostics.json")
record("results/passage-rl-v1/quality-calibration/checks.json")

# Passage trials share one held-out 16-scenario evaluation (pilot is separate).
rows = []
for label, metrics in [("Instruction base", unigram["baseline"]),
                       ("Bigram RL (1,200)", passage["final"]),
                       ("Unigram RL (1,097)", unigram["final"])]:
    un = metrics.get("unigram_kl", count["distribution_metrics"]["unigram"]["policies"]["bigram_trained"]["raw_kl"])
    rows.append([label, f(un), f(metrics["bigram_kl"]), f(metrics["quality_reward_mean"]),
                 f'{metrics["mean_word_count"]:.1f}', percent(metrics["terminated_rate"])])
table("passage-new", "Shakespeare test: 16 held-out scenarios, four passages each. "
      "KL is pooled and smoothed; quality is the same frozen judge used in training.",
      "tab:passage-new", ["Policy (updates)", "Word KL", "Pair KL", "$r_Q$", "Words", "EOS (\\%)"], rows)
rows = []
for label, key in [("Instruction base", "baseline"), ("Bigram RL", "bigram_trained"), ("Unigram RL", "unigram_trained")]:
    rows.append([label] + [f(count["distribution_metrics"][k]["policies"][key]["matched_mean_kl"])
                          for k in ["unigram", "bigram"]])
table("passage-counts", "Post hoc count matching: 200 subsamples without replacement, "
      "6,200 words or 6,136 word pairs per policy (seed 20261119).", "tab:passage-counts",
      ["Policy", "Matched word KL", "Matched pair KL"], rows)

subset_rows = [["Instruction base", "---", f(runs[run_names[0]]["baseline"]["distribution_kl"]),
                f(runs[run_names[0]]["baseline"]["quality_reward_mean"]),
                percent(runs[run_names[0]]["baseline"]["terminated_rate"])]]
for name, label in zip(run_names[:3], ["Quality only", "Quality + word KL", "Quality + word KL"]):
    d = runs[name]; m = d["final"]
    subset_rows.append([label, str(d["distribution_weight"]), f(m["distribution_kl"]),
                        f(m["quality_reward_mean"]), percent(m["terminated_rate"])])
table("subset-new", "Initial short-answer HelpSteer subset: each arm completed 200 updates "
      "and 3,200 training answers. Evaluation: 16 prompts, two answers each; cap 384 tokens.",
      "tab:subset-new", ["Policy", "$\\lambda$", "Word KL", "$r_Q$", "EOS (\\%)"], subset_rows)
rows = []
for name, label in [("legacy", "Local BT 3B"), ("skywork", "Frozen Skywork 8B")]:
    a = audits[name]
    rows.append([label, percent(a["non_tie_accuracy"]),
                 "[" + ", ".join(percent(x) for x in a["accuracy_95pct"]) + "]",
                 percent(a["length_matched"]["accuracy"]), percent(a["human_coherence"]["accuracy"]),
                 "Pass" if a["passed"] else "Fail"])
table("reward-audit-new", "Stronger reward-model selection audit. Main accuracy: 280 non-tied "
      "human-preference pairs; length matched: 73; coherence comparisons: 146. All accuracies in percent. "
      "Intervals describe validation selection, not an independent benchmark.", "tab:reward-audit-new",
      ["Scorer", "Accuracy", "95\\% interval", "Length", "Coherence", "Gate"], rows, size="footnotesize")
rows = []
for name, label in [("broad-v2", "Broad"), ("programming-unigram", "Programming")]:
    d = runs[name]
    for phase in ["baseline", "final"]:
        m=d[phase]
        rows.append([label + (" base" if phase=="baseline" else " RL"), f(m["distribution_kl"]),
                     f(m["quality_reward_mean"]), f'{m["mean_word_count"]:.1f}', percent(m["terminated_rate"])])
table("corrected-new", "Corrected word-target experiments: 32 test questions, four answers each. "
      "Broad uses a full word vocabulary; programming uses frequent words plus OTHER. "
      "The targets and test questions differ, so absolute KL values are not comparable across rows from different tasks.",
      "tab:corrected-new", ["Protocol / policy", "Target KL", "$r_Q$", "Words", "EOS (\\%)"], rows)

# Shared semantic evaluation; reconstruct energy scores from saved embeddings.
npz_path = "results/helpsteer-v1/semantic-mmd-preparation/semantic-reference-vectors.npz"
sources[npz_path] = hashlib.sha256((ROOT / npz_path).read_bytes()).hexdigest()
vectors = np.load(ROOT / npz_path)
reference = dict(zip(vectors["case_ids"].tolist(), vectors["vectors"].astype(float)))
semantic_names = run_names[5:]
labels = ["Gaussian v1 (94)", "Calibrated Gaussian (50)", "Energy (53)", "Energy continued (200)"]
question_records = []
intervals = {}
semantic_rows = []
judge_rows = []
for index, (name, label) in enumerate(zip(semantic_names, labels)):
    d = runs[name]
    groups = {}
    for phase in ["baseline", "final"]:
        pq = record(help_root + name + f"/{phase}-conditional-mmd.json")
        samples = source_rows(help_root + name + f"/{phase}-samples.jsonl")
        grouped = defaultdict(list)
        for sample in samples:
            grouped[sample["case_id"]].append(sample)
        assert len(pq) == len(grouped) == 19 and len(samples) == 76
        groups[phase] = {}
        for case_id in sorted(pq):
            values = grouped[case_id]
            z = np.array([x["semantic_embedding"] for x in values])
            x = reference[case_id]
            zz = cdist(z,z); xx=cdist(x,x); zx=cdist(z,x)
            energy = 2*zx.mean() - zz.sum()/(len(z)*(len(z)-1)) - xx.mean()
            rbf = pq[case_id].get("conditional_rbf_u_unscaled", pq[case_id]["conditional_mmd2_u"])
            quality = np.mean([x["quality_reward"] for x in values])
            groups[phase][case_id] = {"rbf":float(rbf), "energy":float(energy), "quality":float(quality)}
            question_records.append({"run":name,"phase":phase,"case_id":case_id,**groups[phase][case_id]})
        assert set(groups[phase]) == set(pq)
        assert np.isclose(np.mean([v["quality"] for v in groups[phase].values()]), d[phase]["quality_reward_mean"])
    assert set(groups["baseline"]) == set(groups["final"])
    ids=sorted(groups["baseline"])
    intervals[name] = {metric:bootstrap([groups["final"][c][metric]-groups["baseline"][c][metric] for c in ids])
                       for metric in ["rbf","energy","quality"]}
    for phase in (["baseline","final"] if index==0 else ["final"]):
        label_now = "Instruction base" if phase=="baseline" else label
        m=d[phase]
        rbf_mean=np.mean([q["rbf"] for q in groups[phase].values()])
        energy_mean=np.mean([q["energy"] for q in groups[phase].values()])
        if index or phase=="baseline":
            assert np.isclose(rbf_mean, m.get("conditional_rbf_u_unscaled",m["conditional_mmd2_u"]))
        semantic_rows.append([label_now,f(rbf_mean),f(energy_mean),f(m["quality_reward_mean"]),f(m["reference_prototype_coverage"])])
        judge_rows.append([label_now]+[f(m[k]) for k in ["independent_coherence","independent_readability","independent_prompt_fulfillment"]]
                          +[f'{m["mean_word_count"]:.1f}',percent(m["terminated_rate"])])
table("semantic-new", "Shared held-out semantic evaluation: 19 exact questions, four answers each. "
      "All discrepancies are unscaled generated-U/fixed-reference-V estimates; lower is better. "
      "Quality and two-prototype coverage are higher-is-better. Updates are in parentheses; the 200-update row continues the 53-update energy run.",
      "tab:semantic-new", ["Policy (updates)", "Gaussian", "Energy", "$r_Q$", "Coverage"], semantic_rows)
table("semantic-judge-new", "Separate Qwen2.5-7B-Instruct audit (expected scores on 1--5), "
      "response length and termination on the same 76 answers. These are model judgments, not human ratings.",
      "tab:semantic-judge-new", ["Policy", "Coherence", "Readability", "Fulfillment", "Words", "EOS (\\%)"], judge_rows, size="footnotesize")
ci_rows=[]
for name,label in zip(semantic_names,labels):
    ci_rows.append([label]+[f'{intervals[name][k]["mean_delta"]:+.4f} [' +
                   ", ".join(f'{x:+.4f}' for x in intervals[name][k]["ci95"])+"]"
                   for k in ["rbf","energy","quality"]])
table("semantic-ci-new", "Final minus base mean changes [95\\% paired question-bootstrap intervals]. "
      "10,000 resamples of 19 question-level estimates, seed 20261122. Negative discrepancy changes and positive quality changes are favorable. "
      "All intervals include zero; they exclude training-seed and reference-population uncertainty.",
      "tab:semantic-ci-new", ["Policy", "$\\Delta$ Gaussian", "$\\Delta$ energy", "$\\Delta r_Q$"], ci_rows, size="scriptsize")

diag_rows=[]; diag_ci={}
for scope in ["seen","unseen"]:
    d=diag[scope];ids=sorted(d["baseline"]["per_question"])
    assert ids==sorted(d["final"]["per_question"])
    delta=[d["final"]["per_question"][c]["conditional_energy_u_unscaled"]-
           d["baseline"]["per_question"][c]["conditional_energy_u_unscaled"] for c in ids]
    diag_ci[scope]=bootstrap(delta)
    for phase in ["baseline","final"]:
        m=d[phase]
        diag_rows.append([scope.capitalize()+" "+("base" if phase=="baseline" else "RL"),
                          f(m["conditional_rbf_u_unscaled"]),f(m["conditional_energy_u_unscaled"]),
                          f'{m["mean_word_count"]:.1f}',percent(m["terminated_rate"])])
table("diagnostic-new", "Distribution-only diagnostic: four fixed training questions (seen), "
      "32 evaluation answers each; four validation questions (unseen), 16 each. Quality is neither rewarded nor measured.",
      "tab:diagnostic-new", ["Scope / policy","Gaussian","Energy","Words","EOS (\\%)"], diag_rows)
rows=[]
for scope in ["seen","unseen"]:
    for phase in ["baseline","final"]:
        m=word["scopes"][scope]["matched_question_mean_metrics"][phase]
        rows.append([scope.capitalize()+" "+("base" if phase=="baseline" else "RL")]+[f(m[k]) for k in ["forward_kl","jensen_shannon","total_variation"]])
table("word-audit-new", "Post hoc exact-question word matching on saved diagnostic outputs. "
      "Both reference answers are pooled by word count; question metrics are averaged equally. KL uses total generated pseudocount mass one; JSD and TV are unsmoothed with OTHER.",
      "tab:word-audit-new", ["Scope / policy","Word KL","Word JSD","Word TV"], rows)
rows=[]
for scope in ["seen","unseen"]:
    for protocol,label in [("raw","Raw Unicode"),("letters","Letters a--z")]:
        m=char["scopes"][scope]["protocols"][protocol]["question_mean_metrics"]
        rows.append([scope.capitalize(),label]+[f'{m[p][k]:.5f}' for k in ["forward_kl","jensen_shannon","total_variation"] for p in ["baseline","final"]])
table("character-audit-new", "CPU-only character-unigram audit of the same diagnostic samples. "
      "Each pair of columns is base $\\to$ RL. Raw counts preserve case, spaces, newlines and punctuation; letters retain lowercase a--z only. "
      "Reference pooling is by character count; KL smoothing and OTHER follow the word audit.",
      "tab:character-audit-new", ["Scope","Features","KL base","KL RL","JS base","JS RL","TV base","TV RL"], rows, size="scriptsize")

inventory = [("Shakespeare pilot",pilot,1), ("Shakespeare bigram",passage,2),
             ("Shakespeare unigram",unigram,2)]
inventory += [(label,runs[name],2) for name,label in zip(run_names,
              ["Subset quality only","Subset word KL 0.1","Subset word KL 0.5",
               "Broad word KL","Programming word KL","Gaussian MMD v1",
               "Calibrated Gaussian","Calibrated energy","Energy continuation"])]
inventory += [("Distribution only",diag,2)]
rows=[]
for label,d,gpus in inventory:
    n=d["training_updates"]; answers=d["training_passages"]
    if label=="Energy continuation":
        n=d["training_updates_this_run"];answers=d["training_passages_this_run"]
    rows.append([label,str(gpus),f"{n:,}",f"{answers:,}",f'{d["elapsed_seconds"]/60:.1f}'])
table("text-inventory-new", "Achieved text-policy budgets, not configured update ceilings. "
      "Minutes are each comparison record's measured process time, not scheduler allocation limits. "
      "The energy continuation adds 147 updates to 53 existing updates; its cumulative total is 200.",
      "tab:text-inventory-new", ["Policy trial","H200s","Updates","Training answers","Minutes"],rows,size="footnotesize")

# Figures use question-level paired bootstrap, not answer-level pseudo-replication.
plt.rcParams.update({"font.family":"DejaVu Sans","font.size":9,"axes.spines.top":False,"axes.spines.right":False})
fig,axes=plt.subplots(1,3,figsize=(9.1,3.2),sharey=True)
for ax,metric,title in zip(axes,["rbf","energy","quality"],["Gaussian discrepancy ↓","Energy discrepancy ↓","Quality reward ↑"]):
    for i,name in enumerate(semantic_names):
        val=intervals[name][metric];lo,hi=val["ci95"];center=val["mean_delta"]
        ax.errorbar(center,i,xerr=[[center-lo],[hi-center]],fmt="o",color="#23658a",capsize=3)
    ax.axvline(0,color="#777777",ls="--",lw=.8);ax.set_title(title,fontsize=10);ax.set_xlabel("Final − base (95% interval)")
axes[0].set_yticks(range(4),["Gaussian v1 / 94","Cal. Gaussian / 50","Energy / 53","Energy / 200"])
axes[0].invert_yaxis();fig.tight_layout();fig.savefig(PAPER/"figures/semantic-changes.pdf");plt.close(fig)

steps=[0,53,100,150,200]; trajectory=[]
for step in steps:
    path=help_root+("kcgm-gpu-comparison/energy/validation-0-metrics.json" if step==0 else f"energy-long/validation-{step}-metrics.json")
    trajectory.append(record(path))
fig,axes=plt.subplots(1,2,figsize=(8,2.8))
for ax,key,title in zip(axes,["conditional_energy_u_unscaled","quality_reward_mean"],["Validation energy distance ↓","Validation quality reward ↑"]):
    ax.plot(steps,[v[key] for v in trajectory],"o-",color="#23658a");ax.axvline(53,color="#777777",ls="--",lw=.8)
    ax.set_xlabel("Cumulative updates");ax.set_title(title,fontsize=10)
fig.tight_layout();fig.savefig(PAPER/"figures/energy-validation.pdf");plt.close(fig)

with (SUP/"semantic-question-statistics.csv").open("w") as handle:
    writer=csv.DictWriter(handle,fieldnames=list(question_records[0]));writer.writeheader();writer.writerows(question_records)
summary={"revision_date":"2026-11-22","bootstrap":{"seed":20261122,"resamples":10000,"unit":"question"},
         "semantic_intervals":intervals,"diagnostic_intervals":diag_ci}
(SUP/"analysis-summary.json").write_text(json.dumps(summary,indent=2)+"\n")
(SUP/"source-records.json").write_text(json.dumps(sources,indent=2,sort_keys=True)+"\n")
print(json.dumps(summary,indent=2))
print(f"Read and hashed {len(sources)} source records; regenerated 12 tables and 2 vector figures.")
