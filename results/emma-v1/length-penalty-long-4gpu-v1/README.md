# Longer length-penalty run: job 530088

Submitted November 26, 2026 after the user requested the longer four-GPU run.
Target: 500 REINFORCE updates / 8,000 training answers on four H200s, capped
at three hours / 12 GPU-hours. No automatic retries or continuation. See
status-snapshot.json for the latest recorded state; it is not a live status.
At submission there were no trained-policy results for this run.

Fresh Qwen2.5-3B-Instruct actor, LoRA rank 16 / alpha 32, AdamW lr 5e-6,
50-step warmup and 500-step cosine schedule. This does not resume the saved
two-GPU pilot. Global batch 16 = four prompts × four answers. Length penalty
0.2 per 100 words outside 250–350, applied before prompt-LOO quality credit.
Word-distribution lambda 0.1 and reference-policy beta 0.002 are unchanged.
Online W&B is required. All 43 CPU checks passed on Tillicum before submission.
Checkpoints every 25 steps, validation every 50; 30 minutes reserved for final
evaluation. The deadline can stop updates before the 500-step target.

comparison-plan.json records parameters matched to unpenalized job 524624.
Evaluate raw judge quality, band compliance, and equal-word-budget KL/JSD
separately. The fixed final scheduled checkpoint is the comparison target;
intermediate validation probes are diagnostics, not post hoc model selection.
Reused test prompts, one seed, and the training judge make the comparison
exploratory, not a new independent benchmark. The two-GPU pilot checkpoint
and results remain saved separately.

## Interim validation: November 27, step 250

The run remains active (279/500 updates at the saved snapshot). These are
32 answers on eight validation prompts, not final test results. Raw judge
quality is 0.808338 → 0.869651; its prompt-bootstrap change interval is
[0.037465, 0.085005] (10,000 draws, seed 20261127). Equal-9,488-word-budget
KL is 0.736907 → 0.707900 (3.94% lower; 2,000 hypergeometric subsamples).
Raw KL is 0.736907 → 0.674056 and JSD 0.084627 → 0.083646.

Length compliance is 20/32 → 21/32, but answers over 350 words increase
5/32 → 9/32; mean words increase 296.5 → 338.5 and mean penalty
0.015813 → 0.042125. Thus quality/distribution point estimates are encouraging,
while length control is mixed. The training judge is not independent human
evaluation; count matching is a sensitivity diagnostic, not KL bias correction.
`interim-250-analysis.json` records settings, source hashes and limitations.
No checkpoint is selected or hyperparameter changed from these interim probes.
