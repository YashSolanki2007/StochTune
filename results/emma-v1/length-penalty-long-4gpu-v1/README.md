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
