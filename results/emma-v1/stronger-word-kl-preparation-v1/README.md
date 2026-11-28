# Single 1,000-step stronger-word-KL run — awaiting approval

Prepared November 27, 2026 after the user requested one run before any grid and
asked to see the weights before starting. No GPU jobs have been submitted or
authorized. Weights: quality 1; word KL 0.3; length 0.2 per 100 words outside
250–350; reference-policy KL 0.002; nontermination penalty 0.1.

Fresh Qwen2.5-3B-Instruct with 1,000-step cosine schedule, 50 warmup steps,
AdamW lr 5e-6 and existing LoRA settings. Global batch 16; 16,000 training
answers. Four H200s, at most five hours / 20 GPU-hours; 30 minutes reserved
for final evaluation. No automatic retries, continuation, grid or extra seeds.
Online W&B is required. Validation remains eight prompts × four answers;
final probes use sixteen test prompts × four answers. Results are exploratory
because test prompts have been inspected and the training judge evaluates them.

All **50 local CPU tests passed**. The standalone staged CPU preflight passed,
and GPU submission remains blocked pending final approval. The code, exact
proposed run configuration, resource cap, tests and provenance are recorded here.
No model results are available for this unrun experiment. Changing both KL
weight and training horizon prevents attributing any improvement to one change.

The user subsequently chose word-KL weight **0.5** and explicitly authorized
submission. This 0.3 draft is preserved as historical preparation and was never
submitted. The separate `stronger-word-kl05-1000-4gpu-v1` record tracks the
approved 0.5 run; no grid is authorized.
