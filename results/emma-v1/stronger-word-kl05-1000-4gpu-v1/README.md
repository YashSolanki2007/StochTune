# Single stronger-word-KL run: job 535088

Submitted November 27, 2026 (Pacific) after explicit user approval. Target:
1,000 REINFORCE steps / 16,000 training answers, four H200s, five-hour maximum
(20 GPU-hours). See status-snapshot.json for the observation time and scheduler
state; it is not live status. No model results were available at submission.

Weights: quality 1; word-distribution KL **0.5**; length **0.2 per 100 words
outside 250–350**; reference-policy KL **0.002**; nontermination penalty **0.1**.
This supersedes the unsubmitted 0.3 draft. It is one fresh actor, not a grid or
continuation. New 1,000-step cosine schedule, 50 warmup steps, learning rate
5e-6, existing LoRA rank 16 / alpha 32. Global batch 16 = four prompts × four
answers. Judge, target PMF, models/data revisions and sampling are unchanged.
Checkpoints every 25 and validation every 50; 30-minute final-reporting reserve.
The time deadline can stop training before the 1,000-step target.

All **50 CPU tests passed on Tillicum**. The installed sequence-likelihood
implementation matches the local source. W&B project access was verified;
online W&B logging is mandatory during both GPU stages. No automatic retries,
continuation, grid jobs or additional seeds are authorized/submitted.

Validation uses eight prompts × four answers; final probes use sixteen test
prompts × four answers. Judge quality is not independent human evaluation,
existing test prompts have been inspected, and changing both training horizon
and word-KL weight prevents attributing any change to just one factor.
The old 500-step runs and 100-step pilot remain recorded separately.
