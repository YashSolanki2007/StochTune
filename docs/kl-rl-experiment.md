# Completed batch-KL-reward RL pilot

The pilot continues the completed Qwen2.5-3B SFT LoRA adapter. The direct
soft-target KL supervised trainer is a separate implementation and has not been
run. See [the full implementation guide](implementation.md) for reward equations,
gradient handling, checkpointing, and cluster installation.

## Reproduce the recorded analysis without a GPU

```bash
python3.12 -m venv .venv
source .venv/bin/activate
python -m pip install -e '.[dev,analysis]'
python -m pytest -q
python scripts/analyze_rl_results.py --recorded
```

The last command reads all 200,000 committed SFT/RL test generations and writes
the report, figure, per-prompt metrics, and bootstrap comparisons to
`outputs/rl-results/`. It does not allocate a GPU or create a W&B run. Omit
`--recorded` to analyze fresh outputs in the original cluster directory layout.

The snapshot in `results/rl-v1/` contains all SFT/RL test responses, evaluation
identities, configuration and dataset hashes, training metrics/history, aggregate
12-case validation metrics at recorded checkpoints, and the comparison report.
`provenance.json` records the completed jobs, W&B URLs, and SHA-256 hashes of the
snapshot files. Training rollouts, trained adapters, and resumable optimizer
checkpoints remain in cluster storage and W&B artifacts; they are not Git files.

## Training setup

| Setting | Value |
|---|---|
| Base model | Qwen/Qwen2.5-3B |
| Base revision | `3aab1f1954e9cc14eb9509a215f9e5ca08227a9b` |
| Initial policy | Completed 2,000-update SFT LoRA adapter |
| Trainable parameters | Existing attention LoRA adapters, rank 16 / alpha 32 |
| Algorithm | On-policy REINFORCE, leave-one-out group reward baseline |
| Target reward | Negative forward histogram KL, up to 32 bins, pseudocount 0.1 |
| Reference regularization | Reverse sequence KL to frozen initial SFT, coefficient 0.02 |
| Budget | 250 updates × 2 prompts × 256 independent draws = 128,000 responses |
| Learning rate | `1e-5`, cosine decay, 3% warmup |
| Rollout batch / gradient microbatch | 128 / 16 |
| Validation | 12 disjoint configurations × 256 draws, every 25 updates |
| Hardware | One NVIDIA H200, BF16 |
| Seed | 42 |

Dropout is disabled for rollouts and recomputation. Each generated batch is used
for one update. No target numeric answer labels are used during RL. A batch-wide
shared reward is not normalized as in ordinary GRPO, because this would erase
its learning signal. The final adapter at update 250 was tested; the retained
best validation adapter at update 225 was not used for the test comparison.

## Run a new experiment on Tillicum

First follow the installation, data generation, model cache, and W&B login steps
in [the implementation guide](implementation.md). Obtain the initial SFT
adapter by reproducing SFT or restoring its W&B artifact. Keep credentials in
the user's login environment, never in config files or Git.

After approval for a new GPU experiment, from the cluster project directory:

```bash
bash scripts/submit_kl_rl.sh YOUR_SLURM_ACCOUNT
```

The helper chains a 20-update timing benchmark, a fresh 250-update RL run, and
100,000 final test generations. It requires online W&B logging. Existing output
directories are protected, so archive them or choose new destinations before
submitting another experiment. The recorded run used jobs 493753, 493754, and
493755; all completed successfully. Runtime was 2m20s for the benchmark, 22m34s
for training, and 5m43s for evaluation, excluding queue time.

## Findings and limits

Mean test histogram KL fell 2.24%, median KL 11.29%, median KS 8.90%, and median
normalized Wasserstein 9.76%. Numeric response validity changed from 98.677% to
98.591%. Benefits were mixed across families, and the unseen-family aggregate
worsened slightly on median KS and Wasserstein.

Test KL uses 1,000 draws per prompt; training and validation use 256. These
finite-sample, smoothed estimates should not be compared directly in absolute
value. KS and Wasserstein use valid numeric answers while retaining numeric
support violations; histogram KL also penalizes malformed, truncated, and
out-of-support responses through an invalid category.

One training seed and no equal-compute SFT continuation control limit causal
claims. The prompt bootstrap in the report measures variation across these
test prompts, not training seeds. This pilot does not establish exact sampling,
independence, cryptographic randomness, or reliable unseen-family transfer.
