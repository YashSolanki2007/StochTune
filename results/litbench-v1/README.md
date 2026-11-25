# Cancelled LitBench attempt

Job **519547** was cancelled after eight minutes on two H200s, before RL.
Its initial validation report contained NumPy boolean scalars that Python's
JSON encoder rejected. No usable reward model or policy results were exported.

`submission.json` records the original two-hour/two-GPU submission, **not actual
usage**; `cancellation.json` records observed elapsed time and cancellation.
`data-manifest.json` identifies the pinned dataset, prepared counts and hashes.

The current [implementation](../../experiments/litbench-rl/README.md) converts
validation checks to native booleans and tests serialization of the actual
report. This fix does not imply that the uploaded job was rerun or that its
planned training succeeded. The subsequent Emma experiment is separate.
