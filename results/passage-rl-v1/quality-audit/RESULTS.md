# Shakespeare quality-judge pilot

Frozen-model diagnostic only. No RL or SFT was performed; scores have not
been validated by humans. The authored anchors are assistant-written controls.

| Condition | Valid / requested | Coherence | Fluency | Task | Style | Quality |
|---|---:|---:|---:|---:|---:|---:|
| authored_anchor | 6/6 | 4.67 | 4.50 | 5.00 | 4.00 | 0.93 |
| markov_bigram | 1/6 | 2.00 | 2.00 | 3.00 | 3.00 | 0.33 |
| off_prompt | 2/6 | 4.00 | 4.00 | 4.50 | 4.00 | 0.79 |
| pretrained_base | 7/12 | 3.57 | 3.00 | 4.43 | 3.43 | 0.67 |
| repetition | 0/6 | — | — | — | — | — |
| shuffled_words | 4/6 | 2.00 | 2.00 | 3.25 | 3.00 | 0.35 |
| source_excerpt | 6/6 | 4.83 | 4.50 | — | 4.17 | — |

## Perturbation checks

- shuffled_words: anchor strictly preferred on the specified dimension in 4/4 valid comparisons.
- repetition: anchor strictly preferred on the specified dimension in 0/0 valid comparisons.
- off_prompt: anchor strictly preferred on the specified dimension in 1/2 valid comparisons.

## Distribution diagnostics

Counts pool within passages only, using lowercase word bigrams. Full
reference support plus UNKNOWN uses total pseudocount mass 1. Estimates
are sparse and length-sensitive; they are not ground-truth quality scores.
Raw KL, JSD, copying-window overlap, and repetition are in
`distribution-diagnostics.json`. Source excerpts naturally overlap the corpus.
The contiguous corpus split is not play-disjoint or pretraining-clean.

## Limits

- Only six prompts and one judge; no quality ground truth.
- Same-family actor/judge; independent audit still needed.
- Corpus excerpts lack task prompts; overall reward omitted.
- Bigram estimates are sparse finite-batch diagnostics.
