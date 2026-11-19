# Corpus provenance

Shakespeare source text from the pinned char-rnn Tiny Shakespeare snapshot recorded
in manifest.json. reference.txt is the first 80% of paragraphs with at least ten
normalized words; heldout.txt is the remaining eligible paragraphs. candidates.jsonl
contains the controlled evaluator-audit samples, including assistant-written anchors.
These are distinct from the DDP train/validation/test scenario prompts.

Source: https://github.com/karpathy/char-rnn/blob/370cbcd448eb7daf32f21a6be560b70e0b33c4e3/data/tinyshakespeare/input.txt
