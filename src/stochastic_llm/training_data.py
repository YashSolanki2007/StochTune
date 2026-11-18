"""Completion-only examples and padding that preserves real EOS labels."""

import torch
from torch.utils.data import Dataset

from .io import read_jsonl


def encode_example(tokenizer, prompt, completion, max_length):
    prompt_ids = tokenizer.encode(prompt, add_special_tokens=False)
    completion_ids = tokenizer.encode(completion, add_special_tokens=False) + [
        tokenizer.eos_token_id
    ]
    ids = prompt_ids + completion_ids
    if not prompt_ids or len(ids) > max_length:
        raise ValueError(
            f"Example has {len(ids)} tokens (limit {max_length}); refusing to truncate"
        )
    return {"input_ids": ids, "labels": [-100] * len(prompt_ids) + completion_ids}


class CompletionDataset(Dataset):
    def __init__(self, path, tokenizer, max_length):
        self.examples = [
            encode_example(tokenizer, r["prompt"], r["completion"], max_length)
            for r in read_jsonl(path)
        ]
        if not self.examples:
            raise ValueError(f"Empty dataset: {path}")

    def __len__(self):
        return len(self.examples)

    def __getitem__(self, index):
        return self.examples[index]


class CompletionCollator:
    def __init__(self, pad_token_id):
        self.pad_token_id = pad_token_id

    def __call__(self, examples):
        length = max(len(x["input_ids"]) for x in examples)
        ids, labels, masks = [], [], []
        for example in examples:
            size = len(example["input_ids"])
            ids.append(example["input_ids"] + [self.pad_token_id] * (length - size))
            labels.append(example["labels"] + [-100] * (length - size))
            masks.append([1] * size + [0] * (length - size))
        return {
            "input_ids": torch.tensor(ids),
            "labels": torch.tensor(labels),
            "attention_mask": torch.tensor(masks),
        }
