"""Completion masks and sparse target tries without vocabulary-sized labels."""

from collections import OrderedDict

import numpy as np
import torch
from torch.utils.data import Dataset

from .distributions import finite_target
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


class TargetTrie:
    def __init__(self, token_paths, masses):
        self.children = [{}]
        self.mass = [0.0]
        self.paths = token_paths
        self.probabilities = np.asarray(masses, dtype=np.float64)
        self.probabilities /= self.probabilities.sum()
        for path, mass in zip(token_paths, self.probabilities):
            node = 0
            self.mass[node] += mass
            for token in path:
                if token not in self.children[node]:
                    self.children[node][token] = len(self.children)
                    self.children.append({})
                    self.mass.append(0.0)
                node = self.children[node][token]
                self.mass[node] += mass

    def targets(self, path):
        node = 0
        targets = []
        for token in path:
            children = self.children[node]
            ids = list(children)
            probabilities = [self.mass[children[t]] / self.mass[node] for t in ids]
            targets.append((ids, probabilities))
            node = children[token]
        return targets


class SoftTargetDataset(Dataset):
    def __init__(self, path, tokenizer, config, paths_per_config=None):
        self.cases = read_jsonl(path)
        self.tokenizer = tokenizer
        self.config = config
        self.repeats = paths_per_config or int(config.get("paths_per_config", 3))
        if not self.cases or self.repeats < 1:
            raise ValueError("Empty soft-target dataset or invalid paths_per_config")
        self.cache = OrderedDict()

    def __len__(self):
        return len(self.cases) * self.repeats

    def trie_for(self, case_index):
        if case_index in self.cache:
            self.cache.move_to_end(case_index)
            return self.cache[case_index]
        outputs, masses = finite_target(
            self.cases[case_index],
            decimals=self.config.get("decimals", 5),
            max_bins=self.config.get("max_bins", 1001),
            tail_probability=self.config.get("tail_probability", 0.001),
        )
        paths = self.tokenizer(outputs, add_special_tokens=False)["input_ids"]
        paths = [p + [self.tokenizer.eos_token_id] for p in paths]
        if len({tuple(p) for p in paths}) != len(paths):
            raise ValueError("Tokenizer cannot distinguish all canonical numeric outputs")
        trie = TargetTrie(paths, masses)
        self.cache[case_index] = trie
        if len(self.cache) > int(self.config.get("trie_cache_size", 64)):
            self.cache.popitem(last=False)
        return trie

    def __getitem__(self, index):
        case_index = index % len(self.cases)
        case = self.cases[case_index]
        trie = self.trie_for(case_index)
        # Stateless seeds keep path sampling reproducible across workers/resume.
        rng = np.random.default_rng(np.random.SeedSequence([self.config["seed"], index]))
        path = trie.paths[int(rng.choice(len(trie.paths), p=trie.probabilities))]
        prompt = self.tokenizer.encode(case["prompt"], add_special_tokens=False)
        if len(prompt) + len(path) > self.config["max_sequence_length"]:
            raise ValueError("Soft-target example exceeds max_sequence_length; increase the limit")
        return {
            "input_ids": prompt + path,
            "labels": [-100] * len(prompt) + path,
            "soft_targets": [
                (len(prompt) - 1 + i, ids, probabilities)
                for i, (ids, probabilities) in enumerate(trie.targets(path))
            ],
        }


class CompletionCollator:
    def __init__(self, pad_token_id):
        self.pad_token_id = pad_token_id

    def __call__(self, examples):
        length = max(len(x["input_ids"]) for x in examples)
        input_ids, labels, masks = [], [], []
        sparse = {"batch": [], "position": [], "token": [], "probability": []}
        for batch_index, example in enumerate(examples):
            size = len(example["input_ids"])
            input_ids.append(example["input_ids"] + [self.pad_token_id] * (length - size))
            labels.append(example["labels"] + [-100] * (length - size))
            masks.append([1] * size + [0] * (length - size))
            for position, ids, probabilities in example.get("soft_targets", []):
                sparse["batch"].extend([batch_index] * len(ids))
                sparse["position"].extend([position] * len(ids))
                sparse["token"].extend(ids)
                sparse["probability"].extend(probabilities)
        batch = {
            "input_ids": torch.tensor(input_ids),
            "labels": torch.tensor(labels),
            "attention_mask": torch.tensor(masks),
        }
        if sparse["batch"]:
            batch["soft_targets"] = {
                key: torch.tensor(
                    values, dtype=(torch.float32 if key == "probability" else torch.long)
                )
                for key, values in sparse.items()
            }
        return batch


def soft_kl_loss(logits, targets):
    """Forward KL summed over prefixes, averaged over sampled paths.

    Prefix paths are sampled from P. The chain rule then gives sequence-level
    KL(P || Q) in expectation. Do not renormalize Q over valid children: invalid
    output mass must also be penalized.
    """
    batch, position, token = (targets[key] for key in ("batch", "position", "token"))
    pairs = torch.stack((batch, position), dim=1)
    unique, inverse = torch.unique(pairs, dim=0, return_inverse=True)
    next_logits = logits[unique[:, 0], unique[:, 1]].float()
    log_normalizer = torch.logsumexp(next_logits, dim=-1)
    log_q = next_logits[inverse, token] - log_normalizer[inverse]
    p = targets["probability"]
    # Tiny tail masses can underflow to zero in float32; xlogy defines 0 log 0 = 0.
    return (torch.special.xlogy(p, p) - p * log_q).sum() / logits.shape[0]
