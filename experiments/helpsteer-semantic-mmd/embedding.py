"""Frozen MiniLM answer embeddings with full-text, nonoverlapping chunk pooling."""

import numpy as np


def content_chunks(token_ids, content_limit):
    if content_limit < 1:
        raise ValueError("Chunk size must be positive")
    return [token_ids[i : i + content_limit] for i in range(0, len(token_ids), content_limit)] or [
        []
    ]


class AnswerEncoder:
    def __init__(self, cfg, device="cpu"):
        import torch
        from transformers import AutoModel, AutoTokenizer

        self.device = device
        self.batch_size = cfg["semantic_encoder_batch_size"]
        self.max_length = cfg["semantic_encoder_max_tokens"]
        kwargs = {"revision": cfg["semantic_encoder_revision"], "local_files_only": True}
        source = cfg.get("semantic_encoder_local_path", cfg["semantic_encoder_model"])
        self.tokenizer = AutoTokenizer.from_pretrained(source, **kwargs)
        # We tokenize full answers before explicit chunking, never send an overlong
        # sequence to the model. Disable the tokenizer's pre-chunk warning only.
        self.tokenizer.model_max_length = 10**9
        self.model = AutoModel.from_pretrained(source, **kwargs)
        self.model.to(device=device, dtype=torch.float32).eval().requires_grad_(False)
        self.content_limit = self.max_length - self.tokenizer.num_special_tokens_to_add(pair=False)

    def encode(self, texts):
        import torch
        import torch.nn.functional as F

        if not texts:
            return np.empty((0, self.model.config.hidden_size), dtype=np.float32)
        jobs, owners, weights = [], [], []
        for i, text in enumerate(texts):
            ids = self.tokenizer.encode(text, add_special_tokens=False, truncation=False)
            for chunk in content_chunks(ids, self.content_limit):
                jobs.append(
                    self.tokenizer.prepare_for_model(
                        chunk,
                        add_special_tokens=True,
                        truncation=False,
                        return_token_type_ids=False,
                    )
                )
                owners.append(i)
                weights.append(max(1, len(chunk)))
        vectors = []
        for start in range(0, len(jobs), self.batch_size):
            batch = self.tokenizer.pad(jobs[start : start + self.batch_size], return_tensors="pt")
            if batch.input_ids.shape[1] > self.max_length:
                raise ValueError("Embedding input exceeds chunk limit")
            batch = batch.to(self.device)
            with torch.inference_mode():
                hidden = self.model(**batch).last_hidden_state
                mask = batch.attention_mask.unsqueeze(-1).to(hidden.dtype)
                pooled = (hidden * mask).sum(1) / mask.sum(1).clamp_min(1)
                vectors.extend(F.normalize(pooled, p=2, dim=1).cpu().numpy())
        result = np.zeros((len(texts), self.model.config.hidden_size), dtype=np.float64)
        for owner, weight, vector in zip(owners, weights, vectors):
            result[owner] += weight * vector
        norms = np.linalg.norm(result, axis=1, keepdims=True)
        if not np.all(np.isfinite(result)) or np.any(norms < 1e-12):
            raise ValueError("Invalid answer embeddings")
        return (result / norms).astype(np.float32)
