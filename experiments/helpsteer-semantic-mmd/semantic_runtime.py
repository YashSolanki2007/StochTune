"""Runtime bridge for the existing DDP policy and the conditional MMD objective."""

import json

import numpy as np
from common import ROOT, distribution_diagnostics
from embedding import AnswerEncoder
from mmd import conditional_credit, per_question_metrics


class ConditionalSemanticTarget:
    def __init__(self, cfg, word_target, device):
        self.encoder = AnswerEncoder(cfg, device)
        self.word_target = word_target
        archive = np.load(ROOT / "data/semantic-reference-vectors.npz", allow_pickle=False)
        self.references = dict(zip(archive["case_ids"].tolist(), archive["vectors"]))
        metadata = json.loads((ROOT / "data/semantic-metadata.json").read_text())
        self.bandwidth_squared = metadata["bandwidth_squared"]
        self.train_ids = set(metadata["training_prompt_ids"])

    def embed_records(self, rows):
        vectors = self.encoder.encode([row["passage"] for row in rows])
        return [
            {**row, "semantic_embedding": vector.tolist()} for row, vector in zip(rows, vectors)
        ]

    def training_credit(self, rows, quality):
        if any(row["case_id"] not in self.train_ids for row in rows):
            raise ValueError("Held-out semantic references cannot enter policy optimization")
        return conditional_credit(rows, quality, self.references, self.bandwidth_squared)

    def per_prompt_metrics(self, rows):
        return per_question_metrics(rows, self.references, self.bandwidth_squared)

    def diagnostics(self, rows):
        results = list(self.per_prompt_metrics(rows).values())
        metrics = {
            name: float(np.mean([row[name] for row in results]))
            for name in (
                "conditional_mmd2_u",
                "conditional_mmd2_v",
                "reference_prototype_coverage",
                "reference_prototype_one_fraction",
            )
        }
        metrics["mmd_questions"] = len(results)
        return {
            **distribution_diagnostics(self.word_target, [r["passage"] for r in rows]),
            **metrics,
        }
