"""Rerank retrieval candidates with an optional cross-encoder model."""

from __future__ import annotations

import re
from typing import Any

import numpy as np

from rag.schemas import RetrievedChunk


def tokenize(text: str) -> list[str]:
    return re.findall(r"\w+", text.casefold(), flags=re.UNICODE)


class Reranker:
    """Improve final precision using a cross-encoder or lexical fallback."""

    def __init__(self, model: Any | None = None) -> None:
        self.model = model

    def _scores(self, query: str, candidates: list[RetrievedChunk]) -> np.ndarray:
        if self.model is not None:
            pairs = [(query, candidate.chunk.text) for candidate in candidates]
            scores = np.asarray(self.model.predict(pairs), dtype=np.float32)
            if scores.shape != (len(candidates),) or not np.all(np.isfinite(scores)):
                raise ValueError("Reranker model returned invalid scores")
            return scores

        query_terms = set(tokenize(query))
        if not query_terms:
            return np.zeros(len(candidates), dtype=np.float32)
        return np.asarray(
            [
                0.8 * len(query_terms.intersection(tokenize(candidate.chunk.text)))
                / len(query_terms)
                + 0.2 * candidate.score
                for candidate in candidates
            ],
            dtype=np.float32,
        )

    def rerank(
        self,
        query: str,
        candidates: list[RetrievedChunk],
        top_k: int = 5,
    ) -> list[RetrievedChunk]:
        """Return candidates ordered by final relevance."""
        if top_k <= 0:
            raise ValueError("top_k must be greater than zero")
        if not candidates:
            return []
        scores = self._scores(query, candidates)
        order = np.argsort(-scores, kind="stable")[: min(top_k, len(candidates))]
        return [
            RetrievedChunk(candidates[int(index)].chunk, float(scores[index]))
            for index in order
        ]

