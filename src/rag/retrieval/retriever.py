"""Semantic retrieval orchestration."""

from __future__ import annotations

import re
from typing import Protocol

import numpy as np
from numpy.typing import NDArray

from rag.retrieval.index import VectorIndex
from rag.schemas import RetrievedChunk


class QueryEmbedder(Protocol):
    def embed_query(self, query: str) -> NDArray[np.float32]: ...


def _deduplication_key(text: str) -> str:
    return re.sub(r"\s+", " ", text).strip().casefold()


class Retriever:
    """Embed a query and return filtered, deduplicated vector matches."""

    def __init__(
        self,
        embedder: QueryEmbedder,
        index: VectorIndex,
        similarity_threshold: float = 0.5,
    ) -> None:
        if not -1.0 <= similarity_threshold <= 1.0:
            raise ValueError("similarity_threshold must be between -1 and 1")
        self.embedder = embedder
        self.index = index
        self.similarity_threshold = similarity_threshold

    def retrieve(self, query: str, top_k: int = 5) -> list[RetrievedChunk]:
        """Retrieve the most relevant unique chunks for a query."""
        if top_k <= 0:
            raise ValueError("top_k must be greater than zero")
        query_embedding = self.embedder.embed_query(query)
        candidates = self.index.search(query_embedding, top_k=max(top_k, self.index.size))

        results: list[RetrievedChunk] = []
        seen: set[str] = set()
        for candidate in candidates:
            if candidate.score < self.similarity_threshold:
                continue
            key = _deduplication_key(candidate.chunk.text)
            if not key or key in seen:
                continue
            seen.add(key)
            results.append(candidate)
            if len(results) == top_k:
                break
        return results

