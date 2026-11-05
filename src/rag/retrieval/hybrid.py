"""Framework-free BM25, reciprocal-rank fusion, and hybrid retrieval."""

from __future__ import annotations

import math
from collections import Counter
from typing import Protocol

import numpy as np
from numpy.typing import NDArray

from rag.reranking import Reranker
from rag.reranking.reranker import tokenize
from rag.retrieval.index import VectorIndex
from rag.schemas import DocumentChunk, RetrievedChunk


def _chunk_key(chunk: DocumentChunk) -> tuple[str, object, object, str]:
    metadata = chunk.metadata
    return (
        str(metadata.get("source") or metadata.get("file_name") or ""),
        metadata.get("page"),
        metadata.get("chunk_index"),
        chunk.text,
    )


class BM25Index:
    """Small in-memory Okapi BM25 index implemented from first principles."""

    def __init__(
        self,
        chunks: list[DocumentChunk] | tuple[DocumentChunk, ...],
        k1: float = 1.5,
        b: float = 0.75,
    ) -> None:
        if k1 <= 0 or not 0 <= b <= 1:
            raise ValueError("BM25 requires k1 > 0 and 0 <= b <= 1")
        self.chunks = list(chunks)
        self.k1 = k1
        self.b = b
        self._tokens = [tokenize(chunk.text) for chunk in self.chunks]
        self._frequencies = [Counter(tokens) for tokens in self._tokens]
        self._document_frequency = Counter(
            term for tokens in self._tokens for term in set(tokens)
        )
        self._average_length = (
            sum(map(len, self._tokens)) / len(self._tokens) if self._tokens else 0.0
        )

    def search(self, query: str, top_k: int = 20) -> list[RetrievedChunk]:
        if top_k <= 0:
            raise ValueError("top_k must be greater than zero")
        if not self.chunks:
            return []
        query_terms = tokenize(query)
        scores = np.zeros(len(self.chunks), dtype=np.float32)
        document_count = len(self.chunks)

        for index, frequencies in enumerate(self._frequencies):
            document_length = len(self._tokens[index])
            for term in query_terms:
                frequency = frequencies.get(term, 0)
                if not frequency:
                    continue
                document_frequency = self._document_frequency[term]
                inverse_document_frequency = math.log(
                    1 + (document_count - document_frequency + 0.5)
                    / (document_frequency + 0.5)
                )
                length_ratio = (
                    document_length / self._average_length
                    if self._average_length
                    else 0.0
                )
                denominator = frequency + self.k1 * (1 - self.b + self.b * length_ratio)
                scores[index] += inverse_document_frequency * (
                    frequency * (self.k1 + 1) / denominator
                )

        ranked = [index for index in np.argsort(-scores, kind="stable") if scores[index] > 0]
        return [
            RetrievedChunk(self.chunks[int(index)], float(scores[index]))
            for index in ranked[:top_k]
        ]


def reciprocal_rank_fusion(
    rankings: list[list[RetrievedChunk]],
    top_k: int = 20,
    rank_constant: int = 60,
) -> list[RetrievedChunk]:
    """Fuse ranked result lists without requiring comparable raw scores."""
    if top_k <= 0 or rank_constant <= 0:
        raise ValueError("top_k and rank_constant must be greater than zero")
    fused_scores: dict[tuple[str, object, object, str], float] = {}
    chunks: dict[tuple[str, object, object, str], DocumentChunk] = {}
    for ranking in rankings:
        for rank, result in enumerate(ranking, start=1):
            key = _chunk_key(result.chunk)
            chunks[key] = result.chunk
            fused_scores[key] = fused_scores.get(key, 0.0) + 1.0 / (
                rank_constant + rank
            )
    ordered = sorted(fused_scores, key=fused_scores.get, reverse=True)[:top_k]
    return [RetrievedChunk(chunks[key], fused_scores[key]) for key in ordered]


class QueryEmbedder(Protocol):
    def embed_query(self, query: str) -> NDArray[np.float32]: ...


class HybridRetriever:
    """Combine semantic and lexical recall, then rerank for precision."""

    def __init__(
        self,
        embedder: QueryEmbedder,
        vector_index: VectorIndex,
        reranker: Reranker | None = None,
        similarity_threshold: float = 0.0,
        candidate_k: int = 20,
    ) -> None:
        if not -1 <= similarity_threshold <= 1:
            raise ValueError("similarity_threshold must be between -1 and 1")
        if candidate_k <= 0:
            raise ValueError("candidate_k must be greater than zero")
        self.embedder = embedder
        self.vector_index = vector_index
        self.lexical_index = BM25Index(vector_index.chunks)
        self.reranker = reranker or Reranker()
        self.similarity_threshold = similarity_threshold
        self.candidate_k = candidate_k

    def retrieve(self, query: str, top_k: int = 5) -> list[RetrievedChunk]:
        if top_k <= 0:
            raise ValueError("top_k must be greater than zero")
        query_embedding = self.embedder.embed_query(query)
        vector_results = [
            result
            for result in self.vector_index.search(query_embedding, self.candidate_k)
            if result.score >= self.similarity_threshold
        ]
        lexical_results = self.lexical_index.search(query, self.candidate_k)
        candidates = reciprocal_rank_fusion(
            [vector_results, lexical_results], top_k=self.candidate_k
        )
        return self.reranker.rerank(query, candidates, top_k=top_k)

