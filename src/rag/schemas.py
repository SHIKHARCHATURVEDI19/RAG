"""Shared data structures used throughout the RAG pipeline."""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any


@dataclass(slots=True)
class Document:
    """Text loaded from one source unit, such as a file or PDF page."""

    text: str
    metadata: dict[str, Any] = field(default_factory=dict)


@dataclass(slots=True)
class DocumentChunk:
    """A searchable portion of a document with inherited source metadata."""

    text: str
    metadata: dict[str, Any] = field(default_factory=dict)


@dataclass(slots=True)
class RetrievedChunk:
    """A document chunk paired with its retrieval relevance score."""

    chunk: DocumentChunk
    score: float


@dataclass(slots=True)
class Source:
    """Citation metadata derived from a retrieved chunk."""

    citation_id: int
    source: str
    file: str
    page: int | None = None
    chunk_index: int | None = None
    score: float | None = None


@dataclass(slots=True)
class RAGResult:
    """Public result returned by the end-to-end RAG pipeline."""

    answer: str
    sources: list[Source]
    retrieved_chunks: list[RetrievedChunk]
    metadata: dict[str, Any] = field(default_factory=dict)

    def to_dict(self, include_chunks: bool = False) -> dict[str, Any]:
        """Convert the result into a JSON-serializable dictionary."""
        payload: dict[str, Any] = {
            "answer": self.answer,
            "sources": [
                {
                    "citation_id": source.citation_id,
                    "source": source.source,
                    "file": source.file,
                    "page": source.page,
                    "chunk_index": source.chunk_index,
                    "score": source.score,
                }
                for source in self.sources
            ],
            "metadata": self.metadata,
        }
        if include_chunks:
            payload["retrieved_chunks"] = [
                {
                    "text": result.chunk.text,
                    "metadata": result.chunk.metadata,
                    "score": result.score,
                }
                for result in self.retrieved_chunks
            ]
        return payload
