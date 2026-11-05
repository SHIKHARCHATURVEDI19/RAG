"""In-memory NumPy vector index with disk persistence."""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any

import numpy as np
from numpy.typing import NDArray

from rag.schemas import DocumentChunk, RetrievedChunk


class VectorIndex:
    """Store normalized embeddings and perform exact cosine search."""

    FORMAT_VERSION = 1

    def __init__(self) -> None:
        self._embeddings = np.empty((0, 0), dtype=np.float32)
        self._chunks: list[DocumentChunk] = []

    @property
    def size(self) -> int:
        return len(self._chunks)

    @property
    def dimension(self) -> int:
        return self._embeddings.shape[1] if self._embeddings.ndim == 2 else 0

    @property
    def chunks(self) -> tuple[DocumentChunk, ...]:
        return tuple(self._chunks)

    @staticmethod
    def _normalize(embeddings: NDArray[np.floating[Any]]) -> NDArray[np.float32]:
        matrix = np.asarray(embeddings, dtype=np.float32)
        if matrix.ndim == 1:
            matrix = matrix.reshape(1, -1)
        if matrix.ndim != 2:
            raise ValueError("embeddings must be a two-dimensional array")
        if not np.all(np.isfinite(matrix)):
            raise ValueError("embeddings contain non-finite values")
        norms = np.linalg.norm(matrix, axis=1, keepdims=True)
        return np.divide(
            matrix,
            norms,
            out=np.zeros_like(matrix),
            where=norms > 0,
        )

    def add(
        self,
        chunks: list[DocumentChunk],
        embeddings: NDArray[np.floating[Any]],
    ) -> None:
        """Add chunks and their matching embeddings to the index."""
        matrix = self._normalize(embeddings)
        if len(chunks) != matrix.shape[0]:
            raise ValueError("chunk and embedding counts must match")
        if not chunks:
            return
        if self.size and matrix.shape[1] != self.dimension:
            raise ValueError(
                f"embedding dimension {matrix.shape[1]} does not match index "
                f"dimension {self.dimension}"
            )

        self._embeddings = (
            np.vstack([self._embeddings, matrix]) if self.size else matrix.copy()
        )
        self._chunks.extend(chunks)

    def search(
        self,
        query_embedding: NDArray[np.floating[Any]],
        top_k: int = 5,
    ) -> list[RetrievedChunk]:
        """Return the highest cosine-similarity matches."""
        if top_k <= 0:
            raise ValueError("top_k must be greater than zero")
        if not self.size:
            return []

        query = self._normalize(np.asarray(query_embedding, dtype=np.float32))[0]
        if query.shape[0] != self.dimension:
            raise ValueError(
                f"query dimension {query.shape[0]} does not match index "
                f"dimension {self.dimension}"
            )

        scores = self._embeddings @ query
        indices = np.argsort(-scores, kind="stable")[: min(top_k, self.size)]
        return [
            RetrievedChunk(chunk=self._chunks[int(index)], score=float(scores[index]))
            for index in indices
        ]

    def save(self, directory: str | Path) -> None:
        """Persist embeddings, chunks, and index configuration."""
        target = Path(directory)
        target.mkdir(parents=True, exist_ok=True)
        np.save(target / "embeddings.npy", self._embeddings, allow_pickle=False)
        (target / "chunks.json").write_text(
            json.dumps(
                [
                    {"text": chunk.text, "metadata": chunk.metadata}
                    for chunk in self._chunks
                ],
                ensure_ascii=False,
                indent=2,
            ),
            encoding="utf-8",
        )
        (target / "index.json").write_text(
            json.dumps(
                {
                    "format_version": self.FORMAT_VERSION,
                    "size": self.size,
                    "dimension": self.dimension,
                },
                indent=2,
            ),
            encoding="utf-8",
        )

    @classmethod
    def load(cls, directory: str | Path) -> "VectorIndex":
        """Restore an index previously saved by :meth:`save`."""
        source = Path(directory)
        config = json.loads((source / "index.json").read_text(encoding="utf-8"))
        if config.get("format_version") != cls.FORMAT_VERSION:
            raise ValueError("Unsupported vector index format")

        items = json.loads((source / "chunks.json").read_text(encoding="utf-8"))
        chunks = [
            DocumentChunk(text=item["text"], metadata=item.get("metadata", {}))
            for item in items
        ]
        embeddings = np.load(source / "embeddings.npy", allow_pickle=False)

        index = cls()
        index.add(chunks, embeddings)
        if index.size != config.get("size") or index.dimension != config.get("dimension"):
            raise ValueError("Persisted index files are inconsistent")
        return index

