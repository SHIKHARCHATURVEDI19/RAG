"""Local sentence-transformer embedding adapter."""

from __future__ import annotations

import os
from typing import Any

import numpy as np
from numpy.typing import NDArray

DEFAULT_MODEL = "sentence-transformers/all-MiniLM-L6-v2"


class Embedder:
    """Encode text with a lazily loaded local Sentence Transformers model."""

    def __init__(
        self,
        model_name: str | None = None,
        batch_size: int = 32,
        model: Any | None = None,
    ) -> None:
        if batch_size <= 0:
            raise ValueError("batch_size must be greater than zero")
        self.model_name = model_name or os.getenv("EMBEDDING_MODEL", DEFAULT_MODEL)
        self.batch_size = batch_size
        self._model = model

    @property
    def model(self) -> Any:
        """Load the model only when embeddings are first requested."""
        if self._model is None:
            from sentence_transformers import SentenceTransformer

            self._model = SentenceTransformer(self.model_name)
        return self._model

    @staticmethod
    def _normalize(vectors: NDArray[np.floating[Any]]) -> NDArray[np.float32]:
        matrix = np.asarray(vectors, dtype=np.float32)
        if matrix.ndim == 1:
            matrix = matrix.reshape(1, -1)
        if matrix.ndim != 2:
            raise ValueError("Embedding model must return a two-dimensional array")
        if not np.all(np.isfinite(matrix)):
            raise ValueError("Embedding model returned non-finite values")

        norms = np.linalg.norm(matrix, axis=1, keepdims=True)
        return np.divide(
            matrix,
            norms,
            out=np.zeros_like(matrix),
            where=norms > 0,
        )

    def embed_documents(self, texts: list[str]) -> NDArray[np.float32]:
        """Embed documents in bounded batches and return normalized vectors."""
        if not texts:
            dimension_getter = getattr(self.model, "get_sentence_embedding_dimension", None)
            dimension = int(dimension_getter()) if dimension_getter else 0
            return np.empty((0, dimension), dtype=np.float32)
        if any(not isinstance(text, str) for text in texts):
            raise TypeError("Every document must be a string")

        batches: list[NDArray[np.float32]] = []
        for start in range(0, len(texts), self.batch_size):
            encoded = self.model.encode(
                texts[start : start + self.batch_size],
                convert_to_numpy=True,
                show_progress_bar=False,
            )
            batches.append(self._normalize(encoded))
        return np.vstack(batches)

    def embed_query(self, query: str) -> NDArray[np.float32]:
        """Embed one query using the same normalization as documents."""
        if not isinstance(query, str):
            raise TypeError("query must be a string")
        if not query.strip():
            raise ValueError("query cannot be empty")
        return self.embed_documents([query])[0]

