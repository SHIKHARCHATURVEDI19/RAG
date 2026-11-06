from pathlib import Path

import numpy as np

from rag.config import Settings
from rag.generation import Generator
from rag.runtime import RAGService


class FakeEmbedder:
    def _vector(self, text: str) -> np.ndarray:
        terms = text.casefold()
        return np.asarray(
            [terms.count("refund") + 1, terms.count("shipping"), 1.0],
            dtype=np.float32,
        )

    def embed_documents(self, texts: list[str]) -> np.ndarray:
        vectors = np.vstack([self._vector(text) for text in texts])
        return vectors / np.linalg.norm(vectors, axis=1, keepdims=True)

    def embed_query(self, query: str) -> np.ndarray:
        vector = self._vector(query)
        return vector / np.linalg.norm(vector)


def test_service_ingests_persists_and_queries(tmp_path: Path) -> None:
    documents = tmp_path / "documents"
    documents.mkdir()
    (documents / "policy.md").write_text(
        "Refund requests are accepted for 30 days.", encoding="utf-8"
    )
    settings = Settings(
        index_directory=tmp_path / "index",
        chunk_size=100,
        chunk_overlap=10,
        similarity_threshold=-1.0,
    )
    service = RAGService(settings, embedder=FakeEmbedder(), generator=Generator())

    summary = service.ingest(documents)
    result = service.query("What is the refund period?")

    assert summary["documents"] == 1
    assert summary["chunks"] == 1
    assert (settings.index_directory / "index.json").exists()
    assert result.sources[0].file == "policy.md"
    assert "30 days" in result.answer


def test_service_loads_existing_index(tmp_path: Path) -> None:
    document = tmp_path / "policy.txt"
    document.write_text("Refunds last 30 days.", encoding="utf-8")
    settings = Settings(
        index_directory=tmp_path / "index", similarity_threshold=-1.0
    )
    RAGService(settings, embedder=FakeEmbedder()).ingest(document)

    restored = RAGService(settings, embedder=FakeEmbedder())

    assert restored.index.size == 1

