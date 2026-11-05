import numpy as np
import pytest

from rag.embeddings import Embedder


class FakeModel:
    def __init__(self) -> None:
        self.calls: list[list[str]] = []

    def encode(self, texts: list[str], **_: object) -> np.ndarray:
        self.calls.append(texts)
        return np.asarray(
            [[len(text), sum(map(ord, text)) % 11, 1.0] for text in texts],
            dtype=np.float32,
        )

    def get_sentence_embedding_dimension(self) -> int:
        return 3


def test_document_embeddings_are_batched_normalized_and_finite() -> None:
    model = FakeModel()
    embedder = Embedder(model=model, batch_size=2)

    vectors = embedder.embed_documents(["one", "two", "three", "four", "five"])

    assert vectors.shape == (5, 3)
    assert [len(call) for call in model.calls] == [2, 2, 1]
    assert np.all(np.isfinite(vectors))
    assert np.allclose(np.linalg.norm(vectors, axis=1), 1.0)


def test_query_and_document_dimensions_match() -> None:
    embedder = Embedder(model=FakeModel())

    documents = embedder.embed_documents(["document"])
    query = embedder.embed_query("question")

    assert query.shape == (documents.shape[1],)


def test_empty_input_returns_empty_matrix_with_known_dimension() -> None:
    vectors = Embedder(model=FakeModel()).embed_documents([])

    assert vectors.shape == (0, 3)


def test_invalid_batch_size_raises() -> None:
    with pytest.raises(ValueError, match="batch_size"):
        Embedder(model=FakeModel(), batch_size=0)


def test_empty_query_raises() -> None:
    with pytest.raises(ValueError, match="empty"):
        Embedder(model=FakeModel()).embed_query("  ")


def test_non_finite_model_output_raises() -> None:
    class BrokenModel(FakeModel):
        def encode(self, texts: list[str], **_: object) -> np.ndarray:
            return np.asarray([[np.nan, 0.0]], dtype=np.float32)

    with pytest.raises(ValueError, match="non-finite"):
        Embedder(model=BrokenModel()).embed_documents(["document"])

