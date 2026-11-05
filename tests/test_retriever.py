import numpy as np
import pytest

from rag.retrieval import Retriever, VectorIndex
from rag.schemas import DocumentChunk


class FakeEmbedder:
    def embed_query(self, query: str) -> np.ndarray:
        if not query.strip():
            raise ValueError("query cannot be empty")
        return np.asarray([1.0, 0.0], dtype=np.float32)


def build_retriever(threshold: float = -1.0) -> Retriever:
    index = VectorIndex()
    index.add(
        [
            DocumentChunk("best match"),
            DocumentChunk("BEST   MATCH"),
            DocumentChunk("middle match"),
            DocumentChunk("irrelevant"),
        ],
        np.asarray(
            [[1.0, 0.0], [0.99, 0.01], [0.6, 0.8], [-1.0, 0.0]],
            dtype=np.float32,
        ),
    )
    return Retriever(FakeEmbedder(), index, similarity_threshold=threshold)


def test_results_are_ordered_and_deduplicated() -> None:
    results = build_retriever().retrieve("query", top_k=3)

    assert [result.chunk.text for result in results] == [
        "best match",
        "middle match",
        "irrelevant",
    ]
    assert results[0].score >= results[1].score >= results[2].score


def test_top_k_is_respected_after_deduplication() -> None:
    results = build_retriever().retrieve("query", top_k=1)

    assert len(results) == 1
    assert results[0].chunk.text == "best match"


def test_threshold_filters_results() -> None:
    results = build_retriever(threshold=0.7).retrieve("query", top_k=5)

    assert [result.chunk.text for result in results] == ["best match"]


def test_empty_result_is_supported() -> None:
    results = Retriever(
        FakeEmbedder(), VectorIndex(), similarity_threshold=0.5
    ).retrieve("query", top_k=5)

    assert results == []


def test_invalid_configuration_raises() -> None:
    with pytest.raises(ValueError, match="threshold"):
        build_retriever(threshold=1.1)
    with pytest.raises(ValueError, match="top_k"):
        build_retriever().retrieve("query", top_k=0)
