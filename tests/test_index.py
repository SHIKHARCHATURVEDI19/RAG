from pathlib import Path

import numpy as np
import pytest

from rag.retrieval import VectorIndex
from rag.schemas import DocumentChunk


def build_index() -> VectorIndex:
    index = VectorIndex()
    index.add(
        [
            DocumentChunk("east", {"file_name": "directions.txt"}),
            DocumentChunk("north", {"file_name": "directions.txt"}),
            DocumentChunk("west", {"file_name": "directions.txt"}),
        ],
        np.asarray([[1.0, 0.0], [0.0, 1.0], [-1.0, 0.0]], dtype=np.float32),
    )
    return index


def test_search_orders_by_cosine_similarity() -> None:
    results = build_index().search(np.asarray([0.9, 0.1]), top_k=3)

    assert [result.chunk.text for result in results] == ["east", "north", "west"]
    assert results[0].score > results[1].score > results[2].score


def test_search_respects_top_k() -> None:
    results = build_index().search(np.asarray([1.0, 0.0]), top_k=2)

    assert len(results) == 2


def test_empty_index_returns_no_results() -> None:
    assert VectorIndex().search(np.asarray([1.0]), top_k=5) == []


def test_add_validates_counts_and_dimensions() -> None:
    index = build_index()

    with pytest.raises(ValueError, match="counts"):
        index.add([DocumentChunk("one")], np.ones((2, 2), dtype=np.float32))
    with pytest.raises(ValueError, match="dimension"):
        index.add([DocumentChunk("one")], np.ones((1, 3), dtype=np.float32))


def test_save_and_load_round_trip(tmp_path: Path) -> None:
    original = build_index()
    original.save(tmp_path)

    restored = VectorIndex.load(tmp_path)
    results = restored.search(np.asarray([1.0, 0.0]), top_k=1)

    assert restored.size == 3
    assert restored.dimension == 2
    assert results[0].chunk.text == "east"
    assert results[0].chunk.metadata["file_name"] == "directions.txt"


def test_invalid_search_arguments_raise() -> None:
    index = build_index()

    with pytest.raises(ValueError, match="top_k"):
        index.search(np.asarray([1.0, 0.0]), top_k=0)
    with pytest.raises(ValueError, match="dimension"):
        index.search(np.asarray([1.0, 0.0, 0.0]))

