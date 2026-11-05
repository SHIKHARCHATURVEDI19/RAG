import pytest

from rag.ingestion.chunker import chunk_document, chunk_text
from rag.schemas import Document


def test_empty_document_has_no_chunks() -> None:
    assert chunk_document(Document(text="")) == []


def test_small_document_has_one_chunk() -> None:
    chunks = chunk_document(
        Document(text="small", metadata={"source": "notes.txt"}),
        chunk_size=10,
        overlap=2,
    )

    assert len(chunks) == 1
    assert chunks[0].text == "small"
    assert chunks[0].metadata == {"source": "notes.txt", "chunk_index": 0}


def test_exact_boundary_has_one_chunk() -> None:
    assert chunk_text("abcdefghij", chunk_size=10, overlap=2) == ["abcdefghij"]


def test_large_document_uses_overlap() -> None:
    chunks = chunk_text("abcdefghijklmnop", chunk_size=10, overlap=4)

    assert chunks == ["abcdefghij", "ghijklmnop"]
    assert chunks[0][-4:] == chunks[1][:4]


@pytest.mark.parametrize(
    ("chunk_size", "overlap"),
    [(0, 0), (-1, 0), (10, -1), (10, 10), (10, 11)],
)
def test_invalid_configuration_raises(chunk_size: int, overlap: int) -> None:
    with pytest.raises(ValueError):
        chunk_text("content", chunk_size=chunk_size, overlap=overlap)
