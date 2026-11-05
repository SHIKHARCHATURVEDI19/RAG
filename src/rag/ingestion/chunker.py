"""Character-based text chunking with overlap."""

from __future__ import annotations

from collections.abc import Iterable

from rag.schemas import Document, DocumentChunk


def _validate_configuration(chunk_size: int, overlap: int) -> None:
    if chunk_size <= 0:
        raise ValueError("chunk_size must be greater than zero")
    if overlap < 0:
        raise ValueError("overlap cannot be negative")
    if overlap >= chunk_size:
        raise ValueError("overlap must be smaller than chunk_size")


def chunk_text(text: str, chunk_size: int = 500, overlap: int = 100) -> list[str]:
    """Split text into fixed-size character windows with an overlap."""
    _validate_configuration(chunk_size, overlap)
    if not text:
        return []

    step = chunk_size - overlap
    chunks: list[str] = []
    start = 0
    while start < len(text):
        chunks.append(text[start : start + chunk_size])
        if start + chunk_size >= len(text):
            break
        start += step
    return chunks


def chunk_document(
    document: Document,
    chunk_size: int = 500,
    overlap: int = 100,
) -> list[DocumentChunk]:
    """Chunk one document while preserving all source metadata."""
    return [
        DocumentChunk(
            text=text,
            metadata={**document.metadata, "chunk_index": chunk_index},
        )
        for chunk_index, text in enumerate(
            chunk_text(document.text, chunk_size=chunk_size, overlap=overlap)
        )
    ]


def chunk_documents(
    documents: Iterable[Document],
    chunk_size: int = 500,
    overlap: int = 100,
) -> list[DocumentChunk]:
    """Chunk multiple documents into one ordered list."""
    chunks: list[DocumentChunk] = []
    for document in documents:
        chunks.extend(
            chunk_document(document, chunk_size=chunk_size, overlap=overlap)
        )
    return chunks
