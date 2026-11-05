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

