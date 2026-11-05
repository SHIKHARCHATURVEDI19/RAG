"""Load supported files into a common document representation."""

from __future__ import annotations

from pathlib import Path

from pypdf import PdfReader

from rag.schemas import Document

SUPPORTED_SUFFIXES = {".txt", ".md", ".pdf"}


def _metadata(path: Path) -> dict[str, str]:
    return {
        "source": str(path.resolve()),
        "file_name": path.name,
        "file_type": path.suffix.lower().lstrip("."),
    }


def load_document(path: str | Path) -> list[Document]:
    """Load a TXT, Markdown, or PDF file.

    Text and Markdown files produce one document. PDFs produce one document per
    page so page-level citations remain available throughout the pipeline.
    """
    file_path = Path(path).expanduser()
    if not file_path.exists():
        raise FileNotFoundError(f"Document does not exist: {file_path}")
    if not file_path.is_file():
        raise ValueError(f"Expected a file, received: {file_path}")

    suffix = file_path.suffix.lower()
    if suffix not in SUPPORTED_SUFFIXES:
        supported = ", ".join(sorted(SUPPORTED_SUFFIXES))
        raise ValueError(f"Unsupported document type '{suffix}'. Supported: {supported}")

    metadata = _metadata(file_path)
    if suffix in {".txt", ".md"}:
        return [
            Document(
                text=file_path.read_text(encoding="utf-8-sig"),
                metadata=metadata,
            )
        ]

    reader = PdfReader(str(file_path))
    return [
        Document(
            text=page.extract_text() or "",
            metadata={**metadata, "page": page_number},
        )
        for page_number, page in enumerate(reader.pages, start=1)
    ]


def load_documents(path: str | Path) -> list[Document]:
    """Load one supported file or every supported file below a directory."""
    source_path = Path(path).expanduser()
    if not source_path.exists():
        raise FileNotFoundError(f"Input path does not exist: {source_path}")
    if source_path.is_file():
        return load_document(source_path)

    documents: list[Document] = []
    for file_path in sorted(
        item
        for item in source_path.rglob("*")
        if item.is_file() and item.suffix.lower() in SUPPORTED_SUFFIXES
    ):
        documents.extend(load_document(file_path))
    return documents

