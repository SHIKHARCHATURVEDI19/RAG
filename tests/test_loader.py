from pathlib import Path

import pytest

from rag.ingestion import loader


def test_load_txt(tmp_path: Path) -> None:
    path = tmp_path / "notes.txt"
    path.write_text("plain text", encoding="utf-8")

    documents = loader.load_document(path)

    assert len(documents) == 1
    assert documents[0].text == "plain text"
    assert documents[0].metadata["file_name"] == "notes.txt"
    assert documents[0].metadata["file_type"] == "txt"


def test_load_markdown(tmp_path: Path) -> None:
    path = tmp_path / "guide.md"
    path.write_text("# Guide", encoding="utf-8")

    document = loader.load_document(path)[0]

    assert document.text == "# Guide"
    assert document.metadata["file_type"] == "md"


def test_load_pdf_creates_one_document_per_page(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    path = tmp_path / "manual.pdf"
    path.write_bytes(b"fake pdf for mocked reader")

    class FakePage:
        def __init__(self, text: str | None) -> None:
            self.text = text

        def extract_text(self) -> str | None:
            return self.text

    class FakeReader:
        def __init__(self, _: str) -> None:
            self.pages = [FakePage("first"), FakePage(None)]

    monkeypatch.setattr(loader, "PdfReader", FakeReader)

    documents = loader.load_document(path)

    assert [document.text for document in documents] == ["first", ""]
    assert [document.metadata["page"] for document in documents] == [1, 2]
    assert all(document.metadata["file_name"] == "manual.pdf" for document in documents)


def test_missing_file_raises(tmp_path: Path) -> None:
    with pytest.raises(FileNotFoundError):
        loader.load_document(tmp_path / "missing.txt")


def test_unsupported_file_raises(tmp_path: Path) -> None:
    path = tmp_path / "image.png"
    path.write_bytes(b"png")

    with pytest.raises(ValueError, match="Unsupported"):
        loader.load_document(path)


def test_load_directory_ignores_unsupported_files(tmp_path: Path) -> None:
    (tmp_path / "one.txt").write_text("one", encoding="utf-8")
    (tmp_path / "two.md").write_text("two", encoding="utf-8")
    (tmp_path / "ignored.csv").write_text("three", encoding="utf-8")

    documents = loader.load_documents(tmp_path)

    assert [document.text for document in documents] == ["one", "two"]

