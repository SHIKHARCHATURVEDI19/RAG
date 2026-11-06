"""Command-line interface for ingestion and RAG queries."""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parent
sys.path.insert(0, str(ROOT / "src"))

from rag.config import Settings
from rag.runtime import RAGService


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="RAG From Scratch CLI")
    parser.add_argument("--version", action="version", version="RAG From Scratch 1.0.0")
    subparsers = parser.add_subparsers(dest="command", required=True)

    ingest_parser = subparsers.add_parser("ingest", help="Build an index from documents")
    ingest_parser.add_argument("path", type=Path, help="Supported file or directory")

    query_parser = subparsers.add_parser("query", help="Ask a question")
    query_parser.add_argument("question", help="Question to answer")
    query_parser.add_argument("--top-k", type=int, help="Final number of chunks")
    return parser


def _print_query_result(result) -> None:
    print("Answer")
    print("------")
    print(result.answer)
    print("\nSources")
    print("-------")
    if result.sources:
        for source in result.sources:
            location = source.file
            if source.page is not None:
                location += f" — page {source.page}"
            print(f"[{source.citation_id}] {location}")
    else:
        print("No sources retrieved.")
    print("\nRetrieval")
    print("---------")
    print(f"Top K: {result.metadata['top_k']}")
    print(f"Returned: {result.metadata['returned']}")
    print(f"Similarity threshold: {result.metadata['similarity_threshold']}")


def main(argv: list[str] | None = None) -> int:
    args = build_parser().parse_args(argv)
    service = RAGService(Settings.from_env(ROOT / ".env"))
    if args.command == "ingest":
        summary = service.ingest(args.path)
        print(
            f"Indexed {summary['documents']} document unit(s) into "
            f"{summary['chunks']} chunk(s), dimension {summary['dimension']}."
        )
        print(f"Saved index to {summary['index_directory']}.")
        return 0

    _print_query_result(service.query(args.question, top_k=args.top_k))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

