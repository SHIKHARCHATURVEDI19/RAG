"""Convenience wrapper for building the persisted document index."""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from rag.config import Settings
from rag.runtime import RAGService


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("path", type=Path, nargs="?", default=ROOT / "data" / "raw")
    args = parser.parse_args()
    summary = RAGService(Settings.from_env(ROOT / ".env")).ingest(args.path)
    print(
        f"Indexed {summary['documents']} document unit(s) into "
        f"{summary['chunks']} chunk(s)."
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

