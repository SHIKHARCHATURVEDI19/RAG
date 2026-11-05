"""Run the evaluation dataset against a persisted RAG index."""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from rag.embeddings import Embedder
from rag.evaluation import Evaluator
from rag.generation import Generator
from rag.pipeline import RAGPipeline
from rag.retrieval import HybridRetriever, VectorIndex


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--dataset",
        type=Path,
        default=ROOT / "data" / "evaluation" / "dataset.json",
    )
    parser.add_argument(
        "--index",
        type=Path,
        default=ROOT / "data" / "processed",
    )
    parser.add_argument("--top-k", type=int, default=5)
    parser.add_argument("--output", type=Path)
    return parser


def main() -> int:
    args = build_parser().parse_args()
    if not (args.index / "index.json").exists():
        raise SystemExit(
            f"No index found in {args.index}. Run `python main.py ingest data/raw` first."
        )

    index = VectorIndex.load(args.index)
    retriever = HybridRetriever(Embedder(), index)
    pipeline = RAGPipeline(retriever, Generator(), top_k=args.top_k)
    cases = Evaluator.load_dataset(args.dataset)
    report = Evaluator(pipeline, top_k=args.top_k).evaluate(cases)

    output = json.dumps(report, indent=2)
    print(output)
    if args.output:
        args.output.parent.mkdir(parents=True, exist_ok=True)
        args.output.write_text(output + "\n", encoding="utf-8")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

