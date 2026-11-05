"""Lightweight, reproducible evaluation for retrieval and grounded answers."""

from __future__ import annotations

import json
import re
import time
from dataclasses import dataclass
from pathlib import Path
from statistics import mean
from typing import Any, Protocol

from rag.schemas import RAGResult


def _tokens(text: str) -> set[str]:
    return set(re.findall(r"\w+", text.casefold(), flags=re.UNICODE))


def _normalized(text: str) -> str:
    return " ".join(re.findall(r"\w+", text.casefold(), flags=re.UNICODE))


@dataclass(slots=True)
class EvaluationCase:
    question: str
    expected_answer: str
    expected_sources: list[str]

    @classmethod
    def from_dict(cls, value: dict[str, Any]) -> "EvaluationCase":
        return cls(
            question=str(value["question"]),
            expected_answer=str(value["expected_answer"]),
            expected_sources=[str(source) for source in value["expected_sources"]],
        )


class QueryPipeline(Protocol):
    def query(self, question: str, top_k: int | None = None) -> RAGResult: ...


class Evaluator:
    """Evaluate retrieval, answer, citation, faithfulness, and latency."""

    def __init__(self, pipeline: QueryPipeline, top_k: int = 5) -> None:
        if top_k <= 0:
            raise ValueError("top_k must be greater than zero")
        self.pipeline = pipeline
        self.top_k = top_k

    @staticmethod
    def load_dataset(path: str | Path) -> list[EvaluationCase]:
        payload = json.loads(Path(path).read_text(encoding="utf-8"))
        if not isinstance(payload, list):
            raise ValueError("Evaluation dataset must be a JSON list")
        return [EvaluationCase.from_dict(item) for item in payload]

    @staticmethod
    def _faithfulness(result: RAGResult) -> float:
        answer_tokens = _tokens(re.sub(r"\[\d+\]", "", result.answer))
        if not answer_tokens:
            return 1.0
        context_tokens = _tokens(
            " ".join(item.chunk.text for item in result.retrieved_chunks)
        )
        return len(answer_tokens & context_tokens) / len(answer_tokens)

    def evaluate_case(self, case: EvaluationCase) -> dict[str, Any]:
        started = time.perf_counter()
        result = self.pipeline.query(case.question, top_k=self.top_k)
        latency = time.perf_counter() - started

        expected_sources = set(case.expected_sources)
        retrieved_sources = {source.file for source in result.sources}
        relevant_sources = expected_sources & retrieved_sources
        recall = (
            len(relevant_sources) / len(expected_sources) if expected_sources else 1.0
        )
        precision = (
            len(relevant_sources) / len(retrieved_sources)
            if retrieved_sources
            else float(not expected_sources)
        )
        answer_correctness = float(
            _normalized(case.expected_answer) in _normalized(result.answer)
        )

        return {
            "question": case.question,
            "recall_at_k": recall,
            "precision_at_k": precision,
            "answer_correctness": answer_correctness,
            "citation_accuracy": precision,
            "faithfulness": self._faithfulness(result),
            "latency_seconds": latency,
            "answer": result.answer,
            "retrieved_sources": sorted(retrieved_sources),
        }

    def evaluate(self, cases: list[EvaluationCase]) -> dict[str, Any]:
        if not cases:
            raise ValueError("Evaluation dataset cannot be empty")
        results = [self.evaluate_case(case) for case in cases]
        metric_names = [
            "recall_at_k",
            "precision_at_k",
            "answer_correctness",
            "citation_accuracy",
            "faithfulness",
            "latency_seconds",
        ]
        summary = {
            metric: mean(float(result[metric]) for result in results)
            for metric in metric_names
        }
        return {
            "questions": len(results),
            "top_k": self.top_k,
            "metrics": summary,
            "results": results,
        }

