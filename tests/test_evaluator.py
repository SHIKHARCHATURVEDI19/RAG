import pytest

from rag.evaluation import EvaluationCase, Evaluator
from rag.schemas import DocumentChunk, RAGResult, RetrievedChunk, Source


class StubPipeline:
    def query(self, question: str, top_k: int | None = None) -> RAGResult:
        chunk = RetrievedChunk(
            DocumentChunk(
                "Customers may request a refund within 30 calendar days.",
                {"file_name": "refund-policy.md"},
            ),
            0.9,
        )
        return RAGResult(
            answer="Customers may request a refund within 30 calendar days. [1]",
            sources=[
                Source(1, "refund-policy.md", "refund-policy.md", score=0.9)
            ],
            retrieved_chunks=[chunk],
            metadata={"top_k": top_k},
        )


def test_evaluator_computes_expected_metrics() -> None:
    case = EvaluationCase(
        question="What is the refund period?",
        expected_answer="30 calendar days",
        expected_sources=["refund-policy.md"],
    )

    report = Evaluator(StubPipeline(), top_k=5).evaluate([case])

    assert report["questions"] == 1
    assert report["metrics"]["recall_at_k"] == 1.0
    assert report["metrics"]["precision_at_k"] == 1.0
    assert report["metrics"]["answer_correctness"] == 1.0
    assert report["metrics"]["citation_accuracy"] == 1.0
    assert report["metrics"]["faithfulness"] == 1.0
    assert report["metrics"]["latency_seconds"] >= 0


def test_evaluator_rejects_empty_dataset() -> None:
    with pytest.raises(ValueError, match="empty"):
        Evaluator(StubPipeline()).evaluate([])


def test_answer_correctness_allows_non_semantic_article_differences() -> None:
    class ArticlePipeline:
        def query(self, question: str, top_k: int | None = None) -> RAGResult:
            return RAGResult(
                answer="The order number and the reason.",
                sources=[],
                retrieved_chunks=[],
            )

    case = EvaluationCase(
        question="What is required?",
        expected_answer="order number and reason",
        expected_sources=[],
    )

    report = Evaluator(ArticlePipeline()).evaluate([case])

    assert report["metrics"]["answer_correctness"] == 1.0
