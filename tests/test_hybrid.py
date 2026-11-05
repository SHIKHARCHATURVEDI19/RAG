import numpy as np

from rag.reranking import Reranker
from rag.retrieval import (
    BM25Index,
    HybridRetriever,
    VectorIndex,
    reciprocal_rank_fusion,
)
from rag.schemas import DocumentChunk, RetrievedChunk


class FakeEmbedder:
    def embed_query(self, query: str) -> np.ndarray:
        return np.asarray([1.0, 0.0], dtype=np.float32)


def chunks() -> list[DocumentChunk]:
    return [
        DocumentChunk(
            "Customers receive a refund within thirty days.",
            {"file_name": "policy.md", "chunk_index": 0},
        ),
        DocumentChunk(
            "Error code ZX-491 requires a restart.",
            {"file_name": "manual.md", "chunk_index": 0},
        ),
        DocumentChunk(
            "Shipping usually takes one week.",
            {"file_name": "shipping.md", "chunk_index": 0},
        ),
    ]


def test_bm25_finds_exact_keyword_match() -> None:
    results = BM25Index(chunks()).search("ZX-491", top_k=2)

    assert results[0].chunk.metadata["file_name"] == "manual.md"
    assert results[0].score > 0


def test_reciprocal_rank_fusion_rewards_results_from_both_lists() -> None:
    first, second, third = chunks()
    fused = reciprocal_rank_fusion(
        [
            [RetrievedChunk(first, 0.9), RetrievedChunk(second, 0.8)],
            [RetrievedChunk(second, 5.0), RetrievedChunk(third, 4.0)],
        ],
        top_k=3,
    )

    assert fused[0].chunk is second


def test_reranker_promotes_query_term_overlap() -> None:
    candidates = [
        RetrievedChunk(chunks()[2], 0.9),
        RetrievedChunk(chunks()[0], 0.1),
    ]

    results = Reranker().rerank("refund days", candidates, top_k=2)

    assert results[0].chunk.metadata["file_name"] == "policy.md"


def test_cross_encoder_adapter_uses_model_predictions() -> None:
    class FakeCrossEncoder:
        def predict(self, pairs):
            assert len(pairs) == 2
            return [0.1, 0.9]

    candidates = [
        RetrievedChunk(chunks()[0], 1.0),
        RetrievedChunk(chunks()[1], 0.0),
    ]

    results = Reranker(model=FakeCrossEncoder()).rerank("query", candidates, top_k=1)

    assert results[0].chunk.metadata["file_name"] == "manual.md"


def test_hybrid_retriever_combines_and_limits_results() -> None:
    index = VectorIndex()
    index.add(
        chunks(),
        np.asarray([[1.0, 0.0], [0.2, 0.8], [0.8, 0.2]], dtype=np.float32),
    )
    retriever = HybridRetriever(FakeEmbedder(), index, candidate_k=3)

    results = retriever.retrieve("ZX-491 restart", top_k=2)

    assert len(results) == 2
    assert results[0].chunk.metadata["file_name"] == "manual.md"

