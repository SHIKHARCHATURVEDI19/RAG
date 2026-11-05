from rag.generation import Generator
from rag.pipeline import RAGPipeline
from rag.schemas import DocumentChunk, RetrievedChunk


class StubRetriever:
    similarity_threshold = 0.5

    def __init__(self, results: list[RetrievedChunk]) -> None:
        self.results = results
        self.calls: list[tuple[str, int]] = []

    def retrieve(self, query: str, top_k: int = 5) -> list[RetrievedChunk]:
        self.calls.append((query, top_k))
        return self.results[:top_k]


def test_rag_query_returns_answer_and_real_citations() -> None:
    retrieved = [
        RetrievedChunk(
            DocumentChunk(
                "Refunds are available for 30 days.",
                {
                    "source": "/documents/refund-policy.pdf",
                    "file_name": "refund-policy.pdf",
                    "page": 4,
                    "chunk_index": 2,
                },
            ),
            0.91,
        )
    ]
    retriever = StubRetriever(retrieved)
    generator = Generator(completion_fn=lambda _: "Refunds are available for 30 days. [1]")

    result = RAGPipeline(retriever, generator, top_k=5).query(
        "What is the refund policy?"
    )

    assert result.answer.endswith("[1]")
    assert result.sources[0].file == "refund-policy.pdf"
    assert result.sources[0].page == 4
    assert result.sources[0].source == "/documents/refund-policy.pdf"
    assert result.metadata == {
        "top_k": 5,
        "returned": 1,
        "similarity_threshold": 0.5,
    }
    assert retriever.calls == [("What is the refund policy?", 5)]


def test_citations_cannot_be_invented_by_generator() -> None:
    retrieved = [
        RetrievedChunk(DocumentChunk("Fact", {"file_name": "real.txt"}), 0.8)
    ]
    generator = Generator(
        completion_fn=lambda _: "Claim [99] from imaginary.pdf"
    )

    result = RAGPipeline(StubRetriever(retrieved), generator).query("Question?")

    assert [source.file for source in result.sources] == ["real.txt"]


def test_no_results_returns_no_sources() -> None:
    result = RAGPipeline(StubRetriever([]), Generator()).query("Unknown question?")

    assert result.sources == []
    assert result.retrieved_chunks == []
    assert "not have enough information" in result.answer


def test_result_serialization_can_include_chunks() -> None:
    retrieved = [RetrievedChunk(DocumentChunk("Fact", {"file_name": "one.md"}), 0.7)]
    result = RAGPipeline(StubRetriever(retrieved), Generator()).query("Question?")

    payload = result.to_dict(include_chunks=True)

    assert payload["sources"][0]["file"] == "one.md"
    assert payload["retrieved_chunks"][0]["text"] == "Fact"

