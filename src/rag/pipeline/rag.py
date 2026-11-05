"""End-to-end retrieval-augmented generation pipeline."""

from __future__ import annotations

from typing import Protocol

from rag.generation import ContextBuilder, Generator
from rag.schemas import RAGResult, RetrievedChunk, Source


class RetrievalBackend(Protocol):
    similarity_threshold: float

    def retrieve(self, query: str, top_k: int = 5) -> list[RetrievedChunk]: ...


class RAGPipeline:
    """Expose retrieval, context construction, generation, and citations as one API."""

    def __init__(
        self,
        retriever: RetrievalBackend,
        generator: Generator,
        context_builder: ContextBuilder | None = None,
        top_k: int = 5,
    ) -> None:
        if top_k <= 0:
            raise ValueError("top_k must be greater than zero")
        self.retriever = retriever
        self.generator = generator
        self.context_builder = context_builder or ContextBuilder()
        self.top_k = top_k

    @staticmethod
    def _sources(results: list[RetrievedChunk]) -> list[Source]:
        sources: list[Source] = []
        seen: set[tuple[str, int | None, str]] = set()
        for result in results:
            metadata = result.chunk.metadata
            source = str(metadata.get("source") or metadata.get("file_name") or "unknown")
            file_name = str(metadata.get("file_name") or source)
            page = metadata.get("page")
            key = (file_name, page, result.chunk.text.strip())
            if key in seen or not key[2]:
                continue
            seen.add(key)
            sources.append(
                Source(
                    citation_id=len(sources) + 1,
                    source=source,
                    file=file_name,
                    page=page if isinstance(page, int) else None,
                    chunk_index=metadata.get("chunk_index"),
                    score=result.score,
                )
            )
        return sources

    def query(self, question: str, top_k: int | None = None) -> RAGResult:
        """Answer one question and attach citations from actual retrieval results."""
        if not isinstance(question, str) or not question.strip():
            raise ValueError("question cannot be empty")
        requested_top_k = self.top_k if top_k is None else top_k
        if requested_top_k <= 0:
            raise ValueError("top_k must be greater than zero")

        results = self.retriever.retrieve(question.strip(), top_k=requested_top_k)
        context = self.context_builder.build(results)
        answer = self.generator.generate(question.strip(), context)
        return RAGResult(
            answer=answer,
            sources=self._sources(results),
            retrieved_chunks=results,
            metadata={
                "top_k": requested_top_k,
                "returned": len(results),
                "similarity_threshold": getattr(
                    self.retriever, "similarity_threshold", None
                ),
            },
        )

