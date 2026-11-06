"""Reusable application service shared by the CLI and API."""

from __future__ import annotations

import logging
from pathlib import Path
from typing import Any

from rag.config import Settings
from rag.embeddings import Embedder
from rag.generation import Generator
from rag.ingestion import chunk_documents, load_documents
from rag.pipeline import RAGPipeline
from rag.retrieval import HybridRetriever, VectorIndex
from rag.schemas import RAGResult

logger = logging.getLogger(__name__)


class RAGService:
    """Own ingestion, persistence, and querying for one RAG index."""

    def __init__(
        self,
        settings: Settings | None = None,
        embedder: Any | None = None,
        generator: Generator | None = None,
    ) -> None:
        self.settings = settings or Settings.from_env()
        self.embedder = embedder or Embedder(
            model_name=self.settings.embedding_model
        )
        self.generator = generator or Generator(
            provider=self.settings.generation_provider
        )
        self._index: VectorIndex | None = None

    @property
    def index(self) -> VectorIndex:
        if self._index is None:
            index_file = self.settings.index_directory / "index.json"
            if not index_file.exists():
                raise FileNotFoundError(
                    "No persisted index found. Run `python main.py ingest data/raw` first."
                )
            self._index = VectorIndex.load(self.settings.index_directory)
            logger.info(
                "Loaded persisted index with %s chunks from %s",
                self._index.size,
                self.settings.index_directory,
            )
        return self._index

    def ingest(self, path: str | Path) -> dict[str, Any]:
        documents = load_documents(path)
        chunks = chunk_documents(
            documents,
            chunk_size=self.settings.chunk_size,
            overlap=self.settings.chunk_overlap,
        )
        if not chunks:
            raise ValueError("No text chunks were produced from the input documents")

        embeddings = self.embedder.embed_documents([chunk.text for chunk in chunks])
        index = VectorIndex()
        index.add(chunks, embeddings)
        index.save(self.settings.index_directory)
        self._index = index
        logger.info(
            "Indexed %s document units into %s chunks at %s",
            len(documents),
            len(chunks),
            self.settings.index_directory,
        )
        return {
            "documents": len(documents),
            "chunks": len(chunks),
            "dimension": index.dimension,
            "index_directory": str(self.settings.index_directory),
        }

    def pipeline(self, top_k: int | None = None) -> RAGPipeline:
        retriever = HybridRetriever(
            self.embedder,
            self.index,
            similarity_threshold=self.settings.similarity_threshold,
            candidate_k=self.settings.candidate_k,
        )
        return RAGPipeline(
            retriever,
            self.generator,
            top_k=top_k or self.settings.top_k,
        )

    def query(self, question: str, top_k: int | None = None) -> RAGResult:
        return self.pipeline(top_k=top_k).query(question, top_k=top_k)
