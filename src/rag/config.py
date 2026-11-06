"""Environment-backed application configuration."""

from __future__ import annotations

import os
from dataclasses import dataclass
from pathlib import Path

from dotenv import load_dotenv


@dataclass(frozen=True, slots=True)
class Settings:
    embedding_model: str = "sentence-transformers/all-MiniLM-L6-v2"
    generation_provider: str = "auto"
    chunk_size: int = 500
    chunk_overlap: int = 100
    top_k: int = 5
    similarity_threshold: float = 0.5
    candidate_k: int = 20
    index_directory: Path = Path("data/processed")

    def __post_init__(self) -> None:
        if self.chunk_size <= 0:
            raise ValueError("chunk_size must be greater than zero")
        if not 0 <= self.chunk_overlap < self.chunk_size:
            raise ValueError("chunk_overlap must be between 0 and chunk_size")
        if self.top_k <= 0 or self.candidate_k <= 0:
            raise ValueError("top_k and candidate_k must be greater than zero")
        if not -1 <= self.similarity_threshold <= 1:
            raise ValueError("similarity_threshold must be between -1 and 1")
        if self.generation_provider not in {"auto", "extractive", "openai_compatible"}:
            raise ValueError(
                "generation_provider must be auto, extractive, or openai_compatible"
            )

    @classmethod
    def from_env(cls, env_file: str | Path | None = None) -> "Settings":
        load_dotenv(dotenv_path=env_file)
        return cls(
            embedding_model=os.getenv(
                "EMBEDDING_MODEL", "sentence-transformers/all-MiniLM-L6-v2"
            ),
            generation_provider=os.getenv("RAG_GENERATION_PROVIDER", "auto"),
            chunk_size=int(os.getenv("RAG_CHUNK_SIZE", "500")),
            chunk_overlap=int(os.getenv("RAG_CHUNK_OVERLAP", "100")),
            top_k=int(os.getenv("RAG_TOP_K", "5")),
            similarity_threshold=float(
                os.getenv("RAG_SIMILARITY_THRESHOLD", "0.5")
            ),
            candidate_k=int(os.getenv("RAG_CANDIDATE_K", "20")),
            index_directory=Path(
                os.getenv("RAG_INDEX_DIRECTORY", "data/processed")
            ),
        )
