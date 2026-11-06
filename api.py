"""FastAPI interface for the RAG service."""

from __future__ import annotations

import sys
import logging
from pathlib import Path
from typing import Any

from fastapi import FastAPI, HTTPException
from pydantic import BaseModel, Field

ROOT = Path(__file__).resolve().parent
sys.path.insert(0, str(ROOT / "src"))

from rag.config import Settings
from rag.runtime import RAGService

logger = logging.getLogger(__name__)

app = FastAPI(
    title="RAG From Scratch",
    description="Framework-free ingestion, hybrid retrieval, and grounded generation.",
    version="1.0.0",
)
service = RAGService(Settings.from_env(ROOT / ".env"))


class IngestRequest(BaseModel):
    path: str = Field(default="data/raw", min_length=1)


class QueryRequest(BaseModel):
    query: str = Field(min_length=1)
    top_k: int | None = Field(default=None, ge=1)


class IngestResponse(BaseModel):
    documents: int
    chunks: int
    dimension: int
    index_directory: str


class SourceResponse(BaseModel):
    citation_id: int
    source: str
    file: str
    page: int | None = None
    chunk_index: int | None = None
    score: float | None = None


class QueryResponse(BaseModel):
    answer: str
    sources: list[SourceResponse]
    metadata: dict[str, Any]


@app.get("/health", tags=["system"])
def health() -> dict[str, str]:
    return {"status": "ok"}


@app.post("/ingest", response_model=IngestResponse, tags=["rag"])
def ingest(request: IngestRequest) -> IngestResponse:
    try:
        response = IngestResponse(**service.ingest(request.path))
        logger.info("Ingested %s into %s chunks", request.path, response.chunks)
        return response
    except (FileNotFoundError, ValueError) as error:
        raise HTTPException(status_code=400, detail=str(error)) from error


@app.post("/query", response_model=QueryResponse, tags=["rag"])
def query(request: QueryRequest) -> QueryResponse:
    try:
        result = service.query(request.query, top_k=request.top_k)
    except FileNotFoundError as error:
        raise HTTPException(status_code=409, detail=str(error)) from error
    except ValueError as error:
        raise HTTPException(status_code=400, detail=str(error)) from error
    response = QueryResponse(**result.to_dict())
    logger.info("Answered query with %s source(s)", len(response.sources))
    return response
