"""FastAPI entry point for the RAG service."""

from fastapi import FastAPI

app = FastAPI(
    title="RAG From Scratch",
    description="A retrieval-augmented generation API built from first principles.",
    version="0.1.0",
)


@app.get("/health", tags=["system"])
def health() -> dict[str, str]:
    """Report whether the API process is healthy."""
    return {"status": "ok"}

