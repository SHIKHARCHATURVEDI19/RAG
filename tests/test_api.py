from fastapi.testclient import TestClient

import api
from rag.schemas import RAGResult, Source


class StubService:
    def ingest(self, path: str):
        return {"documents": 1, "chunks": 2, "dimension": 3, "index_directory": path}

    def query(self, question: str, top_k: int | None = None):
        return RAGResult(
            answer="Grounded answer [1]",
            sources=[Source(1, "policy.md", "policy.md")],
            retrieved_chunks=[],
            metadata={"top_k": top_k or 5, "returned": 1},
        )


def test_health_endpoint() -> None:
    response = TestClient(api.app).get("/health")

    assert response.status_code == 200
    assert response.json() == {"status": "ok"}


def test_ingest_and_query_endpoints(monkeypatch) -> None:
    monkeypatch.setattr(api, "service", StubService())
    client = TestClient(api.app)

    ingest_response = client.post("/ingest", json={"path": "data/raw"})
    query_response = client.post(
        "/query", json={"query": "What is the policy?", "top_k": 3}
    )

    assert ingest_response.status_code == 200
    assert ingest_response.json()["chunks"] == 2
    assert query_response.status_code == 200
    assert query_response.json()["answer"] == "Grounded answer [1]"
    assert query_response.json()["sources"][0]["file"] == "policy.md"

