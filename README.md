# RAG From Scratch

A complete retrieval-augmented generation engine built in Python without
LangChain or LlamaIndex. The project makes document ingestion, chunking,
embeddings, vector similarity, BM25, fusion, reranking, context construction,
generation, citations, and evaluation visible and replaceable.

## What is RAG?

Retrieval-augmented generation finds relevant passages before asking a language
model to answer. The retrieved text gives the model current, source-specific
evidence and enables citations without retraining the model.

## Features

- TXT, Markdown, and page-aware PDF loading
- Overlapping character chunking with preserved metadata
- Batched, normalized Sentence Transformers embeddings
- Exact cosine similarity implemented with NumPy
- Persisted embeddings, chunk text, metadata, and index configuration
- Semantic retrieval with thresholds and deduplication
- BM25 lexical search and reciprocal-rank fusion
- Pluggable reranker with a deterministic offline fallback
- Source-aware prompts and citations owned by the retrieval pipeline
- OpenAI-compatible generation with an offline grounded fallback
- Recall, precision, correctness, citation, faithfulness, and latency evaluation
- CLI, FastAPI, Docker Compose, tests, and GitHub Actions CI

## Architecture

```text
                   User query
                       |
            +----------+----------+
            |                     |
      Vector search              BM25
            |                     |
            +----------+----------+
                       |
             Reciprocal-rank fusion
                       |
                    Reranker
                       |
                  Top passages
                       |
                Context builder
                       |
               Generation backend
                       |
              Answer + citations
```

## Data flow

```text
Documents -> Loader -> Chunker -> Embedder -> Vector index -> Persistence
Query -> Hybrid retriever -> Reranker -> Context -> Generator -> Result
```

Every PDF page becomes a separate `Document`. Every `DocumentChunk` retains its
source path, file name, file type, page when applicable, and chunk index.

## Supported documents

| Format | Extension | Citation granularity |
|---|---|---|
| Plain text | `.txt` | File |
| Markdown | `.md` | File |
| PDF | `.pdf` | Page |

## Chunking strategy

The default window is 500 characters with a 100-character overlap. Overlap
reduces information loss at boundaries. Both values are configurable through
`RAG_CHUNK_SIZE` and `RAG_CHUNK_OVERLAP`; the best values depend on the document
style and must be evaluated rather than guessed.

## Embedding model

The default model is `sentence-transformers/all-MiniLM-L6-v2`. Encoding is
batched and vectors are normalized once, allowing cosine similarity to become a
fast matrix-vector dot product. The `Embedder` interface keeps model-specific
code out of the retrieval pipeline.

## Retrieval

`VectorIndex` performs exact NumPy cosine search. This is intentionally simple
and educational; it is appropriate for small collections. The index is saved as
`embeddings.npy`, `chunks.json`, and `index.json` under `data/processed/`.

## Hybrid search

Semantic search is combined with an in-house Okapi BM25 implementation. Because
their raw scores are not comparable, reciprocal-rank fusion combines their
rankings. This supports both paraphrased questions and exact identifiers such as
error codes.

## Reranking

Hybrid retrieval first gathers up to 20 candidates, then reranks the candidates
and returns the final top five. `Reranker` accepts a cross-encoder-compatible
model. Without one, it uses a reproducible lexical relevance fallback.

## Generation

`RAG_GENERATION_PROVIDER=auto` calls an OpenAI-compatible provider when
`OPENAI_API_KEY` is set and otherwise uses the local fallback. Set it to
`openai_compatible` to require the remote provider or `extractive` to force
offline operation. The remote backend calls an
`/chat/completions` endpoint. The system prompt requires answers to use only the
provided context, decline unsupported questions, avoid invented facts, and cite
the bracketed source numbers. Without a key, a deterministic extractive fallback
keeps local usage and evaluation available offline.

## Citations

The model produces answer text, but it never controls source metadata. Citation
records are constructed directly from retrieved chunks and include the source,
file, page, chunk index, and score.

## Installation

Python 3.10 or newer is required.

```powershell
python -m venv .venv
.venv\Scripts\Activate.ps1
python -m pip install --upgrade pip
python -m pip install -r requirements.txt
Copy-Item .env.example .env
```

On macOS or Linux, activate with `source .venv/bin/activate` and copy the
environment file with `cp .env.example .env`.

## CLI usage

Build the persisted index:

```bash
python main.py ingest data/raw
```

Ask a question:

```bash
python main.py query "How long do customers have to request a refund?"
```

The command prints the grounded answer, source list, requested top-k, returned
result count, and similarity threshold.

## API

Start the development server:

```bash
uvicorn api:app --reload
```

Available endpoints:

- `GET /health`
- `POST /ingest` with `{"path": "data/raw"}`
- `POST /query` with `{"query": "What is the refund policy?", "top_k": 5}`

Interactive OpenAPI documentation is available at `http://127.0.0.1:8000/docs`.

## Evaluation

After ingestion, run:

```bash
python scripts/evaluate.py
```

The report contains Recall@K, Precision@K, answer correctness, citation
accuracy, faithfulness, latency, and per-question details. The correctness and
faithfulness metrics are deliberately transparent lexical heuristics, not
LLM-as-judge claims.

## Benchmark results

No benchmark numbers are hard-coded here. Run the evaluation command against
your selected generation backend and dataset to produce results for that exact
configuration.

## Testing

```bash
pytest -q
python -m compileall -q src tests scripts main.py api.py
```

Tests use injected embedding and generation doubles, so they do not download
models or call external APIs.

## Docker

```bash
docker compose up --build
```

The API is exposed on port 8000 and the local `data/` directory is mounted into
the container so the persisted index survives container replacement.

## Configuration

| Variable | Default | Purpose |
|---|---|---|
| `EMBEDDING_MODEL` | `sentence-transformers/all-MiniLM-L6-v2` | Local embedding model |
| `OPENAI_API_KEY` | empty | Enables remote LLM generation |
| `OPENAI_MODEL` | `gpt-4o-mini` | Generation model |
| `OPENAI_BASE_URL` | `https://api.openai.com/v1` | Compatible API root |
| `RAG_GENERATION_PROVIDER` | `auto` | `auto`, `openai_compatible`, or `extractive` |
| `RAG_CHUNK_SIZE` | `500` | Chunk character count |
| `RAG_CHUNK_OVERLAP` | `100` | Repeated boundary characters |
| `RAG_TOP_K` | `5` | Final retrieved passages |
| `RAG_CANDIDATE_K` | `20` | Candidates before reranking |
| `RAG_SIMILARITY_THRESHOLD` | `0.50` | Minimum semantic similarity |
| `RAG_INDEX_DIRECTORY` | `data/processed` | Persisted index location |

## Limitations

- Exact vector search scans every chunk and is not intended for millions of
  vectors.
- The offline generator is extractive, not a substitute for a language model.
- The default reranker is lexical unless a cross-encoder is injected.
- Character chunking does not understand semantic or sentence boundaries.
- Evaluation heuristics should be supplemented with expert labels for serious
  deployments.
- The API uses one in-process index and has no authentication or job queue.

## Future improvements

- Approximate nearest-neighbor indexing for large collections
- Token-aware or semantic chunking and context compression
- Production cross-encoder configuration and model caching
- Query rewriting with measured retrieval impact
- Incremental ingestion, document deletion, and index versioning
- Authentication, rate limits, observability, and background ingestion jobs
- Larger domain-specific regression datasets and human evaluation
