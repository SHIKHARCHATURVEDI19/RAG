# RAG From Scratch

A retrieval-augmented generation (RAG) engine built from first principles in
Python. The initial implementation intentionally avoids LangChain and
LlamaIndex so that document loading, chunking, embeddings, similarity search,
retrieval, context construction, generation, and citations remain explicit.

## Project status

Phase 1 establishes the project foundation. The implementation will be added
in small, tested commits in this order:

1. Document loading and chunking
2. Embedding pipeline
3. Vector index and similarity search
4. Retrieval pipeline
5. Context building and generation
6. End-to-end RAG with citations
7. Reranking, hybrid search, and query rewriting
8. Evaluation, API, deployment, and CI

## Planned architecture

```text
Documents -> Loader -> Cleaning -> Chunking -> Embeddings -> Vector Index
          -> Retriever -> Context Builder -> LLM -> Answer + Sources
```

## Setup

Python 3.10 or newer is required.

```bash
python -m venv .venv
```

Activate the environment on Windows:

```powershell
.venv\Scripts\Activate.ps1
```

Install the dependencies:

```bash
python -m pip install --upgrade pip
python -m pip install -r requirements.txt
```

Create a local environment file if you need to configure an external model:

```powershell
Copy-Item .env.example .env
```

## Run

Check the command-line entry point:

```bash
python main.py --help
```

Start the API during development:

```bash
uvicorn api:app --reload
```

Then open `http://127.0.0.1:8000/health` to verify the service is running.

## Tests

```bash
pytest
```

## Design constraints

- No LangChain or LlamaIndex in the foundational implementation.
- Components communicate through small, replaceable interfaces.
- Answers will cite the chunks and source documents used to generate them.
- Evaluation results will be measured; benchmark values will never be invented.

