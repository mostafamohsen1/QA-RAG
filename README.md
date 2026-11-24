# Document QA RAG

A production-style retrieval-augmented generation service: upload PDF/DOCX/TXT/MD documents, ask questions, get answers grounded in your documents with page-level citations. Runs fully on CPU.

## Architecture

```
upload → parse (PyMuPDF / python-docx) → recursive chunking (1000 chars, 150 overlap)
       → fastembed bge-small-en-v1.5 (384-d, ONNX on CPU) → Postgres + pgvector (HNSW)
query → hybrid retrieval (pgvector HNSW dense + rank_bm25 lexical)
      → cross-encoder reranker (ms-marco-MiniLM-L-6-v2, top-20 → top-5)
      → LLM with strict citation prompt ([n] cites) → abstains when context lacks the answer
```

CPU optimization: ONNX Runtime inference (no PyTorch), batched embedding, singleton models, HNSW ANN index (`m=16, ef_construction=64`) plus a GIN `tsvector` index for the full-text baseline.

## Setup

```bash
docker compose up -d          # Postgres 16 + pgvector on :5432
cp .env.example .env          # edit LLM settings
pip install -r requirements.txt
uvicorn app.main:app --reload # http://localhost:8000  (UI at /, API docs at /docs)
```

Point the LLM at any OpenAI-compatible endpoint (OpenAI, Groq, OpenRouter, vLLM, LM Studio) or set `LLM_PROVIDER=ollama` for local Ollama. The app itself, embeddings and reranking run locally on CPU.

## API

| Method | Path | Purpose |
|---|---|---|
| GET | `/api/health` | model config |
| POST | `/api/documents` | upload file (multipart `file`) |
| GET | `/api/documents` | list indexed documents |
| DELETE | `/api/documents/{doc_id}` | delete doc + chunks |
| POST | `/api/query` | `{question, top_k?, top_n?, doc_ids?}` → answer + citations |

Error handling: 415 unknown type, 400 empty file, 413 >50 MB, 422 corrupt/unparseable or empty-query, 503 LLM timeout/unreachable, 404 unknown doc. Re-uploading identical content is deduplicated by SHA-256.

## Evaluation

```bash
python -m eval.run_eval --k 5            # retrieval + answer metrics
python -m eval.run_eval --faithful       # + LLM-as-judge faithfulness
```

Metrics over `eval/eval_set.json` (7 grounded + 3 must-abstain questions): Hit@k, MRR, context precision/recall, answer correctness (token overlap), abstain accuracy, faithfulness.

## Tests

```bash
pytest tests -v        # needs Postgres running; ingestion, retrieval, API suites
```
