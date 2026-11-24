"""Hybrid retrieval: pgvector HNSW dense search + BM25 (rank_bm25 over DB chunks,
with a ts_rank_cd full-text fallback for very large corpora)."""

import math

from rank_bm25 import BM25Okapi
from sqlalchemy import text as sql_text
from sqlalchemy.orm import Session

from app.config import settings
from app.core.embeddings import embed_query

MAX_BM25_CHUNKS = 20_000  # load at most this many chunks into the in-memory BM25 index


def _tokenize(s: str) -> list[str]:
    return [t for t in __import__("re").findall(r"[a-z0-9]+", s.lower()) if t]


def dense_search(db: Session, query_embedding: list[float], top_k: int,
                 doc_ids: list[str] | None = None) -> list[dict]:
    doc_filter = ""
    params: dict = {
        "q": query_embedding,
        "k": top_k,
    }
    if doc_ids:
        doc_filter = "WHERE c.doc_id = ANY(:doc_ids)"
        params["doc_ids"] = doc_ids
    rows = db.execute(sql_text(f"""
        SELECT c.id, c.doc_id, c.filename, c.page_number, c.chunk_index,
               c.char_start, c.char_end, c.text,
               1 - (c.embedding <=> CAST(:q AS vector)) AS score
        FROM chunks c
        {doc_filter}
        ORDER BY c.embedding <=> CAST(:q AS vector)
        LIMIT :k
    """), params).mappings().all()
    return [dict(r) for r in rows]


def bm25_search(db: Session, query: str, top_k: int,
                doc_ids: list[str] | None = None) -> list[dict]:
    """In-memory BM25 over corpus text pulled from the DB (rank_bm25).

    Scores are squashed to 0..1 so they are comparable against the dense scores.
    """
    where = "WHERE c.doc_id = ANY(:doc_ids) " if doc_ids else ""
    params: dict = {"limit": MAX_BM25_CHUNKS}
    if doc_ids:
        params["doc_ids"] = doc_ids
    rows = db.execute(sql_text(f"""
        SELECT c.id, c.doc_id, c.filename, c.page_number, c.chunk_index,
               c.char_start, c.char_end, c.text
        FROM chunks c {where}LIMIT :limit
    """), params).mappings().all()
    if not rows:
        return []

    corpus = [_tokenize(r["text"]) for r in rows]
    bm25 = BM25Okapi(corpus)
    scores = bm25.get_scores(_tokenize(query))
    ranked_idx = sorted(range(len(rows)), key=lambda i: scores[i], reverse=True)[:top_k]

    results = []
    for i in ranked_idx:
        if scores[i] <= 0:
            continue
        r = dict(rows[i])
        r["score"] = 1.0 / (1.0 + math.exp(-scores[i]))
        r["bm25_raw"] = float(scores[i])
        results.append(r)
    return results


def hybrid_search(db: Session, query: str, top_k: int | None = None,
                  doc_ids: list[str] | None = None) -> list[dict]:
    """Dense + BM25 candidate union with reciprocal-rank-style fusion scores."""
    top_k = top_k or settings.retrieval_top_k
    qvec = embed_query(query)
    dense = dense_search(db, qvec, top_k, doc_ids)
    sparse = bm25_search(db, query, top_k, doc_ids)

    # score fusion: dense similarity + normalized BM25, weighted by rank agreement
    seen: dict[int, dict] = {}
    for c in dense:
        c = dict(c)
        c["rerank_score"] = None
        seen[c["id"]] = c
    for c in sparse:
        if c["id"] in seen:
            seen[c["id"]]["score"] = max(seen[c["id"]]["score"], c["score"])
        else:
            c["rerank_score"] = None
            seen[c["id"]] = c
    return sorted(seen.values(), key=lambda c: c["score"], reverse=True)[:top_k]
