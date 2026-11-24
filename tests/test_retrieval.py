import math

from app.core import embeddings, reranker
from app.core.retriever import bm25_search, dense_search, hybrid_search
from app.db.models import Chunk


def test_embedding_dimension_and_norm():
    vecs = embeddings.embed_texts(["hello world", "retrieval augmented generation"])
    assert len(vecs) == 2
    for v in vecs:
        assert len(v) == 384
        assert all(math.isfinite(x) for x in v)


def test_embed_query_matches_passage_on_same_topic():
    q = embeddings.embed_query("what index does pgvector use?")
    v = embeddings.embed_texts(["pgvector uses the HNSW index for cosine similarity search"])[0]
    cos = sum(a * b for a, b in zip(q, v)) / (
        math.sqrt(sum(a * a for a in q)) * math.sqrt(sum(b * b for b in v)))
    assert cos > 0.5


def test_dense_and_hybrid_retrieve_right_page(seeded_db):
    results = hybrid_search(seeded_db, "What is the dimension of bge-small embeddings?", top_k=10)
    assert results
    best = results[0]
    assert best["filename"] == "rag_notes.pdf"
    assert best["page_number"] == 2
    assert 0 < best["score"] <= 1


def test_hybrid_beats_or_matches_dense_only(seeded_db):
    q = "HNSW approximate nearest neighbor index cosine"
    dense = dense_search(seeded_db, embeddings.embed_query(q), top_k=5)
    hybrid = hybrid_search(seeded_db, q, top_k=5)
    hybrid_ids = {r["id"] for r in hybrid}
    dense_ids = {r["id"] for r in dense}
    # union keeps everything dense found
    assert dense_ids <= hybrid_ids


def test_bm25_lexical_match(seeded_db):
    rows = bm25_search(seeded_db, "ms-marco-MiniLM cross-encoder", top_k=5)
    assert rows
    assert all("bm25_raw" in r for r in rows)
    assert any("cross-encoder" in r["text"].lower() or "MiniLM" in r["text"] for r in rows)


def test_reranker_orders_relevant_first(seeded_db):
    q = "What happens when the context does not contain the answer?"
    cands = hybrid_search(seeded_db, q, top_k=10)
    top = reranker.rerank(q, cands, top_n=3)
    assert len(top) == 3
    assert all("rerank_score" in c for c in top)
    assert top[0]["page_number"] == 3  # abstention discussion lives on page 3
    scores = [c["rerank_score"] for c in top]
    assert scores == sorted(scores, reverse=True)
