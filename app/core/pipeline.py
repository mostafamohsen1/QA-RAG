"""End-to-end RAG pipeline: ingest (parse->chunk->embed->store) and query
(retrieve->rerank->generate)."""

import re

from sqlalchemy.orm import Session

from app.config import settings
from app.core import embeddings, llm, reranker
from app.core.ingestion import chunk_document, parse_document, sha256_content
from app.core.retriever import hybrid_search
from app.db.models import Chunk, Document


def ingest_document(db: Session, filename: str, data: bytes,
                    chunk_size: int | None = None, chunk_overlap: int | None = None) -> dict:
    """Parse, chunk, embed and store a document. Re-ingesting the same content
    returns the existing document (content-hash idempotency)."""
    doc_id = sha256_content(data)
    existing = db.query(Document).filter_by(doc_id=doc_id).first()
    if existing:
        return {
            "doc_id": existing.doc_id, "filename": existing.filename,
            "n_pages": existing.n_pages, "n_chunks": existing.n_chunks,
            "deduplicated": True,
        }

    parsed = parse_document(filename, data)
    if not parsed.text.strip():
        raise ValueError("Document has no extractable text")
    chunks = chunk_document(
        parsed, filename,
        chunk_size=chunk_size or settings.chunk_size,
        chunk_overlap=chunk_overlap or settings.chunk_overlap,
    )
    vectors = embeddings.embed_texts([c.text for c in chunks])

    doc = Document(
        doc_id=doc_id, filename=filename, file_size=len(data),
        n_pages=parsed.n_pages, n_chunks=len(chunks),
    )
    db.add(doc)
    db.flush()

    for meta, vec in zip(chunks, vectors):
        db.add(Chunk(
            document_row_id=doc.id, doc_id=doc.doc_id, filename=meta.filename,
            page_number=meta.page_number, chunk_index=meta.chunk_index,
            char_start=meta.char_start, char_end=meta.char_end,
            text=meta.text, embedding=vec,
        ))
    # populate tsvector for the full-text baseline
    from sqlalchemy import text as sql_text
    db.execute(sql_text(
        "UPDATE chunks SET tsv = to_tsvector('english', text) WHERE doc_id = :d"
    ), {"d": doc_id})
    db.commit()
    return {
        "doc_id": doc_id, "filename": filename, "n_pages": parsed.n_pages,
        "n_chunks": len(chunks), "deduplicated": False,
    }


def _extractive_fallback(question: str, top: list[dict]) -> dict:
    """Best-effort answer without an LLM: pick the sentences from the retrieved
    chunks that overlap the question most, and cite their chunk. Keeps the app
    useful when no generative LLM endpoint is configured."""
    q_tokens = set(re.findall(r"[a-z0-9]+", question.lower()))
    if len(q_tokens) > 4:  # drop obvious stop words for scoring
        for w in ("what", "which", "who", "when", "where", "why", "how", "is",
                  "are", "the", "a", "an", "does", "do", "of", "in", "to", "and",
                  "about", "talk", "tell", "say", "this", "that", "it", "its",
                  "was", "were", "will", "can", "me", "my", "you", "your"):
            q_tokens.discard(w)
    scored: list[tuple[float, int, str]] = []
    for i, c in enumerate(top, start=1):
        for sent in re.split(r"(?<=[.!?])\s+", c["text"]):
            s_tokens = set(re.findall(r"[a-z0-9]+", sent.lower()))
            overlap = len(q_tokens & s_tokens) / max(1, len(q_tokens))
            if overlap > 0:
                scored.append((overlap, i, sent.strip()))
    scored.sort(reverse=True)
    picked, cited = [], []
    for _, i, sent in scored:
        if any(sent[:60] in p or p[:60] in sent for p in picked):
            continue
        picked.append(sent)
        if i not in cited:
            cited.append(i)
        if len(picked) >= settings.max_extractive_sentences or \
                len(" ".join(picked)) >= settings.max_extractive_chars:
            break
    if not picked:
        # no lexical overlap — still show the top-ranked chunk's opening lines
        # (retrieval matched it) instead of a hard refusal
        if top:
            c = top[0]
            lead = " ".join(re.split(r"(?<=[.!?])\s+", c["text"])[:2])[:400]
            return {"answer": f"{lead} [1]\n\n(Extractive answer — no exact sentence matched "
                     "\"all words of your question\"; this is the closest passage. No generative "
                     "LLM is configured.)",
                    "abstained": False, "cited_indexes": [1], "answer_mode": "extractive"}
        return {"answer": "I don't know based on the provided documents.",
                "abstained": True, "cited_indexes": [], "answer_mode": "extractive"}
    body = " ".join(picked)
    cites = " ".join(f"[{i}]" for i in cited)
    answer = f"{body} {cites}\n\n(Extractive answer — no generative LLM is configured, " \
             "so the most relevant passages are quoted directly.)"
    return {"answer": answer, "abstained": False,
            "cited_indexes": cited, "answer_mode": "extractive"}


def retrieve(db: Session, question: str, top_k: int | None = None,
             top_n: int | None = None, doc_ids: list[str] | None = None) -> list[dict]:
    """hybrid search -> cross-encoder rerank. Returns the top chunks."""
    if not question.strip():
        raise ValueError("Empty query")
    candidates = hybrid_search(db, question, top_k=top_k, doc_ids=doc_ids)
    if not candidates:
        return []
    return reranker.rerank(question, candidates, top_n=top_n or settings.rerank_top_n)


def _citation_list(top: list[dict], cited_indexes: list[int]) -> list[dict]:
    citations = []
    for idx in cited_indexes:
        c = top[idx - 1]
        citations.append({
            "index": idx, "doc_id": c["doc_id"], "filename": c["filename"],
            "page_number": c["page_number"], "chunk_index": c["chunk_index"],
            "text": c["text"][:400],
        })
    return citations


def query(db: Session, question: str, top_k: int | None = None,
          top_n: int | None = None, doc_ids: list[str] | None = None) -> dict:
    """retrieve -> rerank -> generate with citations. Falls back to an extractive
    (quoted) answer when no LLM endpoint is reachable."""
    top = retrieve(db, question, top_k=top_k, top_n=top_n, doc_ids=doc_ids)
    if not top:
        return {"question": question, "answer":
                "I don't know based on the provided documents.",
                "abstained": True, "citations": [], "retrieved": [],
                "answer_mode": "llm"}
    try:
        result = llm.generate_answer(question, top)
        result["answer_mode"] = "llm"
    except RuntimeError:
        result = _extractive_fallback(question, top)
    citations = _citation_list(top, result["cited_indexes"])
    return {
        "question": question,
        "answer": result["answer"],
        "abstained": result["abstained"],
        "answer_mode": result["answer_mode"],
        "citations": citations,
        "retrieved": [
            {"doc_id": c["doc_id"], "filename": c["filename"],
             "page_number": c["page_number"], "score": round(float(c["score"]), 4),
             "rerank_score": c.get("rerank_score")}
            for c in top
        ],
    }


def query_stream(db: Session, question: str, top_k: int | None = None,
                 top_n: int | None = None, doc_ids: list[str] | None = None) -> dict:
    """Same pipeline as query(), but generation streams token-by-token.

    Returns an iterable of events: {"type": "citations"|"token"|"done", ...}.
    Falls back to streaming the extractive answer as one token when no LLM is
    reachable."""
    top = retrieve(db, question, top_k=top_k, top_n=top_n, doc_ids=doc_ids)
    if not top:
        yield {"type": "done", "answer": "I don't know based on the provided documents.",
               "abstained": True, "citations": [], "answer_mode": "llm"}
        return

    citations = _citation_list(top, list(range(1, len(top) + 1)))
    yield {"type": "citations", "citations": citations,
           "retrieved": [
               {"doc_id": c["doc_id"], "filename": c["filename"],
                "page_number": c["page_number"], "score": round(float(c["score"]), 4),
                "rerank_score": c.get("rerank_score")}
               for c in top
           ]}

    parts: list[str] = []
    try:
        for token in llm.stream_answer(question, top):
            parts.append(token)
            yield {"type": "token", "text": token}
        answer = "".join(parts).strip()
        yield {"type": "done", "answer": answer, "abstained": llm.is_abstain(answer),
               "citations": _citation_list(top, llm.extract_citations(answer, len(top))),
               "answer_mode": "llm"}
    except RuntimeError:
        result = _extractive_fallback(question, top)
        yield {"type": "token", "text": result["answer"]}
        yield {"type": "done", "answer": result["answer"], "abstained": result["abstained"],
               "citations": _citation_list(top, result["cited_indexes"]),
               "answer_mode": "extractive"}
