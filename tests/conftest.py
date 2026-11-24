import textwrap

import pymupdf
import pytest

from app.core.pipeline import ingest_document
from app.db.database import SessionLocal, init_db
from app.db.models import Chunk, Document

PAGES = [
    # page 1
    "Retrieval-augmented generation (RAG) combines a retriever with a generative language "
    "model so that answers are grounded in an external knowledge source. This reduces "
    "hallucinations on facts the model was not trained on and keeps answers fresh when the "
    "corpus changes, because the model reads the retrieved passages at answer time instead of "
    "relying only on what it memorized during training.\n\n"
    "The recommended chunking strategy is recursive character splitting with a chunk size of "
    "about 1000 characters and an overlap of 100 to 200 characters. Overlap keeps context from "
    "being split mid-idea at a chunk boundary. Recursive splitting tries paragraph boundaries "
    "first, then sentence boundaries, and only hard-cuts when a single sentence is longer than "
    "the whole chunk. Every chunk keeps the character offsets it came from in the source text, "
    "so page numbers and citations stay accurate.",
    # page 2
    "The embedding model is BAAI/bge-small-en-v1.5. It produces 384 dimensional vectors and "
    "runs on CPU through fastembed and ONNX Runtime, with no PyTorch dependency. Each chunk is "
    "stored in Postgres together with its vector in a pgvector column, and queries and passages "
    "are prefixed differently so retrieval quality stays high.\n\n"
    "For approximate nearest neighbor search, pgvector uses the HNSW index optimizing cosine "
    "similarity distance. With HNSW, queries stay fast even with millions of chunks, and the "
    "index is built once at table creation time with m equal to sixteen and ef_construction "
    "equal to sixty four.\n\n"
    "The reranker is a cross-encoder, ms-marco-MiniLM. A cross-encoder jointly reads the query "
    "and each candidate chunk and rescores them together, which is more accurate than the "
    "bi-encoder similarity used to fetch the initial candidates. The pipeline retrieves the "
    "top twenty candidates and keeps the best five after reranking.",
    # page 3
    "When the retrieved context does not contain the answer, the system must abstain: it says "
    "it does not know based on the provided documents instead of hallucinating an answer. The "
    "prompt forbids outside knowledge and requires a citation after every claim, and the "
    "abstention string is matched downstream to measure how often the system refuses to guess.\n\n"
    "The defined evaluation metrics are retrieval hit at k, mean reciprocal rank, context "
    "precision and recall against labeled contexts, answer correctness by token overlap, and "
    "faithfulness via an LLM as judge. Abstain accuracy is measured on questions that no "
    "document can answer, where the only correct behavior is to say that the answer is unknown.",
]


def _wrapped_pdf_bytes(pages: list[str]) -> bytes:
    doc = pymupdf.open()
    for text in pages:
        page = doc.new_page()
        y = 72
        for line in textwrap.wrap(text, width=90):
            page.insert_text((72, y), line, fontsize=10)
            y += 14
            if y > 780:  # new column, still same page
                y = 72
                pass
    return doc.tobytes()


@pytest.fixture(scope="session")
def seeded_db():
    init_db()
    db = SessionLocal()
    # clean slate
    db.query(Chunk).delete()
    db.query(Document).delete()
    db.commit()

    ingest_document(db, "rag_notes.pdf", _wrapped_pdf_bytes(PAGES))
    yield db
    db.close()

