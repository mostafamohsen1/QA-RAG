import pytest
from fastapi.testclient import TestClient

from app.core import llm
from app.main import app


@pytest.fixture
def client(seeded_db, monkeypatch):
    from app.db.database import get_db

    app.dependency_overrides[get_db] = lambda: seeded_db

    def fake_generate_answer(question, chunks):
        # simulate a grounded, citing answer
        if chunks:
            return {"answer": f"Grounded answer with [1] citation.",
                    "abstained": False, "cited_indexes": [1]}
        return {"answer": "I don't know based on the provided documents.",
                "abstained": True, "cited_indexes": []}

    monkeypatch.setattr(llm, "generate_answer", fake_generate_answer)
    with TestClient(app) as c:  # context manager triggers lifespan (init_db)
        yield c
    app.dependency_overrides.clear()


def _pdf(text):
    from tests.conftest import _wrapped_pdf_bytes
    return _wrapped_pdf_bytes([text])


def test_health(client):
    r = client.get("/api/health")
    assert r.status_code == 200
    body = r.json()
    assert body["status"] == "ok"
    assert body["embedding_model"] == "BAAI/bge-small-en-v1.5"


def test_upload_list_delete_cycle(client):
    files = {"file": ("cycle.pdf", _pdf("Test document about the upload cycle."), "application/pdf")}
    r = client.post("/api/documents", files=files)
    assert r.status_code == 200, r.text
    body = r.json()
    assert body["n_chunks"] >= 1 and body["n_pages"] == 1
    doc_id = body["doc_id"]

    r = client.get("/api/documents")
    assert any(d["doc_id"] == doc_id for d in r.json())

    # duplicate upload -> deduplicated
    r2 = client.post("/api/documents", files=files)
    assert r2.json()["deduplicated"] is True

    r = client.delete(f"/api/documents/{doc_id}")
    assert r.status_code == 200
    assert client.delete(f"/api/documents/{doc_id}").status_code == 404


def test_upload_rejects_bad_type(client):
    r = client.post("/api/documents", files={"file": ("x.exe", b"MZ", "application/octet-stream")})
    assert r.status_code == 415
    r = client.post("/api/documents", files={"file": ("empty.pdf", b"", "application/pdf")})
    assert r.status_code == 400


def test_query_with_citations(client):
    r = client.post("/api/query", json={"question": "What chunk size does the spec recommend?"})
    assert r.status_code == 200, r.text
    body = r.json()
    assert body["abstained"] is False
    assert body["citations"], "expected at least one citation"
    c = body["citations"][0]
    assert c["filename"] == "rag_notes.pdf" and c["page_number"] >= 1
    assert body["retrieved"], "expected retrieved chunk metadata"


def test_upload_multipart_without_filename_still_works(client):
    """Regression: some mobile WebViews/proxies strip the filename from the part;
    the endpoint must parse the body itself instead of 422ing."""
    pdf = _pdf("Uploaded without a filename in the multipart part.")
    boundary = "----regressiontest"
    body = (f"--{boundary}\r\n"
            f"Content-Disposition: form-data; name=\"file\"\r\n\r\n").encode() + pdf + \
           f"\r\n--{boundary}--\r\n".encode()
    r = client.post("/api/documents?filename=noname.pdf", content=body,
                    headers={"Content-Type": f"multipart/form-data; boundary={boundary}"})
    assert r.status_code == 200, r.text
    assert r.json()["n_chunks"] >= 1
    assert client.delete("/api/documents/" + r.json()["doc_id"]).status_code == 200


def test_upload_raw_bytes_with_query_filename(client):
    r = client.post("/api/documents?filename=raw.txt",
                    content="raw body upload path".encode(),
                    headers={"Content-Type": "application/octet-stream"})
    assert r.status_code == 200, r.text
    assert r.json()["filename"].endswith(".txt")
    client.delete("/api/documents/" + r.json()["doc_id"])


def test_query_empty_rejected(client):
    r = client.post("/api/query", json={"question": "   "})
    assert r.status_code == 422


def test_query_falls_back_to_extractive_when_llm_down(seeded_db, monkeypatch):
    from fastapi.testclient import TestClient

    from app.core import pipeline
    from app.db.database import get_db
    from app.main import app

    def _llm_down(*args, **kwargs):
        raise RuntimeError("LLM call failed: connection refused (test)")

    monkeypatch.setattr(pipeline.llm, "generate_answer", _llm_down)
    app.dependency_overrides[get_db] = lambda: seeded_db
    try:
        with TestClient(app) as c:
            r = c.post("/api/query", json={"question": "What does a cross-encoder do?"})
        assert r.status_code == 200, r.text
        body = r.json()
        assert body["answer_mode"] == "extractive"
        assert body["abstained"] is False
        assert "no generative LLM" in body["answer"]
        assert body["citations"], "extractive answers must still cite chunks"
    finally:
        app.dependency_overrides.clear()


def test_extractive_fallback_quotes_closest_passage_when_no_overlap(seeded_db):
    """Generic questions get the closest passage, not a hard refusal — the
    hard refusal is reserved for when retrieval itself finds nothing."""
    from app.core.pipeline import _extractive_fallback

    top = [{"text": "Cats are common pets.", "doc_id": "d", "filename": "f",
            "page_number": 1, "chunk_index": 0, "id": 1, "score": 0.9,
            "rerank_score": None}]
    result = _extractive_fallback("quantum flux hyperdrive capacity", top)
    assert result["abstained"] is False
    assert "[1]" in result["answer"]
    assert "closest passage" in result["answer"]
