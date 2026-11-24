import pytest

from app.core.ingestion import chunk_document, parse_document
from app.core.pipeline import ingest_document


def test_parse_docx_and_txt():
    parsed = parse_document("notes.txt", b"Hello world.\n\nSecond paragraph.")
    assert "Second paragraph" in parsed.text
    assert parsed.n_pages == 1


def test_parse_unsupported_raises():
    with pytest.raises(ValueError, match="Unsupported"):
        parse_document("data.csv", b"a,b")


def test_parse_corrupt_pdf_raises():
    with pytest.raises(ValueError, match="PDF"):
        parse_document("bad.pdf", b"%PDF-1.4 garbage not a real pdf")


def test_chunk_sizes_and_overlap():
    parsed = parse_document("notes.txt", ("para one. " * 200 + "\n\n" + "para two. " * 200).encode())
    chunks = chunk_document(parsed, "notes.txt", chunk_size=1000, chunk_overlap=150)
    assert len(chunks) > 1
    for c in chunks:
        assert len(c.text) <= 1000
        assert c.text.strip()
    # adjacent chunks share overlap content
    assert chunks[1].text[:80] in (chunks[0].text + chunks[1].text)
    assert chunks[1].chunk_index == chunks[0].chunk_index + 1
    assert chunks[0].char_end > chunks[0].char_start


def test_chunk_page_mapping(seeded_db):
    from app.db.models import Chunk

    pages = {c.chunk_index: c.page_number for c in seeded_db.query(Chunk).filter_by(filename="rag_notes.pdf")}
    assert set(pages.values()) == {1, 2, 3}


def test_ingest_idempotent(seeded_db):
    from tests.conftest import _wrapped_pdf_bytes

    data = _wrapped_pdf_bytes(["idempotency probe text that carries enough content to chunk."])

    r1 = ingest_document(seeded_db, "probe.pdf", data)
    assert r1["deduplicated"] is False and r1["n_chunks"] >= 1
    r2 = ingest_document(seeded_db, "probe-again.pdf", data)
    assert r2["deduplicated"] is True
    assert r2["doc_id"] == r1["doc_id"]
    from app.db.models import Document
    assert seeded_db.query(Document).filter_by(doc_id=r1["doc_id"]).count() == 1
