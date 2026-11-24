import json
import re

from fastapi import APIRouter, Depends, HTTPException, Request
from fastapi.responses import StreamingResponse
from sqlalchemy import text
from sqlalchemy.orm import Session

from app.api.schemas import (DocumentInfo, HealthResponse, IngestResponse,
                             QueryRequest, QueryResponse)
from app.config import settings
from app.core import llm
from app.core.pipeline import ingest_document, query, query_stream
from app.db.database import get_db
from app.db.models import Document

router = APIRouter()


@router.get("/health", response_model=HealthResponse)
def health() -> HealthResponse:
    return HealthResponse(
        status="ok",
        embedding_model=settings.embedding_model,
        reranker_model=settings.reranker_model,
        llm_provider=llm.resolve_provider(),
        llm_model=settings.llm_model,
    )


def _detect_type(filename: str | None, data: bytes) -> str | None:
    """Resolve the file type: magic bytes first, then the declared extension,
    then a UTF-8 text sniff when there is no extension. None means reject."""
    if data.startswith(b"%PDF"):
        return ".pdf"
    if data.startswith(b"PK\x03\x04"):  # OOXML zip container
        return ".docx"
    ext = "." + filename.rsplit(".", 1)[-1].lower() if (filename and "." in filename) else ""
    if ext in settings.allowed_extensions:
        return ext
    if ext:
        return None  # the name declares a non-document type — reject it
    try:  # no extension: accept plain text only
        data.decode("utf-8")
        return ".txt"
    except UnicodeDecodeError:
        return None


@router.post("/documents", response_model=IngestResponse)
async def upload_document(request: Request, db: Session = Depends(get_db)):
    """Tolerant upload endpoint.

    Parses the request body directly (no FastAPI File binding — Starlette forbids
    reading the body twice, and some mobile WebViews / preview proxies re-encode
    multipart so FastAPI's strict parser would 422 before we run). Accepts:
    1. multipart/form-data with a 'file' part that has a filename
    2. multipart/form-data whose part lost its filename — we find the payload part
       ourselves and take the filename from ?filename= if present
    3. raw bytes with ?filename=doc.pdf
    The file type is verified from magic bytes, not just the extension.
    """
    filename: str | None = request.query_params.get("filename")
    content_type = request.headers.get("content-type", "")
    data: bytes = b""
    raw_body = await request.body()  # read exactly once

    if content_type.startswith("multipart/") and raw_body.startswith(b"--"):
        boundary_match = re.search(r'boundary="?([^";]+)"?', content_type)
        if boundary_match:
            boundary = boundary_match.group(1).encode()
            for part in raw_body.split(b"--" + boundary):
                part = part.strip(b"\r\n")
                if not part or part == b"--" or b"\r\n\r\n" not in part:
                    continue
                headers_blob, payload = part.split(b"\r\n\r\n", 1)
                fn = re.search(rb'filename="([^"]*)"', headers_blob)
                if fn and fn.group(1):
                    filename = fn.group(1).decode("utf-8", "replace")
                    data = payload
                    break
                if not data and payload:
                    data = payload  # candidate without filename; keep scanning
    else:
        data = raw_body

    if not filename:
        filename = "upload.pdf"

    suffix = _detect_type(filename, data)
    if suffix is None or suffix not in settings.allowed_extensions:
        raise HTTPException(415, "That file is not a recognizable PDF, DOCX, TXT or MD "
                                 "document (checked by content, not just the name).")
    if not data:
        raise HTTPException(400, "Empty file")
    if len(data) > 50 * 1024 * 1024:
        raise HTTPException(413, "File larger than 50 MB")
    try:
        result = ingest_document(db, filename if filename.endswith(suffix) else filename + suffix, data)
    except ValueError as exc:
        raise HTTPException(422, str(exc)) from exc
    except Exception as exc:
        raise HTTPException(500, f"Ingestion failed: {exc}") from exc
    return IngestResponse(**result)


@router.get("/documents", response_model=list[DocumentInfo])
def list_documents(db: Session = Depends(get_db)):
    docs = db.query(Document).order_by(Document.created_at.desc()).all()
    return [DocumentInfo(doc_id=d.doc_id, filename=d.filename, n_pages=d.n_pages,
                         n_chunks=d.n_chunks, file_size=d.file_size) for d in docs]


@router.delete("/documents/{doc_id}")
def delete_document(doc_id: str, db: Session = Depends(get_db)):
    doc = db.query(Document).filter_by(doc_id=doc_id).first()
    if not doc:
        raise HTTPException(404, "Document not found")
    db.delete(doc)
    db.commit()
    return {"deleted": doc_id}


@router.post("/query", response_model=QueryResponse)
def run_query(body: QueryRequest, db: Session = Depends(get_db)):
    try:
        result = query(db, body.question, top_k=body.top_k, top_n=body.top_n,
                       doc_ids=body.doc_ids)
    except ValueError as exc:
        raise HTTPException(422, str(exc)) from exc
    except RuntimeError as exc:
        raise HTTPException(503, str(exc)) from exc
    return QueryResponse(**result)


@router.post("/query/stream")
def run_query_stream(body: QueryRequest, db: Session = Depends(get_db)):
    """Server-sent-events query: citations first, then answer tokens as the LLM
    produces them, then a final done event with citations/abstain metadata."""
    def sse():
        try:
            for event in query_stream(db, body.question, top_k=body.top_k,
                                      top_n=body.top_n, doc_ids=body.doc_ids):
                yield f"data: {json.dumps(event)}\n\n"
        except ValueError as exc:
            yield f"data: {json.dumps({'type': 'error', 'message': str(exc)})}\n\n"
        except Exception as exc:
            yield f"data: {json.dumps({'type': 'error', 'message': f'Query failed: {exc}'})}\n\n"

    return StreamingResponse(sse(), media_type="text/event-stream",
                             headers={"Cache-Control": "no-cache",
                                      "X-Accel-Buffering": "no"})


@router.post("/embeddings/hygiene")
def embedding_hygiene(db: Session = Depends(get_db)):
    """Diagnostics: dimension, NULLs and duplicate chunk coverage."""
    row = db.execute(text("""
        SELECT count(*) AS total,
               count(*) FILTER (WHERE embedding IS NULL) AS nulls,
               count(DISTINCT doc_id) AS docs,
               avg(array_length(embedding::real[], 1)) AS avg_dim
        FROM chunks
    """)).mappings().first()
    return dict(row)
