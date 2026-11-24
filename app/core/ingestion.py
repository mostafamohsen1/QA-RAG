"""Document parsing (PDF/DOCX/TXT/MD) and recursive-overlap chunking."""

import hashlib
import re
from dataclasses import dataclass

import pymupdf
from docx import Document as DocxDocument


@dataclass
class ParsedDoc:
    text: str
    n_pages: int
    # page_number -> (char_start, char_end) span of that page inside `text`
    page_spans: list[tuple[int, int, int]]  # (page_number, start, end)


@dataclass
class ChunkMeta:
    doc_id: str
    filename: str
    page_number: int
    chunk_index: int
    char_start: int
    char_end: int
    text: str


def sha256_content(data: bytes) -> str:
    return hashlib.sha256(data).hexdigest()


def _clean(text: str) -> str:
    text = text.replace("\x00", " ").replace("\r\n", "\n").replace("\r", "\n")
    text = re.sub(r"[ \t]+", " ", text)
    text = re.sub(r"\n{3,}", "\n\n", text)
    return text.strip()


def parse_pdf(data: bytes) -> ParsedDoc:
    try:
        with pymupdf.open(stream=data, filetype="pdf") as doc:
            parts: list[str] = []
            spans: list[tuple[int, int, int]] = []
            cursor = 0
            for i, page in enumerate(doc, start=1):
                page_text = _clean(page.get_text("text") or "")
                if not page_text:
                    page_text = ""
                parts.append(page_text)
                start, end = cursor, cursor + len(page_text)
                spans.append((i, start, end))
                cursor = end + 2  # account for the "\n\n" page separator
            full = "\n\n".join(parts)
            return ParsedDoc(text=full, n_pages=len(doc), page_spans=spans)
    except Exception as exc:
        raise ValueError(f"Could not parse PDF: {exc}") from exc


def parse_docx(data: bytes) -> ParsedDoc:
    try:
        import io

        doc = DocxDocument(io.BytesIO(data))
        paragraphs = [p.text for p in doc.paragraphs]
        return ParsedDoc(text=_clean("\n".join(paragraphs)), n_pages=1, page_spans=[(1, 0, 10**9)])
    except Exception as exc:
        raise ValueError(f"Could not parse DOCX: {exc}") from exc


def parse_text(data: bytes) -> ParsedDoc:
    text = data.decode("utf-8", errors="replace")
    return ParsedDoc(text=_clean(text), n_pages=1, page_spans=[(1, 0, 10**9)])


def parse_document(filename: str, data: bytes) -> ParsedDoc:
    suffix = filename.lower().rsplit(".", 1)[-1]
    if suffix == "pdf":
        return parse_pdf(data)
    if suffix == "docx":
        return parse_docx(data)
    if suffix in ("txt", "md"):
        return parse_text(data)
    raise ValueError(f"Unsupported file type: .{suffix}")


def _split_paragraph(text: str, max_len: int) -> list[str]:
    """Last-resort hard split of an oversized paragraph on sentence/word boundaries."""
    pieces = re.split(r"(?<=[.!?])\s+", text)
    out: list[str] = []
    buf = ""
    for piece in pieces:
        if len(buf) + len(piece) + 1 <= max_len or not buf:
            buf = f"{buf} {piece}".strip()
        else:
            out.append(buf)
            buf = piece
    if buf:
        out.append(buf)
    return out


def _recursive_split(text: str, chunk_size: int, overlap: int) -> list[tuple[int, int, str]]:
    """Recursive character splitter: paragraphs -> sentences -> hard cut.
    Returns (char_start, char_end, chunk_text) triples with exact offsets."""
    if len(text) <= chunk_size:
        return [(0, len(text), text)] if text.strip() else []

    # atomic pieces: (char_start, text) of every paragraph / sentence run
    pieces: list[tuple[int, str]] = []
    pos = 0
    for para in text.split("\n\n"):
        p_start = pos
        p_len = len(para)
        pos += p_len + 2  # the "\n\n" separator
        para = para.strip()
        if not para:
            continue
        lead = p_len - len(para.lstrip())
        if len(para) <= chunk_size:
            pieces.append((p_start + lead, para))
        else:
            # sentence-level split inside an oversized paragraph
            cursor = p_start + lead
            remaining = para
            while remaining:
                run = ""
                for sent in re.split(r"(?<=[.!?])\s+", remaining):
                    if run and len(run) + 1 + len(sent) > chunk_size:
                        break
                    run = f"{run} {sent}".strip() if run else sent
                if not run:  # single sentence longer than chunk_size: hard cut
                    run = remaining[:chunk_size]
                idx = text.find(run, cursor)
                if idx == -1:
                    idx = cursor
                pieces.append((idx, run))
                remaining = remaining[len(run):].lstrip()
                cursor = idx + len(run)

    chunks: list[tuple[int, int, str]] = []
    buf: list[tuple[int, str]] = []  # (start, text)

    def buf_len(b: list[tuple[int, str]]) -> int:
        return sum(len(t) for _, t in b) + 2 * (len(b) - 1)

    def flush(b: list[tuple[int, str]]) -> None:
        if not b:
            return
        start = b[0][0]
        end = b[-1][0] + len(b[-1][1])
        chunks.append((start, end, "\n\n".join(t for _, t in b)))

    for i, (p_start, p_text) in enumerate(pieces):
        if buf and buf_len(buf) + 2 + len(p_text) > chunk_size:
            flush(buf)
            # carry a trimmed ~`overlap`-char tail into the new buffer, keeping offsets
            tail: list[tuple[int, str]] = []
            if overlap > 0:
                prev_start, prev_end, prev_text = chunks[-1]
                max_tail = min(overlap, chunk_size - len(p_text) - 2)
                if max_tail > 0:
                    tail_text = prev_text[-max_tail:]
                    tail = [(prev_end - len(tail_text), tail_text)]
            buf = tail
        buf.append((p_start, p_text))
    flush(buf)
    return chunks


def chunk_document(parsed: ParsedDoc, filename: str, chunk_size: int = 1000,
                   chunk_overlap: int = 150) -> list[ChunkMeta]:
    doc_id = sha256_content(parsed.text.encode("utf-8"))
    chunks: list[ChunkMeta] = []
    for i, (char_start, char_end, raw) in enumerate(
            _recursive_split(parsed.text, chunk_size, chunk_overlap)):
        # attribute the page by the chunk's midpoint (chunks can span page boundaries)
        page_number = 1 if parsed.n_pages == 1 else parsed.n_pages
        midpoint = char_start + (char_end - char_start) // 2
        for page_no, start, end in parsed.page_spans:
            if start <= midpoint < end:
                page_number = page_no
                break

        chunks.append(ChunkMeta(
            doc_id=doc_id,
            filename=filename,
            page_number=page_number,
            chunk_index=i,
            char_start=char_start,
            char_end=char_end,
            text=raw.strip(),
        ))
    return chunks
