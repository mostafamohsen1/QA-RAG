"""Cross-encoder reranker via fastembed's TextCrossEncoder (CPU, ONNX)."""

import threading

from fastembed.rerank.cross_encoder import TextCrossEncoder

from app.config import settings

_lock = threading.Lock()
_model: TextCrossEncoder | None = None


def get_reranker() -> TextCrossEncoder:
    global _model
    with _lock:
        if _model is None:
            _model = TextCrossEncoder(model_name=settings.reranker_model)
        return _model


def rerank(query: str, candidates: list[dict], top_n: int | None = None) -> list[dict]:
    """Return candidates sorted by cross-encoder score, best `top_n` first.

    Each candidate dict gets a `rerank_score` key. Falls back to the incoming
    order on unexpected internal errors so retrieval never hard-fails.
    """
    if not candidates:
        return []
    top_n = top_n or settings.rerank_top_n
    try:
        model = get_reranker()
        docs = [c["text"] for c in candidates]
        scores = list(model.rerank(query, docs))
    except Exception:
        return list(candidates[:top_n])
    for cand, score in zip(candidates, scores):
        cand["rerank_score"] = float(score)
    ranked = sorted(candidates, key=lambda c: c.get("rerank_score", 0.0), reverse=True)
    return ranked[:top_n]
