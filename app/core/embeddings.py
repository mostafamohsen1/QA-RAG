"""CPU embedder via fastembed (ONNX Runtime). Loaded once as a singleton."""

import threading

from fastembed import TextEmbedding

from app.config import settings

_lock = threading.Lock()
_model: TextEmbedding | None = None


def get_embedder() -> TextEmbedding:
    global _model
    with _lock:
        if _model is None:
            _model = TextEmbedding(model_name=settings.embedding_model)
        return _model


def embed_texts(texts: list[str], batch_size: int | None = None) -> list[list[float]]:
    """Embed texts in batches; returns one 384-dim vector per input."""
    if not texts:
        return []
    batch = batch_size or settings.embedding_batch_size
    out: list[list[float]] = []
    embedder = get_embedder()
    for i in range(0, len(texts), batch):
        batch_texts = [f"passage: {t}" for t in texts[i:i + batch]]
        out.extend([list(map(float, v)) for v in embedder.embed(batch_texts)])
    return out


def embed_query(query: str) -> list[float]:
    embedder = get_embedder()
    return list(map(float, next(iter(embedder.embed([f"query: {query}"])))))
