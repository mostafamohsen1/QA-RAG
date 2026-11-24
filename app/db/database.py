from sqlalchemy import create_engine, text
from sqlalchemy.orm import DeclarativeBase, sessionmaker

from app.config import settings

engine = create_engine(settings.database_url, pool_pre_ping=True)
SessionLocal = sessionmaker(bind=engine, autoflush=False, expire_on_commit=False)


class Base(DeclarativeBase):
    pass


def init_db() -> None:
    """Create the pgvector extension, tables and search indexes."""
    with engine.begin() as conn:
        conn.execute(text("CREATE EXTENSION IF NOT EXISTS vector"))
    Base.metadata.create_all(engine)
    with engine.begin() as conn:
        # HNSW index for cosine-similarity retrieval (fast CPU ANN search)
        conn.execute(text(
            "CREATE INDEX IF NOT EXISTS idx_chunks_embedding_hnsw "
            "ON chunks USING hnsw (embedding vector_cosine_ops) WITH (m = 16, ef_construction = 64)"
        ))
        # GIN index for the BM25/full-text baseline via ts_rank_cd
        conn.execute(text(
            "CREATE INDEX IF NOT EXISTS idx_chunks_tsv ON chunks USING gin (tsv)"
        ))


def get_db():
    db = SessionLocal()
    try:
        yield db
    finally:
        db.close()
