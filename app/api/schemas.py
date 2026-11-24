from pydantic import BaseModel, Field


class QueryRequest(BaseModel):
    question: str = Field(min_length=1, max_length=4000)
    top_k: int | None = Field(default=None, ge=1, le=50)
    top_n: int | None = Field(default=None, ge=1, le=20)
    doc_ids: list[str] | None = None


class Citation(BaseModel):
    index: int
    doc_id: str
    filename: str
    page_number: int
    chunk_index: int
    text: str


class RetrievedChunk(BaseModel):
    doc_id: str
    filename: str
    page_number: int
    score: float
    rerank_score: float | None = None


class QueryResponse(BaseModel):
    question: str
    answer: str
    abstained: bool
    answer_mode: str = "llm"  # llm | extractive
    citations: list[Citation]
    retrieved: list[RetrievedChunk]


class IngestResponse(BaseModel):
    doc_id: str
    filename: str
    n_pages: int
    n_chunks: int
    deduplicated: bool = False


class DocumentInfo(BaseModel):
    doc_id: str
    filename: str
    n_pages: int
    n_chunks: int
    file_size: int


class HealthResponse(BaseModel):
    status: str
    embedding_model: str
    reranker_model: str
    llm_provider: str
    llm_model: str
