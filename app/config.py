from pathlib import Path

from pydantic_settings import BaseSettings, SettingsConfigDict


class Settings(BaseSettings):
    model_config = SettingsConfigDict(env_file=".env", env_file_encoding="utf-8", extra="ignore")

    database_url: str = "postgresql+psycopg://postgres:postgres@localhost:5432/rag"

    embedding_model: str = "BAAI/bge-small-en-v1.5"
    reranker_model: str = "Xenova/ms-marco-MiniLM-L-6-v2"
    embedding_batch_size: int = 32

    retrieval_top_k: int = 20
    rerank_top_n: int = 5

    # auto = anthropic if ANTHROPIC_API_KEY is set, else an OpenAI-compatible key,
    # else the local Ollama server
    llm_provider: str = "auto"  # auto | openai_compat | anthropic | ollama
    llm_model: str = "llama3.2:3b"
    llm_timeout_seconds: int = 120
    llm_retries: int = 2
    llm_temperature: float = 0.1
    llm_max_tokens: int = 1024

    openai_base_url: str = "http://localhost:11434/v1"
    openai_api_key: str = "change-me"
    ollama_base_url: str = "http://localhost:11434"
    anthropic_base_url: str = "https://api.anthropic.com"
    anthropic_api_key: str = ""
    anthropic_version: str = "2023-06-01"

    chunk_size: int = 1000
    chunk_overlap: int = 150
    upload_dir: Path = Path("./uploads")

    # Extractive fallback: how many sentences and how many characters to quote
    max_extractive_sentences: int = 3
    max_extractive_chars: int = 600

    allowed_extensions: frozenset[str] = frozenset({".pdf", ".docx", ".txt", ".md"})


settings = Settings()
