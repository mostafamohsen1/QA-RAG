from contextlib import asynccontextmanager
from pathlib import Path

from fastapi import FastAPI
from fastapi.responses import FileResponse
from fastapi.staticfiles import StaticFiles

from app.api.routes import router
from app.config import settings
from app.db.database import init_db


@asynccontextmanager
async def lifespan(app: FastAPI):
    settings.upload_dir.mkdir(parents=True, exist_ok=True)
    init_db()  # create pgvector extension, tables, HNSW + GIN indexes
    yield


app = FastAPI(title="Document QA RAG", version="1.0.0", lifespan=lifespan)
app.include_router(router, prefix="/api")

app.mount("/static", StaticFiles(directory=Path(__file__).parent / "static"), name="static")


@app.get("/")
def index():
    return FileResponse(Path(__file__).parent / "static" / "index.html")
