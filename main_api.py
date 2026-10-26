"""FastAPI deployment interface for the research-paper RAG pipeline."""

from __future__ import annotations

from threading import Lock
from typing import Any

from fastapi import FastAPI, HTTPException
from fastapi.staticfiles import StaticFiles
from pydantic import BaseModel, Field

from config import DATASET_DIR, VECTOR_STORE_DIR
from src.multilingual import SUPPORTED_LANGUAGES
from src.rag_pipeline import answer_query_details, build_and_save_index

app = FastAPI(title="Research Intelligence RAG API", version="1.0.0")
app.mount(
    "/images",
    StaticFiles(directory=str(VECTOR_STORE_DIR / "figures"), check_dir=False),
    name="images",
)
_index_lock = Lock()


class AskRequest(BaseModel):
    """Question and generation options accepted by the /ask endpoint."""

    question: str = Field(min_length=1, max_length=10_000)
    response_language: str | None = Field(default=None, max_length=32)
    top_k: int = Field(default=5, ge=1, le=20)


class IndexRequest(BaseModel):
    """Optional index input locations for the /index endpoint."""

    dataset_dir: str | None = None
    vector_store_dir: str | None = None


@app.get("/health")
def health() -> dict[str, Any]:
    """Report API readiness and whether a persisted index is present."""
    return {
        "status": "ok",
        "index_ready": (VECTOR_STORE_DIR / "metadata.pkl").is_file(),
        "supported_languages": list(SUPPORTED_LANGUAGES),
    }


@app.post("/ask")
def ask(request: AskRequest) -> dict[str, Any]:
    """Retrieve and generate a grounded answer with route/citation metadata."""
    question = request.question.strip()
    if not question:
        raise HTTPException(status_code=422, detail="Question must not be blank.")
    try:
        return answer_query_details(
            question,
            vector_store_dir=VECTOR_STORE_DIR,
            top_k=request.top_k,
            response_language=request.response_language,
        )
    except (OSError, RuntimeError, ValueError, KeyError) as exc:
        raise HTTPException(status_code=503, detail=f"RAG query failed: {exc}") from exc


@app.post("/index")
def index(request: IndexRequest | None = None) -> dict[str, Any]:
    """Build and persist the selected dataset's embedding indexes."""
    payload = request or IndexRequest()
    dataset_dir = payload.dataset_dir or str(DATASET_DIR)
    vector_store_dir = payload.vector_store_dir or str(VECTOR_STORE_DIR)
    if not _index_lock.acquire(blocking=False):
        raise HTTPException(status_code=409, detail="An index build is already running.")
    try:
        metadata = build_and_save_index(dataset_dir, vector_store_dir)
        return {
            "status": "indexed",
            "dataset_dir": dataset_dir,
            "vector_store_dir": vector_store_dir,
            "chunk_count": len(metadata.get("chunks", [])),
            "embedding_keys": metadata.get("embedding_keys", []),
        }
    except (OSError, RuntimeError, ValueError) as exc:
        raise HTTPException(status_code=500, detail=f"Index build failed: {exc}") from exc
    finally:
        _index_lock.release()