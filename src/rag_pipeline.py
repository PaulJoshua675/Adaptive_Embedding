from __future__ import annotations

import pickle
from pathlib import Path
from typing import Dict, List

from config import DEFAULT_CHUNK_OVERLAP, DEFAULT_CHUNK_SIZE, DEFAULT_EMBEDDING_MODEL, DEFAULT_TOP_K, METADATA_PATH, VECTOR_STORE_DIR
from .chunk_documents import chunk_documents
from .create_vector_db import VectorStore
from .embedding_model import EmbeddingModel
from .extract_pdf import extract_documents_from_path
from .llm import SimpleLLM
from .retriever import retrieve_top_chunks
from .utils import ensure_directory, write_text


def build_and_save_index(dataset_dir: str | Path, vector_store_dir: str | Path | None = None) -> Dict[str, object]:
    dataset_path = Path(dataset_dir)
    store_dir = Path(vector_store_dir or VECTOR_STORE_DIR)
    ensure_directory(store_dir)

    documents = extract_documents_from_path(dataset_path)
    chunks = chunk_documents(documents, chunk_size=DEFAULT_CHUNK_SIZE, overlap=DEFAULT_CHUNK_OVERLAP)

    if documents:
        extracted_dir = store_dir.parent / "extracted_text"
        ensure_directory(extracted_dir)
        for document in documents:
            write_text(extracted_dir / f"{document['id']}.txt", str(document["text"]))

    embedding_model = EmbeddingModel(model_name=DEFAULT_EMBEDDING_MODEL)
    chunk_texts = [str(chunk["text"]) for chunk in chunks]
    embeddings = embedding_model.embed_texts(chunk_texts)

    vector_store = VectorStore(dimension=len(embeddings[0]) if embeddings else 1)
    vector_store.add_documents(chunks, embeddings)

    metadata = {
        "chunks": chunks,
        "embeddings": embeddings,
        "embedding_model_name": embedding_model.model_name,
        "chunk_size": DEFAULT_CHUNK_SIZE,
        "overlap": DEFAULT_CHUNK_OVERLAP,
        "top_k": DEFAULT_TOP_K,
    }

    metadata_path = store_dir / "metadata.pkl"
    with metadata_path.open("wb") as handle:
        pickle.dump(metadata, handle)

    return metadata


def load_index(vector_store_dir: str | Path | None = None) -> Dict[str, object]:
    store_dir = Path(vector_store_dir or VECTOR_STORE_DIR)
    metadata_path = store_dir / "metadata.pkl"
    if not metadata_path.exists():
        build_and_save_index(dataset_dir=store_dir.parent.parent / "dataset", vector_store_dir=store_dir)

    with metadata_path.open("rb") as handle:
        return pickle.load(handle)


def answer_query(question: str, vector_store_dir: str | Path | None = None, top_k: int | None = None) -> str:
    metadata = load_index(vector_store_dir=vector_store_dir)
    embedding_model = EmbeddingModel(model_name=str(metadata.get("embedding_model_name", DEFAULT_EMBEDDING_MODEL)))
    vector_store = VectorStore(dimension=len(metadata["embeddings"][0]) if metadata.get("embeddings") else 1)
    vector_store.add_documents(metadata["chunks"], metadata["embeddings"])

    retrieved = retrieve_top_chunks(question, vector_store, embedding_model, top_k=top_k or int(metadata.get("top_k", DEFAULT_TOP_K)))
    context_chunks = [{"text": chunk.get("text", "")} for _, chunk in retrieved]
    llm = SimpleLLM()
    return llm.generate_answer(question, context_chunks)
