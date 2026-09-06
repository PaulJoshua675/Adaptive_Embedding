from __future__ import annotations

import pickle
from pathlib import Path
from typing import Dict, List

from config import DEFAULT_CHUNK_OVERLAP, DEFAULT_CHUNK_SIZE, DEFAULT_TOP_K, EMBEDDING_MODELS, VECTOR_STORE_DIR
from .chunk_documents import chunk_documents
from .embeddings import EmbeddingRegistry
from .embedding_model import EmbeddingModel
from .extract_pdf import extract_documents_from_path
from .generator import AnswerGenerator
from .llm import SimpleLLM
from .retrieval import (
    AdaptiveRetriever,
    find_matching_figures,
    inject_visual_payload,
    load_cached_embedding_registry,
    load_cached_vector_store,
    populate_chunk_map_from_store,
    populate_global_chunk_map,
    resolve_index_path,
    section_from_query,
)
from .selector import select_embedding_details
from .utils import ensure_directory, write_text
from .vector_db import FaissVectorStore, MultiVectorStore
from .preprocessing import extract_documents_with_pages
from .profiler import profile_stage


def build_index_from_documents(documents: List[Dict[str, object]]) -> Dict[str, object]:
    """Build a multi-embedding index without writing anything to disk."""
    chunks = chunk_documents(documents, chunk_size=DEFAULT_CHUNK_SIZE, overlap=DEFAULT_CHUNK_OVERLAP)
    registry = EmbeddingRegistry(EMBEDDING_MODELS)
    vectors = registry.encode_documents(chunks)
    stores = {
        key: FaissVectorStore(len(model_vectors[0]) if model_vectors else 1)
        for key, model_vectors in vectors.items()
    }
    for key, store in stores.items():
        store.add(chunks, vectors[key])
    populate_global_chunk_map(chunks)
    return {
        "chunks": chunks,
        "embedding_models": EMBEDDING_MODELS,
        "embedding_keys": list(EMBEDDING_MODELS),
        "top_k": DEFAULT_TOP_K,
        "stores": MultiVectorStore(stores),
    }


def save_index_from_documents(documents: List[Dict[str, object]], vector_store_dir: str | Path) -> Dict[str, object]:
    """Build and persist an isolated multi-embedding index for runtime documents."""
    store_dir = Path(vector_store_dir)
    store_dir.mkdir(parents=True, exist_ok=True)
    index = build_index_from_documents(documents)
    stores = index["stores"]
    if not isinstance(stores, MultiVectorStore):
        raise TypeError("The runtime index must contain a MultiVectorStore")
    populate_global_chunk_map(index["chunks"])
    stores.save(store_dir)
    metadata = {
        "chunks": index["chunks"],
        "embedding_models": index["embedding_models"],
        "embedding_keys": index["embedding_keys"],
        "top_k": index["top_k"],
    }
    with (store_dir / "metadata.pkl").open("wb") as handle:
        pickle.dump(metadata, handle)
    return {**metadata, "stores": stores}


def answer_query_from_index(
    question: str,
    index: Dict[str, object],
    top_k: int | None = None,
) -> Dict[str, object]:
    """Retrieve and generate an answer from a persisted or in-memory index."""
    stores = index["stores"]
    if not isinstance(stores, MultiVectorStore):
        raise TypeError("index['stores'] must be a MultiVectorStore")
    registry = load_cached_embedding_registry(tuple(sorted(dict(index["embedding_models"]).items())))
    query_type, embedding_key = select_embedding_details(question)
    retrieved = AdaptiveRetriever(stores, registry).retrieve(
        question,
        embedding_key=embedding_key,
        top_k=top_k or int(index.get("top_k", DEFAULT_TOP_K)),
        section_type=section_from_query(question),
    )
    result = {
        "answer": _synthesize_answer(question, retrieved),
        "chunks": retrieved,
        "query_type": query_type.value,
        "embedding_key": embedding_key,
        "figures": find_matching_figures(question),
    }
    return inject_visual_payload(result, question)


def build_and_save_index(dataset_dir: str | Path, vector_store_dir: str | Path | None = None) -> Dict[str, object]:
    dataset_path = Path(dataset_dir)
    store_dir = Path(vector_store_dir or VECTOR_STORE_DIR)
    ensure_directory(store_dir)

    figures_dir = store_dir / "figures"
    figures_dir.mkdir(parents=True, exist_ok=True)
    documents = extract_documents_with_pages(dataset_path, figures_dir)
    if not documents:
        documents = extract_documents_from_path(dataset_path)
    chunks = chunk_documents(documents, chunk_size=DEFAULT_CHUNK_SIZE, overlap=DEFAULT_CHUNK_OVERLAP)

    if documents:
        extracted_dir = store_dir.parent / "extracted_text"
        ensure_directory(extracted_dir)
        for document in documents:
            write_text(extracted_dir / f"{document['id']}.txt", str(document["text"]))

    registry = EmbeddingRegistry(EMBEDDING_MODELS)
    vectors = registry.encode_documents(chunks)
    stores = {
        key: FaissVectorStore(len(model_vectors[0]) if model_vectors else 1)
        for key, model_vectors in vectors.items()
    }
    for key, store in stores.items():
        store.add(chunks, vectors[key])
    multi_store = MultiVectorStore(stores)
    multi_store.save(store_dir)

    metadata = {
        "chunks": chunks,
        "embeddings": vectors.get("e5", []),
        "embedding_models": EMBEDDING_MODELS,
        "embedding_keys": list(EMBEDDING_MODELS),
        "chunk_size": DEFAULT_CHUNK_SIZE,
        "overlap": DEFAULT_CHUNK_OVERLAP,
        "top_k": DEFAULT_TOP_K,
    }

    metadata_path = store_dir / "metadata.pkl"
    with metadata_path.open("wb") as handle:
        pickle.dump(metadata, handle)

    return metadata


def load_index(vector_store_dir: str | Path | None = None) -> Dict[str, object]:
    store_dir = resolve_index_path(vector_store_dir or None) if vector_store_dir is None else Path(vector_store_dir)
    metadata_path = store_dir / "metadata.pkl"
    if not metadata_path.exists():
        build_and_save_index(dataset_dir=store_dir.parent / "dataset", vector_store_dir=store_dir)

    with metadata_path.open("rb") as handle:
        return pickle.load(handle)


def answer_query(question: str, vector_store_dir: str | Path | None = None, top_k: int | None = None) -> str:
    return answer_query_details(question, vector_store_dir, top_k)["answer"]


def answer_query_details(question: str, vector_store_dir: str | Path | None = None, top_k: int | None = None) -> Dict[str, object]:
    metadata = load_index(vector_store_dir=vector_store_dir)
    store_dir = resolve_index_path(vector_store_dir or None) if vector_store_dir is None else Path(vector_store_dir)
    if metadata.get("embedding_keys"):
        registry = load_cached_embedding_registry(tuple(sorted(dict(metadata["embedding_models"]).items())))
        stores = load_cached_vector_store(store_dir, list(metadata["embedding_keys"]))
        query_type, embedding_key = select_embedding_details(question)
        populate_chunk_map_from_store(stores, embedding_key)
        retrieved = AdaptiveRetriever(stores, registry).retrieve(
            question,
            embedding_key=embedding_key,
            top_k=top_k or int(metadata.get("top_k", DEFAULT_TOP_K)),
            section_type=section_from_query(question),
        )
        result = {
            "answer": _synthesize_answer(question, retrieved),
            "chunks": retrieved,
            "query_type": query_type.value,
            "embedding_key": embedding_key,
            "figures": find_matching_figures(question, store_dir / "figures" / "figure_manifest.json"),
        }
        return inject_visual_payload(result, question, store_dir / "figures" / "figure_manifest.json")

    # Keep indexes created by the original baseline readable.
    from .create_vector_db import VectorStore
    from .retriever import retrieve_top_chunks
    embedding_model = EmbeddingModel(model_name="simple-bow")
    embedding_model.vocabulary = dict(metadata.get("vocabulary") or {})
    embedding_model._built = bool(embedding_model.vocabulary)
    vector_store = VectorStore(dimension=len(metadata["embeddings"][0]) if metadata.get("embeddings") else 1)
    vector_store.add_documents(metadata["chunks"], metadata["embeddings"])
    retrieved = retrieve_top_chunks(question, vector_store, embedding_model, top_k=top_k or DEFAULT_TOP_K)
    chunks = [{"text": chunk.get("text", ""), **chunk} for _, chunk in retrieved]
    return {
        "answer": SimpleLLM().generate_answer(question, chunks),
        "chunks": chunks,
        "query_type": "legacy",
        "embedding_key": "simple-bow",
    }


def _synthesize_answer(question: str, retrieved: List[Dict[str, object]]) -> str:
    with profile_stage("LLM Synthesis"):
        return AnswerGenerator().generate(question, retrieved)
