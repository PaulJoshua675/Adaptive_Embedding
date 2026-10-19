from __future__ import annotations

import pickle
import re
import time
from functools import lru_cache
from pathlib import Path
from typing import Callable, Dict, Iterable, List

from config import DEFAULT_CHUNK_OVERLAP, DEFAULT_CHUNK_SIZE, DEFAULT_TOP_K, EMBEDDING_MODELS, VECTOR_STORE_DIR
from .chunk_documents import chunk_documents
from .classifier import classify_query
from .embeddings import EmbeddingRegistry
from .embedding_model import EmbeddingModel
from .extract_pdf import extract_documents_from_path
from .generator import AnswerGenerator
from .cache import response_cache
from .retrieval import (
    AdaptiveRetriever,
    inject_visual_payload,
    load_cached_embedding_registry,
    load_cached_vector_store,
    populate_chunk_map_from_store,
    populate_global_chunk_map,
    resolve_index_path,
    section_from_query,
)
from .multilingual import resolve_response_language
from .utils import ensure_directory, write_text
from .vector_db import FaissVectorStore, MultiVectorStore
from .preprocessing import extract_documents_with_pages, trim_query_tokens
from .profiler import profile_stage

try:
    import streamlit as st
except ImportError:  # pragma: no cover - optional runtime dependency
    st = None


def _cache_data(function):
    if st is None:
        return lru_cache(maxsize=16)(function)
    return st.cache_data(show_spinner=False)(function)


def _read_index_metadata(metadata_path: str, modified_ns: int, size: int) -> Dict[str, object]:
    del modified_ns, size
    with Path(metadata_path).open("rb") as handle:
        return pickle.load(handle)


def _cache_namespace(
    index_identity: str,
    response_language: str,
    top_k: int,
    fast_mode: bool,
    query: str,
    embedding_key: str | None = None,
) -> str:
    paper_match = re.search(r"(?<!\d)(\d{4}\.\d{4,5})(?:v\d+)?(?!\d)", query, re.I)
    section = section_from_query(query) or ""
    parts = [
        index_identity,
        response_language,
        str(top_k),
        str(fast_mode),
        paper_match.group(1) if paper_match else "",
        section,
    ]
    if embedding_key is not None:
        parts.append(embedding_key)
    return "|".join(parts)


def _cached_exact_response(
    query: str,
    namespace_prefix: str,
    stream_consumer: Callable[[Iterable[str]], str] | None,
) -> Dict[str, object] | None:
    if not _cacheable_query(query):
        return None
    lookup_started = time.perf_counter()
    cached = response_cache.get_exact(query, namespace_prefix)
    if cached is None:
        return None
    cached["semantic_cache_hit"] = True
    cached["performance_timings_ms"] = {
        "semantic_cache_lookup_ms": (time.perf_counter() - lookup_started) * 1000,
    }
    if stream_consumer is not None:
        stream_consumer(iter((str(cached.get("answer", "")),)))
    return cached


def _cacheable_query(query: str) -> bool:
    return not re.search(r"\b(?:figure|fig\.?|diagram|chart|plot|image|picture)\b", query, re.I)


def _cached_response(
    query: str,
    query_vector: List[float],
    namespace: str,
    stream_consumer: Callable[[Iterable[str]], str] | None,
) -> Dict[str, object] | None:
    if not _cacheable_query(query):
        return None
    lookup_started = time.perf_counter()
    cached = response_cache.get(query, query_vector, namespace)
    if cached is None:
        return None
    cached["semantic_cache_hit"] = True
    cached["performance_timings_ms"] = {
        "semantic_cache_lookup_ms": (time.perf_counter() - lookup_started) * 1000,
    }
    if stream_consumer is not None:
        stream_consumer(iter((str(cached.get("answer", "")),)))
    return cached


@_cache_data
def _read_compact_query_metadata(metadata_path: str, modified_ns: int, size: int) -> Dict[str, object]:
    del modified_ns, size
    with Path(metadata_path).open("rb") as handle:
        metadata = pickle.load(handle)
    if not metadata.get("embedding_keys"):
        return metadata
    return {
        "embedding_models": metadata.get("embedding_models", {}),
        "embedding_keys": metadata.get("embedding_keys", []),
        "top_k": metadata.get("top_k", DEFAULT_TOP_K),
    }


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
    response_language: str | None = None,
    fast_mode: bool = False,
) -> Dict[str, object]:
    """Retrieve and generate an answer from a persisted or in-memory index."""
    stores = index["stores"]
    if not isinstance(stores, MultiVectorStore):
        raise TypeError("index['stores'] must be a MultiVectorStore")
    timings: Dict[str, float] = {}
    language = resolve_response_language(question, response_language)
    result_limit = min(max(int(top_k if top_k is not None else DEFAULT_TOP_K), 3), 5)
    cache_identity = f"memory:{id(stores)}"
    exact_namespace = _cache_namespace(
        cache_identity,
        language,
        result_limit,
        fast_mode,
        question,
    )
    cached = _cached_exact_response(question, exact_namespace + "|", None)
    if cached is not None:
        return cached
    registry = load_cached_embedding_registry(tuple(sorted(dict(index["embedding_models"]).items())))
    stage_started = time.perf_counter()
    route = classify_query(question)
    timings["intent_classification_ms"] = (time.perf_counter() - stage_started) * 1000
    query_type = str(route["query_type"])
    embedding_key = str(route["embedding_key"])
    query_vector = registry[embedding_key].encode_query(trim_query_tokens(question))
    namespace = _cache_namespace(
        cache_identity,
        language,
        result_limit,
        fast_mode,
        question,
        embedding_key,
    )
    cached = _cached_response(question, query_vector, namespace, None)
    if cached is not None:
        cached.update(
            {
                "query_type": query_type,
                "embedding_key": embedding_key,
                "embedding_model": route.get("embedding_model", EMBEDDING_MODELS[embedding_key]),
                "route_confidence": float(route.get("confidence", 0.0)),
                "classifier_model": route.get("classifier_model", "unknown"),
                "response_language": language,
            }
        )
        return cached
    retriever = AdaptiveRetriever(stores, registry)
    retrieved = retriever.retrieve(
        question,
        embedding_key=embedding_key,
        top_k=result_limit,
        candidate_k=5,
        section_type=section_from_query(question),
        query_vector=query_vector,
    )
    timings.update(retriever.last_timings_ms)
    stage_started = time.perf_counter()
    answer = _synthesize_answer(question, retrieved, language, fast_mode=fast_mode)
    timings["answer_generation_ms"] = (time.perf_counter() - stage_started) * 1000
    result = {
        "answer": answer,
        "chunks": retrieved,
        "query_type": query_type,
        "embedding_key": embedding_key,
        "embedding_model": route.get("embedding_model", EMBEDDING_MODELS[embedding_key]),
        "route_confidence": float(route.get("confidence", 0.0)),
        "classifier_model": route.get("classifier_model", "unknown"),
        "response_language": language,
        "figures": [],
        "performance_timings_ms": timings,
    }
    result = inject_visual_payload(result, question)
    if _cacheable_query(question):
        response_cache.set(question, query_vector, result, namespace)
    return result


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
    stat = metadata_path.stat()
    return _read_index_metadata(str(metadata_path.resolve()), stat.st_mtime_ns, stat.st_size)


def answer_query(
    question: str,
    vector_store_dir: str | Path | None = None,
    top_k: int | None = None,
    response_language: str | None = None,
    fast_mode: bool = False,
) -> str:
    return answer_query_details(
        question,
        vector_store_dir,
        top_k,
        response_language,
        fast_mode=fast_mode,
    )["answer"]


def answer_query_details(
    question: str,
    vector_store_dir: str | Path | None = None,
    top_k: int | None = None,
    response_language: str | None = None,
    stream_consumer: Callable[[Iterable[str]], str] | None = None,
    fast_mode: bool = False,
) -> Dict[str, object]:
    store_dir = resolve_index_path(vector_store_dir or None) if vector_store_dir is None else Path(vector_store_dir)
    metadata_path = store_dir / "metadata.pkl"
    if not metadata_path.exists():
        build_and_save_index(dataset_dir=store_dir.parent / "dataset", vector_store_dir=store_dir)
    stat = metadata_path.stat()
    language = resolve_response_language(question, response_language)
    result_limit = min(max(int(top_k if top_k is not None else DEFAULT_TOP_K), 3), 5)
    index_identity = f"{metadata_path.resolve()}:{stat.st_mtime_ns}:{stat.st_size}"
    exact_namespace = _cache_namespace(
        index_identity,
        language,
        result_limit,
        fast_mode,
        question,
    )
    cached = _cached_exact_response(question, exact_namespace + "|", stream_consumer)
    if cached is not None:
        return cached
    metadata = _read_compact_query_metadata(
        str(metadata_path.resolve()),
        stat.st_mtime_ns,
        stat.st_size,
    )
    if metadata.get("embedding_keys"):
        registry = load_cached_embedding_registry(tuple(sorted(dict(metadata["embedding_models"]).items())))
        timings: Dict[str, float] = {}
        stage_started = time.perf_counter()
        route = classify_query(question)
        timings["intent_classification_ms"] = (time.perf_counter() - stage_started) * 1000
        query_type = str(route["query_type"])
        embedding_key = str(route["embedding_key"])
        query_vector = registry[embedding_key].encode_query(trim_query_tokens(question))
        namespace = _cache_namespace(
            index_identity,
            language,
            result_limit,
            fast_mode,
            question,
            embedding_key,
        )
        cache_started = time.perf_counter()
        cached = _cached_response(question, query_vector, namespace, stream_consumer)
        cache_lookup_ms = (time.perf_counter() - cache_started) * 1000
        if cached is not None:
            cached.update(
                {
                    "query_type": query_type,
                    "embedding_key": embedding_key,
                    "embedding_model": route.get("embedding_model", EMBEDDING_MODELS[embedding_key]),
                    "route_confidence": float(route.get("confidence", 0.0)),
                    "classifier_model": route.get("classifier_model", "unknown"),
                    "response_language": language,
                }
            )
            return cached
        timings["semantic_cache_lookup_ms"] = cache_lookup_ms
        stage_started = time.perf_counter()
        stores = load_cached_vector_store(store_dir, [embedding_key])
        index_load_ms = (time.perf_counter() - stage_started) * 1000
        populate_chunk_map_from_store(stores, embedding_key)
        retriever = AdaptiveRetriever(stores, registry)
        retrieved = retriever.retrieve(
            question,
            embedding_key=embedding_key,
            top_k=result_limit,
            candidate_k=5,
            section_type=section_from_query(question),
            query_vector=query_vector,
        )
        timings.update(retriever.last_timings_ms)
        timings["faiss_index_lookup_ms"] += index_load_ms
        stage_started = time.perf_counter()
        answer = _synthesize_answer(question, retrieved, language, stream_consumer, fast_mode)
        timings["answer_generation_ms"] = (time.perf_counter() - stage_started) * 1000
        result = {
            "answer": answer,
            "chunks": retrieved,
            "query_type": query_type,
            "embedding_key": embedding_key,
            "embedding_model": route.get("embedding_model", EMBEDDING_MODELS[embedding_key]),
            "route_confidence": float(route.get("confidence", 0.0)),
            "classifier_model": route.get("classifier_model", "unknown"),
            "response_language": language,
            "figures": [],
            "performance_timings_ms": timings,
        }
        result = inject_visual_payload(result, question, store_dir / "figures" / "figure_manifest.json")
        if _cacheable_query(question):
            response_cache.set(question, query_vector, result, namespace)
        return result

    # Keep indexes created by the original baseline readable.
    from .create_vector_db import VectorStore
    from .retriever import retrieve_top_chunks
    embedding_model = EmbeddingModel(model_name="simple-bow")
    embedding_model.vocabulary = dict(metadata.get("vocabulary") or {})
    embedding_model._built = bool(embedding_model.vocabulary)
    query_vector = embedding_model.embed_text(question)
    namespace = _cache_namespace(
        index_identity,
        language,
        result_limit,
        fast_mode,
        question,
        "simple-bow",
    )
    cached = _cached_response(question, query_vector, namespace, stream_consumer)
    if cached is not None:
        return cached
    vector_store = VectorStore(dimension=len(metadata["embeddings"][0]) if metadata.get("embeddings") else 1)
    vector_store.add_documents(metadata["chunks"], metadata["embeddings"])
    retrieved = retrieve_top_chunks(
        question,
        vector_store,
        embedding_model,
        top_k=result_limit,
        query_vector=query_vector,
    )
    chunks = [{"text": chunk.get("text", ""), **chunk, "score": score} for score, chunk in retrieved]
    answer = _synthesize_answer(question, chunks, language, stream_consumer, fast_mode)
    result = {
        "answer": answer,
        "chunks": chunks,
        "query_type": "legacy",
        "embedding_key": "simple-bow",
        "response_language": language,
    }
    if _cacheable_query(question):
        response_cache.set(question, query_vector, result, namespace)
    return result


def _synthesize_answer(
    question: str,
    retrieved: List[Dict[str, object]],
    response_language: str = "English",
    stream_consumer: Callable[[Iterable[str]], str] | None = None,
    fast_mode: bool = False,
) -> str:
    with profile_stage("LLM Synthesis"):
        generator = AnswerGenerator()
        if fast_mode:
            return generator.generate_fast(question, retrieved, response_language=response_language)
        if stream_consumer is not None:
            return generator.generate_stream(
                question,
                retrieved,
                stream_consumer=stream_consumer,
                response_language=response_language,
            )
        return generator.generate(question, retrieved, response_language=response_language)
