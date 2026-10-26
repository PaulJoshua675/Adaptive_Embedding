from __future__ import annotations

import hashlib
import pickle
import re
import time
from difflib import SequenceMatcher
from functools import lru_cache
from pathlib import Path
from typing import Callable, Dict, Iterable, List

from config import (
    DEFAULT_CHUNK_OVERLAP,
    DEFAULT_CHUNK_SIZE,
    DEFAULT_TOP_K,
    EMBEDDING_MODELS,
    MIN_RETRIEVAL_SIMILARITY,
    VECTOR_STORE_DIR,
)
from .chunk_documents import chunk_documents
from .embeddings import EmbeddingRegistry
from .embedding_model import EmbeddingModel
from .extract_pdf import extract_documents_from_path
from .generator import AnswerGenerator, format_sources
from .ingestion import ingest_corpus
from .cache import response_cache
from .router import route_query
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
from .preprocessing import trim_query_tokens
from .profiler import profile_stage

try:
    import streamlit as st
except ImportError:  # pragma: no cover - optional runtime dependency
    st = None


def _cache_data(function):
    if st is None:
        return lru_cache(maxsize=512)(function)
    return st.cache_data(show_spinner=False, max_entries=512)(function)


@lru_cache(maxsize=512)
def _cached_final_response(
    query_key: str,
    response_identity: str,
    _question: str,
    _serialized_passages: bytes,
    response_language: str,
    fallback: bool,
    fast_mode: bool,
) -> str:
    del query_key, response_identity
    retrieved = pickle.loads(_serialized_passages)
    if not isinstance(retrieved, list):
        raise TypeError("Cached RAG passages must deserialize to a list")
    return _synthesize_answer(
        _question,
        retrieved,
        response_language,
        fast_mode=fast_mode,
        fallback=fallback,
    )


def _synthesize_response(
    question: str,
    retrieved: List[Dict[str, object]],
    response_language: str,
    fast_mode: bool,
    fallback: bool,
    stream_consumer: Callable[[Iterable[str]], str] | None = None,
    cache_identity: str | None = None,
) -> Dict[str, object]:
    if stream_consumer is not None or not _cacheable_query(question):
        answer = _synthesize_answer(
            question,
            retrieved,
            response_language,
            stream_consumer,
            fast_mode,
            fallback,
        )
        return {
            "answer": answer,
            "chunks": retrieved,
            "provenance": format_sources(retrieved),
        }

    response_identity = cache_identity
    if response_identity is None:
        passages_hash = hashlib.sha256()
        for passage in retrieved:
            passage_text = str(passage.get("text", ""))
            passage_id = passage.get(
                "chunk_id",
                passage.get("id", passage.get("document_id", passage.get("source", ""))),
            )
            passage_fingerprint = (
                str(passage_id),
                hashlib.sha256(passage_text.encode("utf-8")).hexdigest(),
                str(passage.get("source", "")),
                str(passage.get("page_number", "")),
                str(passage.get("section_type", "")),
            )
            passages_hash.update(pickle.dumps(passage_fingerprint, protocol=pickle.HIGHEST_PROTOCOL))
        response_identity = passages_hash.hexdigest()

    answer = _cached_final_response(
        question.strip().lower(),
        response_identity,
        question,
        pickle.dumps(retrieved, protocol=pickle.HIGHEST_PROTOCOL),
        response_language,
        fallback,
        fast_mode,
    )
    return {
        "answer": answer,
        "chunks": retrieved,
        "provenance": format_sources(retrieved),
    }


def _classify_query_timed(question: str) -> tuple[Dict[str, object], float]:
    started = time.perf_counter()
    with profile_stage("Intent Classification"):
        route = route_query(question)
    return route, (time.perf_counter() - started) * 1000


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
    document_identifier: str | None = None,
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
    if document_identifier:
        parts.append(document_identifier)
    if embedding_key is not None:
        parts.append(embedding_key)
    return "|".join(parts)


def _complete_response_payload(result: Dict[str, object]) -> Dict[str, object]:
    passages = result.get("retrieved_passages", result.get("chunks", []))
    if not isinstance(passages, list):
        raise TypeError("RAG response passages must be a list")
    result["chunks"] = passages
    result["retrieved_passages"] = passages

    metrics = result.get("eval_metrics", result.get("evaluation_metrics"))
    if not isinstance(metrics, dict):
        from .evaluation_suite import retrieval_metrics

        retrieved_ids = [
            passage.get("id", passage.get("document_id", index))
            if isinstance(passage, dict)
            else index
            for index, passage in enumerate(passages)
        ]
        metrics = retrieval_metrics(retrieved_ids, retrieved_ids[:1], k=3)
    result["eval_metrics"] = dict(metrics)
    result["evaluation_metrics"] = dict(metrics)

    timings = result.get("profile_times", result.get("performance_timings_ms", {}))
    if not isinstance(timings, dict):
        raise TypeError("RAG response profile times must be a mapping")
    result["profile_times"] = dict(timings)
    result["performance_timings_ms"] = dict(timings)
    result.setdefault("citations", result.get("provenance", ""))
    return result


def _cached_exact_response(
    query: str,
    namespace_prefix: str,
    stream_consumer: Callable[[Iterable[str]], str] | None,
) -> Dict[str, object] | None:
    if not _cacheable_query(query):
        return None
    cached = _lookup_exact_response(query.strip().lower(), namespace_prefix, response_cache.generation)
    if cached is None:
        return None
    cached = _complete_response_payload(dict(cached))
    cached["semantic_cache_hit"] = True
    if stream_consumer is not None:
        stream_consumer(iter((str(cached.get("answer", "")),)))
    return cached


@lru_cache(maxsize=500)
def _lookup_exact_response(
    query: str,
    namespace_prefix: str,
    cache_generation: int,
) -> Dict[str, object] | None:
    del cache_generation
    return response_cache.get_exact(query, namespace_prefix)


def _cacheable_query(query: str) -> bool:
    return not re.search(
        r"\b(?:figure|fig\.?|diagram|chart|plot|image|picture|dataset|benchmark|results?|trend|accuracy|scores?)\b",
        query,
        re.I,
    )


def _cached_response(
    query: str,
    query_vector: List[float],
    namespace: str,
    stream_consumer: Callable[[Iterable[str]], str] | None,
) -> Dict[str, object] | None:
    if not _cacheable_query(query):
        return None
    cached = response_cache.get(query, query_vector, namespace)
    if cached is None:
        return None
    cached = _complete_response_payload(cached)
    cached["semantic_cache_hit"] = True
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


def _retrieve_with_fallback(
    retriever: AdaptiveRetriever,
    question: str,
    embedding_key: str,
    top_k: int,
    section_type: str | None,
    query_vector: List[float],
) -> tuple[List[Dict[str, object]], bool]:
    retrieved = retriever.retrieve(
        question,
        embedding_key=embedding_key,
        top_k=top_k,
        candidate_k=3,
        section_type=section_type,
        query_vector=query_vector,
    )
    retrieved = _merge_visual_metadata_matches(
        retriever,
        question,
        embedding_key,
        retrieved,
        top_k,
    )
    best_score = max((float(chunk.get("score", -1.0)) for chunk in retrieved), default=-1.0)
    if retrieved and best_score >= MIN_RETRIEVAL_SIMILARITY:
        return retrieved, False
    fallback_chunks = retriever.retrieve(
        question,
        embedding_key=embedding_key,
        top_k=top_k,
        candidate_k=5,
        section_type=section_type,
        query_vector=query_vector,
        include_low_relevance=True,
    )
    fallback_chunks = _merge_visual_metadata_matches(
        retriever,
        question,
        embedding_key,
        fallback_chunks,
        top_k,
    )
    fallback_score = max(
        (float(chunk.get("score", -1.0)) for chunk in fallback_chunks),
        default=-1.0,
    )
    return fallback_chunks, fallback_score < MIN_RETRIEVAL_SIMILARITY


_VISUAL_QUERY_STOPWORDS = {
    "a", "an", "and", "are", "by", "for", "from", "in", "is", "of", "on",
    "show", "the", "to", "what", "which", "with", "figure", "fig", "image",
    "diagram", "chart", "plot", "graph", "results", "result",
}
_VISUAL_METADATA_THRESHOLD = 0.4


def _visual_metadata_score(query: str, document: Dict[str, object]) -> float:
    visual_type = str(document.get("visual_type", "")).casefold()
    if visual_type not in {"figure", "image", "diagram", "chart", "plot"}:
        return 0.0
    query_tokens = set(re.findall(r"[a-z0-9]+", query.casefold())) - _VISUAL_QUERY_STOPWORDS
    if not query_tokens:
        return 0.0
    descriptions: List[str] = []
    for key in ("caption", "alt_text", "alt-text", "altText", "alt", "image_alt_text", "description"):
        value = document.get(key)
        if isinstance(value, str) and value.strip():
            descriptions.append(value)
    for key in ("figures", "images"):
        records = document.get(key)
        if not isinstance(records, list):
            continue
        for record in records:
            if not isinstance(record, dict):
                continue
            for field in ("caption", "alt_text", "alt-text", "altText", "alt", "image_alt_text", "description"):
                value = record.get(field)
                if isinstance(value, str) and value.strip():
                    descriptions.append(value)
    if not descriptions:
        return 0.0

    query_text = " ".join(sorted(query_tokens))
    best_score = 0.0
    for description in descriptions:
        description_tokens = set(re.findall(r"[a-z0-9]+", description.casefold()))
        overlap_score = len(query_tokens & description_tokens) / len(query_tokens)
        sequence_score = SequenceMatcher(
            None,
            query_text,
            " ".join(sorted(description_tokens)),
        ).ratio()
        best_score = max(best_score, overlap_score, sequence_score)
    return best_score


def _merge_visual_metadata_matches(
    retriever: AdaptiveRetriever,
    question: str,
    embedding_key: str,
    retrieved: List[Dict[str, object]],
    top_k: int,
) -> List[Dict[str, object]]:
    stores = getattr(retriever, "stores", None)
    indexed_stores = getattr(stores, "stores", None)
    store = indexed_stores.get(embedding_key) if isinstance(indexed_stores, dict) else None
    documents = getattr(store, "documents", None)
    if not isinstance(documents, list):
        return retrieved

    def chunk_key(chunk: Dict[str, object], fallback: str) -> str:
        chunk_id = chunk.get("chunk_id")
        if chunk_id is not None:
            return f"chunk:{chunk_id}"
        document_id = chunk.get("document_id")
        if document_id is not None:
            return ":".join(
                (
                    str(document_id),
                    str(chunk.get("visual_type", "")),
                    str(chunk.get("start", "")),
                    str(chunk.get("end", "")),
                )
            )
        return fallback

    merged: Dict[str, Dict[str, object]] = {}
    for index, chunk in enumerate(retrieved):
        key = chunk_key(chunk, f"retrieved:{index}")
        merged[key] = dict(chunk)
    for index, document in enumerate(documents):
        if not isinstance(document, dict):
            continue
        score = _visual_metadata_score(question, document)
        if score < _VISUAL_METADATA_THRESHOLD:
            continue
        key = chunk_key(document, f"indexed:{index}")
        chunk = dict(merged.get(key, document))
        chunk["score"] = max(float(chunk.get("score", 0.0)), score)
        chunk.setdefault("text", str(document.get("text", "")))
        merged[key] = chunk
    return sorted(
        merged.values(),
        key=lambda chunk: float(chunk.get("score", 0.0)),
        reverse=True,
    )[: max(1, int(top_k))]


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
    pipeline_started = time.perf_counter()
    timings: Dict[str, float] = {}
    route, route_time_ms = _classify_query_timed(question)
    if route["route"] == "direct_response":
        return _direct_response_result(
            str(route["direct_kind"]),
            language,
            route_time_ms,
        )
    registry = load_cached_embedding_registry(tuple(sorted(dict(index["embedding_models"]).items())))
    timings["intent_classification_ms"] = route_time_ms
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
                "route": str(route["route"]),
                "embedding_key": embedding_key,
                "embedding_model": route.get("embedding_model", EMBEDDING_MODELS[embedding_key]),
                "route_confidence": float(route.get("confidence", 0.0)),
                "classifier_model": route.get("classifier_model", "unknown"),
                "response_language": language,
            }
        )
        return cached
    retriever = AdaptiveRetriever(stores, registry)
    retrieved, fallback = _retrieve_with_fallback(
        retriever,
        question,
        embedding_key,
        result_limit,
        section_from_query(question),
        query_vector,
    )
    timings.update(retriever.last_timings_ms)
    stage_started = time.perf_counter()
    answer = _synthesize_answer(question, retrieved, language, fast_mode=fast_mode, fallback=fallback)
    timings["answer_generation_ms"] = (time.perf_counter() - stage_started) * 1000
    result = {
        "answer": answer,
        "chunks": retrieved,
        "query_type": query_type,
        "route": str(route["route"]),
        "embedding_key": embedding_key,
        "embedding_model": route.get("embedding_model", EMBEDDING_MODELS[embedding_key]),
        "route_confidence": float(route.get("confidence", 0.0)),
        "classifier_model": route.get("classifier_model", "unknown"),
        "response_language": language,
        "fallback": fallback,
        "similarity_score": max((float(chunk.get("score", -1.0)) for chunk in retrieved), default=-1.0),
        "relevance_threshold": MIN_RETRIEVAL_SIMILARITY,
        "figures": [],
        "performance_timings_ms": timings,
    }
    result = inject_visual_payload(result, question)
    timings["pipeline_round_trip_ms"] = (time.perf_counter() - pipeline_started) * 1000
    result["performance_timings_ms"] = timings
    _complete_response_payload(result)
    if _cacheable_query(question):
        response_cache.set(question, query_vector, result, namespace)
    return result


def build_and_save_index(dataset_dir: str | Path, vector_store_dir: str | Path | None = None) -> Dict[str, object]:
    dataset_path = Path(dataset_dir)
    store_dir = Path(vector_store_dir or VECTOR_STORE_DIR)
    ensure_directory(store_dir)

    figures_dir = store_dir / "figures"
    figures_dir.mkdir(parents=True, exist_ok=True)
    documents = ingest_corpus(dataset_path, figures_dir)
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
    document_identifier: str | None = None,
) -> str:
    return answer_query_details(
        question,
        vector_store_dir,
        top_k,
        response_language,
        fast_mode=fast_mode,
        document_identifier=document_identifier,
    )["answer"]


def get_cached_query_response(
    question: str,
    vector_store_dir: str | Path | None = None,
    top_k: int | None = None,
    response_language: str | None = None,
    stream_consumer: Callable[[Iterable[str]], str] | None = None,
    fast_mode: bool = False,
    document_identifier: str | None = None,
) -> Dict[str, object] | None:
    """Return an exact cached response without loading query or index models."""
    language = resolve_response_language(question, response_language)
    store_dir = resolve_index_path(vector_store_dir or None) if vector_store_dir is None else Path(vector_store_dir)
    metadata_path = store_dir / "metadata.pkl"
    if not metadata_path.exists():
        return None

    stat = metadata_path.stat()
    index_identity = f"{metadata_path.resolve()}:{stat.st_mtime_ns}:{stat.st_size}"
    result_limit = min(max(int(top_k if top_k is not None else DEFAULT_TOP_K), 3), 5)
    exact_namespace = _cache_namespace(
        index_identity,
        language,
        result_limit,
        fast_mode,
        question,
        document_identifier=document_identifier,
    )
    return _cached_exact_response(question, exact_namespace + "|", stream_consumer)


def answer_query_details(
    question: str,
    vector_store_dir: str | Path | None = None,
    top_k: int | None = None,
    response_language: str | None = None,
    stream_consumer: Callable[[Iterable[str]], str] | None = None,
    fast_mode: bool = False,
    document_identifier: str | None = None,
) -> Dict[str, object]:
    cached = get_cached_query_response(
        question,
        vector_store_dir,
        top_k,
        response_language,
        stream_consumer,
        fast_mode,
        document_identifier,
    )
    if cached is not None:
        return cached

    pipeline_started = time.perf_counter()
    language = resolve_response_language(question, response_language)
    store_dir = resolve_index_path(vector_store_dir or None) if vector_store_dir is None else Path(vector_store_dir)
    metadata_path = store_dir / "metadata.pkl"
    result_limit = min(max(int(top_k if top_k is not None else DEFAULT_TOP_K), 3), 5)
    stat = metadata_path.stat() if metadata_path.exists() else None
    if stat is not None:
        index_identity = f"{metadata_path.resolve()}:{stat.st_mtime_ns}:{stat.st_size}"

    route, route_time_ms = _classify_query_timed(question)
    if route["route"] == "direct_response":
        return _direct_response_result(
            str(route["direct_kind"]),
            language,
            route_time_ms,
            stream_consumer,
        )

    if stat is None:
        build_and_save_index(dataset_dir=store_dir.parent / "dataset", vector_store_dir=store_dir)
        stat = metadata_path.stat()
        index_identity = f"{metadata_path.resolve()}:{stat.st_mtime_ns}:{stat.st_size}"

    metadata = _read_compact_query_metadata(
        str(metadata_path.resolve()),
        stat.st_mtime_ns,
        stat.st_size,
    )
    if metadata.get("embedding_keys"):
        registry = load_cached_embedding_registry(tuple(sorted(dict(metadata["embedding_models"]).items())))
        timings: Dict[str, float] = {"intent_classification_ms": route_time_ms}
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
            document_identifier,
        )
        cached = _cached_response(question, query_vector, namespace, stream_consumer)
        if cached is not None:
            cached.update(
                {
                    "query_type": query_type,
                    "route": str(route["route"]),
                    "embedding_key": embedding_key,
                    "embedding_model": route.get("embedding_model", EMBEDDING_MODELS[embedding_key]),
                    "route_confidence": float(route.get("confidence", 0.0)),
                    "classifier_model": route.get("classifier_model", "unknown"),
                    "response_language": language,
                }
            )
            return cached
        stage_started = time.perf_counter()
        stores = load_cached_vector_store(store_dir, [embedding_key])
        index_load_ms = (time.perf_counter() - stage_started) * 1000
        populate_chunk_map_from_store(stores, embedding_key)
        retriever = AdaptiveRetriever(stores, registry)
        retrieved, fallback = _retrieve_with_fallback(
            retriever,
            question,
            embedding_key,
            result_limit,
            section_from_query(question),
            query_vector,
        )
        timings.update(retriever.last_timings_ms)
        timings["faiss_index_lookup_ms"] += index_load_ms
        stage_started = time.perf_counter()
        synthesis = _synthesize_response(
            question,
            retrieved,
            language,
            fast_mode,
            fallback,
            stream_consumer,
            f"{index_identity}|document={document_identifier or ''}|top_k={result_limit}",
        )
        answer = str(synthesis["answer"])
        retrieved = synthesis["chunks"]
        timings["answer_generation_ms"] = (time.perf_counter() - stage_started) * 1000
        result = {
            "answer": answer,
            "chunks": retrieved,
            "provenance": synthesis["provenance"],
            "query_type": query_type,
            "route": str(route["route"]),
            "embedding_key": embedding_key,
            "embedding_model": route.get("embedding_model", EMBEDDING_MODELS[embedding_key]),
            "route_confidence": float(route.get("confidence", 0.0)),
            "classifier_model": route.get("classifier_model", "unknown"),
            "response_language": language,
            "fallback": fallback,
            "similarity_score": max((float(chunk.get("score", -1.0)) for chunk in retrieved), default=-1.0),
            "relevance_threshold": MIN_RETRIEVAL_SIMILARITY,
            "figures": [],
            "performance_timings_ms": timings,
        }
        result = inject_visual_payload(result, question, store_dir / "figures" / "figure_manifest.json")
        timings["pipeline_round_trip_ms"] = (time.perf_counter() - pipeline_started) * 1000
        result["performance_timings_ms"] = timings
        _complete_response_payload(result)
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
        document_identifier,
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
    fallback = not retrieved
    if fallback:
        retrieved = retrieve_top_chunks(
            question,
            vector_store,
            embedding_model,
            top_k=result_limit,
            query_vector=query_vector,
            include_low_relevance=True,
        )
    chunks = [{"text": chunk.get("text", ""), **chunk, "score": score} for score, chunk in retrieved]
    synthesis = _synthesize_response(
        question,
        chunks,
        language,
        fast_mode,
        fallback,
        stream_consumer,
        f"{index_identity}|document={document_identifier or ''}|top_k={result_limit}",
    )
    answer = str(synthesis["answer"])
    chunks = synthesis["chunks"]
    result = {
        "answer": answer,
        "chunks": chunks,
        "provenance": synthesis["provenance"],
        "query_type": "legacy",
        "route": str(route["route"]),
        "embedding_key": "simple-bow",
        "response_language": language,
        "fallback": fallback,
        "similarity_score": max((float(chunk.get("score", -1.0)) for chunk in chunks), default=-1.0),
        "relevance_threshold": MIN_RETRIEVAL_SIMILARITY,
        "figures": [],
    }
    result = inject_visual_payload(result, question, store_dir / "figures" / "figure_manifest.json")
    result["performance_timings_ms"] = {
        "pipeline_round_trip_ms": (time.perf_counter() - pipeline_started) * 1000,
    }
    _complete_response_payload(result)
    if _cacheable_query(question):
        response_cache.set(question, query_vector, result, namespace)
    return result


def _direct_response_result(
    direct_kind: str,
    response_language: str,
    route_time_ms: float,
    stream_consumer: Callable[[Iterable[str]], str] | None = None,
) -> Dict[str, object]:
    responses = {
        "English": {
            "greeting": "Hello! Ask me a question about the indexed research documents.",
            "thanks": "You're welcome! Ask me about any indexed research document.",
            "help": "I can answer questions using the indexed research documents.",
        },
        "Tamil": {
            "greeting": "வணக்கம்! குறியிடப்பட்ட ஆய்வு ஆவணங்களைப் பற்றி கேள்வி கேளுங்கள்.",
            "thanks": "வரவேற்கிறேன்! குறியிடப்பட்ட ஆய்வு ஆவணங்களைப் பற்றி கேளுங்கள்.",
            "help": "குறியிடப்பட்ட ஆய்வு ஆவணங்களைப் பற்றிய கேள்விகளுக்கு என்னால் பதிலளிக்க முடியும்.",
        },
        "Hindi": {
            "greeting": "नमस्ते! अनुक्रमित शोध दस्तावेज़ों के बारे में प्रश्न पूछें।",
            "thanks": "आपका स्वागत है! किसी भी अनुक्रमित शोध दस्तावेज़ के बारे में पूछें।",
            "help": "मैं अनुक्रमित शोध दस्तावेज़ों के बारे में प्रश्नों के उत्तर दे सकता हूँ।",
        },
        "Spanish": {
            "greeting": "¡Hola! Pregunta sobre los documentos de investigación indexados.",
            "thanks": "¡De nada! Pregunta sobre cualquier documento de investigación indexado.",
            "help": "Puedo responder preguntas usando los documentos de investigación indexados.",
        },
        "French": {
            "greeting": "Bonjour ! Posez une question sur les documents de recherche indexés.",
            "thanks": "Avec plaisir ! Posez une question sur les documents de recherche indexés.",
            "help": "Je peux répondre aux questions sur les documents de recherche indexés.",
        },
    }
    language_responses = responses.get(response_language, responses["English"])
    answer = language_responses.get(direct_kind, language_responses["greeting"])
    if stream_consumer is not None:
        stream_consumer(iter((answer,)))
    return {
        "answer": answer,
        "chunks": [],
        "query_type": "direct_response",
        "route": "direct_response",
        "embedding_key": "none",
        "embedding_model": "none",
        "route_confidence": 1.0,
        "classifier_model": "compiled-regex",
        "response_language": response_language,
        "figures": [],
        "performance_timings_ms": {
            "intent_classification_ms": route_time_ms,
            "faiss_index_lookup_ms": 0.0,
            "answer_generation_ms": 0.0,
        },
    }


# Keep the former module-level entry point available without invoking a model.
classify_query = route_query


def _synthesize_answer(
    question: str,
    retrieved: List[Dict[str, object]],
    response_language: str = "English",
    stream_consumer: Callable[[Iterable[str]], str] | None = None,
    fast_mode: bool = False,
    fallback: bool = False,
) -> str:
    with profile_stage("LLM Synthesis"):
        generator = AnswerGenerator()
        if fast_mode:
            return generator.generate_fast(
                question,
                retrieved,
                response_language=response_language,
                fallback=fallback,
            )
        if stream_consumer is not None:
            return generator.generate_stream(
                question,
                retrieved,
                stream_consumer=stream_consumer,
                response_language=response_language,
                fallback=fallback,
            )
        return generator.generate(
            question,
            retrieved,
            response_language=response_language,
            fallback=fallback,
        )
