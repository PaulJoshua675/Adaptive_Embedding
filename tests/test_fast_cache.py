from __future__ import annotations

import math
import time

from src.cache import SemanticResponseCache, response_cache
from src.create_vector_db import VectorStore
from src.embedding_model import EmbeddingModel
from src.generator import AnswerGenerator
from src.retriever import retrieve_top_chunks


def test_semantic_response_cache_requires_similarity_above_threshold() -> None:
    cache = SemanticResponseCache(similarity_threshold=0.92)
    cache.set("original question", [1.0, 0.0], {"answer": "cached"}, "index-a")

    similar_embedding = [0.93, math.sqrt(1 - 0.93**2)]
    boundary_embedding = [0.92, math.sqrt(1 - 0.92**2)]

    assert cache.get("nearby question", similar_embedding, "index-a") == {"answer": "cached"}
    assert cache.get("boundary question", boundary_embedding, "index-a") is None
    assert cache.get("nearby question", similar_embedding, "index-b") is None


def test_fast_answer_generation_does_not_load_an_llm(monkeypatch) -> None:
    monkeypatch.setattr(
        "src.llm.SimpleLLM._get_generator",
        lambda self: (_ for _ in ()).throw(AssertionError("fast mode loaded an LLM")),
    )

    answer = AnswerGenerator().generate_fast(
        "What does RAG retrieve?",
        [{"text": "RAG retrieves passages to answer research questions."}],
    )

    assert "retrieves passages" in answer.lower()
    assert len([line for line in answer.splitlines() if line.startswith("### ")]) == 5


def test_legacy_retrieval_caps_faiss_candidates_at_five(monkeypatch) -> None:
    documents = [{"text": "retrieval systems use semantic embeddings"} for _ in range(8)]
    model = EmbeddingModel().fit([str(document["text"]) for document in documents])
    store = VectorStore(dimension=len(model.embed_text(documents[0]["text"])))
    store.add_documents(documents, model.embed_texts([str(document["text"]) for document in documents]))
    search = store.search
    requested_limits = []

    def tracked_search(query_embedding, top_k=5):
        requested_limits.append(top_k)
        return search(query_embedding, top_k)

    monkeypatch.setattr(store, "search", tracked_search)

    results = retrieve_top_chunks("Explain semantic retrieval systems", store, model, top_k=100)

    assert requested_limits == [5]
    assert len(results) <= 5


def test_exact_pipeline_cache_skips_classifier_and_index_loading(tmp_path, monkeypatch) -> None:
    from src.rag_pipeline import answer_query_details, build_and_save_index

    dataset = tmp_path / "dataset"
    dataset.mkdir()
    (dataset / "paper.txt").write_text(
        "RAG retrieves passages to answer research questions using external evidence.",
        encoding="utf-8",
    )
    index_dir = tmp_path / "vector_store"
    build_and_save_index(dataset, index_dir)
    response_cache.clear()
    query = "What does RAG retrieve to answer research questions?"
    first = answer_query_details(query, index_dir, fast_mode=True)

    def unexpected_query_path(*args, **kwargs):
        raise AssertionError("cache hit entered the query path")

    monkeypatch.setattr("src.rag_pipeline.route_query", unexpected_query_path)
    monkeypatch.setattr("src.rag_pipeline.load_cached_vector_store", unexpected_query_path)
    cache_started = time.perf_counter()
    second = answer_query_details(query, index_dir, fast_mode=True)
    cache_elapsed_ms = (time.perf_counter() - cache_started) * 1000

    assert first["answer"] == second["answer"]
    assert second["semantic_cache_hit"] is True
    assert cache_elapsed_ms < 10
