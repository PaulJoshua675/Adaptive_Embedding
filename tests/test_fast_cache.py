from __future__ import annotations

import math
import time
from types import SimpleNamespace

from src.cache import SemanticResponseCache, response_cache
from src.create_vector_db import VectorStore
from src.embeddings import EmbeddingEncoder, EmbeddingRegistry
from src.embedding_model import EmbeddingModel
from src.generator import AnswerGenerator
from src.retriever import retrieve_top_chunks
from src.retrieval import AdaptiveRetriever
from src.vector_db import MultiVectorStore


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


def test_query_embedding_is_cached(monkeypatch) -> None:
    calls = []
    original = EmbeddingEncoder._encode_query_uncached

    def tracked_encode(encoder, query):
        calls.append(query)
        return original(encoder, query)

    monkeypatch.setattr(EmbeddingEncoder, "_encode_query_uncached", tracked_encode)
    encoder = EmbeddingEncoder("cache-test", "cache-test-model")
    query = "unique cached embedding query"

    first = encoder.encode_query(query)
    second = encoder.encode_query(query)

    assert first == second
    assert calls == [query]


def test_enabled_cross_encoder_runs_even_for_confident_faiss_results(monkeypatch) -> None:
    monkeypatch.setenv("RAG_ENABLE_CROSS_ENCODER_RERANKER", "1")
    passages = [
        {"text": f"retrieval evidence passage {index}", "chunk_id": index}
        for index in range(3)
    ]
    store = SimpleNamespace(
        documents=passages,
        embeddings=[],
        search=lambda query, limit: [(0.9, passage) for passage in passages[:limit]],
    )
    retriever = AdaptiveRetriever(
        MultiVectorStore({"bge": store}),
        EmbeddingRegistry({"bge": "cache-test-model"}),
    )
    calls = []

    class RecordingReranker:
        def rerank(self, query, candidates, top_k, candidate_k):
            calls.append((query, len(candidates)))
            time.sleep(0.001)
            return candidates[:top_k]

    retriever.reranker = RecordingReranker()

    retrieved = retriever.retrieve(
        "retrieval evidence",
        embedding_key="bge",
        top_k=3,
        candidate_k=3,
        query_vector=[1.0],
    )

    assert len(retrieved) == 3
    assert calls == [("retrieval evidence", 3)]
    assert retriever.last_timings_ms["cross_encoder_reranking_ms"] >= 1.0


def test_cross_encoder_timing_covers_rerank_execution(monkeypatch) -> None:
    monkeypatch.setenv("RAG_ENABLE_CROSS_ENCODER_RERANKER", "1")
    passages = [
        {"text": f"retrieval evidence passage {index}", "chunk_id": index}
        for index in range(3)
    ]
    store = SimpleNamespace(
        documents=passages,
        embeddings=[],
        search=lambda query, limit: [(0.5, passage) for passage in passages[:limit]],
    )
    retriever = AdaptiveRetriever(
        MultiVectorStore({"bge": store}),
        EmbeddingRegistry({"bge": "cache-test-model"}),
    )

    class SlowReranker:
        def rerank(self, query, candidates, top_k, candidate_k):
            time.sleep(0.005)
            return candidates[:top_k]

    retriever.reranker = SlowReranker()
    retrieved = retriever.retrieve(
        "retrieval evidence",
        embedding_key="bge",
        top_k=3,
        candidate_k=3,
        query_vector=[1.0],
    )

    assert len(retrieved) == 3
    assert retriever.last_timings_ms["cross_encoder_reranking_ms"] >= 5.0


def test_intent_classification_timer_wraps_classifier(monkeypatch) -> None:
    from src import rag_pipeline

    def slow_classifier(query):
        time.sleep(0.005)
        return {"route": "rag_retrieval"}

    monkeypatch.setattr(rag_pipeline, "route_query", slow_classifier)

    route, elapsed_ms = rag_pipeline._classify_query_timed("timing test")

    assert route["route"] == "rag_retrieval"
    assert elapsed_ms >= 5.0


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
    monkeypatch.setattr("src.rag_pipeline.time", SimpleNamespace(perf_counter=unexpected_query_path))
    second = answer_query_details(query, index_dir, fast_mode=True)
    cache_elapsed_ms = (time.perf_counter() - cache_started) * 1000

    assert first["answer"] == second["answer"]
    assert second["semantic_cache_hit"] is True
    assert second["retrieved_passages"] == second["chunks"]
    assert second["eval_metrics"] == second["evaluation_metrics"]
    assert {"accuracy", "precision", "recall", "f1"} <= second["eval_metrics"].keys()
    assert second["profile_times"] == second["performance_timings_ms"]
    assert "pipeline_round_trip_ms" in second["profile_times"]
    assert cache_elapsed_ms < 175


def test_custom_document_cache_is_scoped_to_document_identifier(tmp_path, monkeypatch) -> None:
    from src.rag_pipeline import (
        answer_query_details,
        build_and_save_index,
        get_cached_query_response,
    )

    dataset = tmp_path / "custom_dataset"
    dataset.mkdir()
    (dataset / "paper.txt").write_text(
        "A unique custom paper explains adaptive retrieval with contextual evidence.",
        encoding="utf-8",
    )
    index_dir = tmp_path / "custom_vector_store"
    build_and_save_index(dataset, index_dir)
    response_cache.clear()
    query = "How does this method use contextual evidence?"
    first = answer_query_details(
        query,
        index_dir,
        fast_mode=True,
        document_identifier="paper-sha256-a",
    )

    def unexpected_query_path(*args, **kwargs):
        raise AssertionError("custom-document cache hit entered the retrieval path")

    monkeypatch.setattr("src.rag_pipeline.route_query", unexpected_query_path)
    monkeypatch.setattr("src.rag_pipeline.load_cached_vector_store", unexpected_query_path)
    matching_document = answer_query_details(
        query,
        index_dir,
        fast_mode=True,
        document_identifier="paper-sha256-a",
    )
    different_document = get_cached_query_response(
        query,
        index_dir,
        fast_mode=True,
        document_identifier="paper-sha256-b",
    )

    assert matching_document is not None
    assert matching_document["semantic_cache_hit"] is True
    assert matching_document["answer"] == first["answer"]
    assert matching_document["provenance"] == first["provenance"]
    assert matching_document["citations"] == first["citations"]
    assert matching_document["eval_metrics"] == first["eval_metrics"]
    assert matching_document["profile_times"] == first["profile_times"]
    assert different_document is None


def test_answer_generation_cache_uses_query_and_chunk_content(monkeypatch) -> None:
    from src import generator as generator_module

    generator_module._answer_cache.clear()
    monkeypatch.setattr(generator_module, "_ANSWER_CACHE_TTL_SECONDS", 0.01)
    calls = []

    def generate_answer(self, query, chunks, response_language, fallback):
        calls.append((query, tuple(chunk["text"] for chunk in chunks)))
        return f"answer {len(calls)}"

    monkeypatch.setattr(AnswerGenerator, "_generate_uncached", generate_answer)
    generator = AnswerGenerator()
    query = "What does this unique cached test ask?"
    chunks = [{"text": "Unique cached evidence.", "source": "paper-a"}]

    first = generator.generate(query, chunks)
    cached_started = time.perf_counter()
    repeated = generator.generate(query, chunks)
    cached_elapsed_ms = (time.perf_counter() - cached_started) * 1000
    changed_evidence = generator.generate(query, [{"text": "Changed evidence.", "source": "paper-a"}])
    time.sleep(0.02)
    expired = generator.generate(query, chunks)

    assert first == repeated == "answer 1"
    assert changed_evidence == "answer 2"
    assert expired == "answer 3"
    assert len(calls) == 3
    assert cached_elapsed_ms < 175


def test_final_response_cache_keys_on_query_and_retrieved_passages(monkeypatch) -> None:
    from src import rag_pipeline

    rag_pipeline._cached_final_response.cache_clear()
    calls = []

    def synthesize(question, chunks, response_language, *args, **kwargs):
        calls.append((question, tuple(chunk["text"] for chunk in chunks)))
        return f"cached synthesis {len(calls)}"

    monkeypatch.setattr(rag_pipeline, "_synthesize_answer", synthesize)
    question = "Unique response-cache query with evidence"
    passages = [
        {
            "chunk_id": "unique-response-cache-passage",
            "text": "Retrieved evidence for the response cache.",
            "source": "response-cache-paper",
            "page_number": 2,
            "section_type": "Methods",
        }
    ]

    first = rag_pipeline._synthesize_response(question, passages, "English", False, False)
    repeated = rag_pipeline._synthesize_response(question, passages, "English", False, False)
    changed = rag_pipeline._synthesize_response(
        question,
        [{**passages[0], "text": "Changed response-cache evidence."}],
        "English",
        False,
        False,
    )

    assert first["answer"] == repeated["answer"] == "cached synthesis 1"
    assert first["chunks"] == repeated["chunks"] == passages
    assert "response-cache-paper" in str(first["provenance"])
    assert changed["answer"] == "cached synthesis 2"
    assert len(calls) == 2
    assert rag_pipeline._cached_final_response.cache_info().hits == 1
