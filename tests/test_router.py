from __future__ import annotations

from src.rag_pipeline import answer_query_details
from src.router import route_query


def test_compiled_router_distinguishes_direct_and_retrieval_queries() -> None:
    assert route_query("Hello!")["route"] == "direct_response"
    assert route_query("BERT")["route"] == "rag_retrieval"
    assert route_query("derive the gradient equation")["embedding_key"] == "bge"
    assert route_query("How does the architecture work?")["embedding_key"] == "e5"
    assert route_query("BERT")["embedding_key"] == "minilm"
    assert route_query.cache_info().maxsize == 500


def test_direct_response_skips_index_and_streams_response(tmp_path, monkeypatch) -> None:
    monkeypatch.setattr(
        "src.rag_pipeline.classify_query",
        lambda *args, **kwargs: (_ for _ in ()).throw(AssertionError("model classifier was called")),
    )
    streamed: list[str] = []

    result = answer_query_details(
        "Hola!",
        tmp_path / "missing-index",
        response_language="Spanish",
        stream_consumer=lambda chunks: streamed.extend(chunks) or "".join(streamed),
    )

    assert result["route"] == "direct_response"
    assert result["response_language"] == "Spanish"
    assert "¡Hola!" in result["answer"]
    assert streamed == [result["answer"]]
    assert result["chunks"] == []
