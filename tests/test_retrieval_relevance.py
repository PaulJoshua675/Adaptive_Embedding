from __future__ import annotations

import json
import re

import pytest

from config import DATASET_DIR, DOMAIN_EXAMPLE_QUESTIONS
from src.classifier import QueryClassifier
from src.create_vector_db import VectorStore
from src.embedding_model import EmbeddingModel
from src.retrieval import inject_visual_payload
from src.rag_pipeline import _retrieve_with_fallback
from src.retriever import retrieve_top_chunks


@pytest.mark.parametrize(
    ("domain", "embedding_key"),
    [
        ("cs.CV", "e5"),
        ("cs.CL", "e5"),
        ("cs.LG", "bge"),
        ("cs.AI", "e5"),
        ("cs.IR", "e5"),
    ],
)
def test_curated_questions_route_to_their_intended_embedding(domain, embedding_key) -> None:
    classifier = QueryClassifier()
    classifier._load_zero_shot = lambda: False

    result = classifier.predict(DOMAIN_EXAMPLE_QUESTIONS[domain][0])

    assert result["embedding_key"] == embedding_key


def test_curated_questions_reference_papers_in_the_local_corpus() -> None:
    for questions in DOMAIN_EXAMPLE_QUESTIONS.values():
        for question in questions:
            match = re.search(r"(?<!\d)(\d{4}\.\d{4,5})(?:v\d+)?(?!\d)", question)
            assert match, f"Curated question has no explicit corpus paper ID: {question}"
            assert (DATASET_DIR / f"{match.group(1)}.pdf").is_file()


def test_short_entity_query_routes_to_minilm() -> None:
    classifier = QueryClassifier()
    classifier._load_zero_shot = lambda: False

    assert classifier.predict("BERT")["embedding_key"] == "minilm"


def test_legacy_retriever_drops_passages_below_similarity_cutoff() -> None:
    documents = [{"text": "Vision transformers improve dense prediction."}]
    model = EmbeddingModel().fit([str(document["text"]) for document in documents])
    store = VectorStore(dimension=len(model.embed_text(documents[0]["text"])))
    store.add_documents(documents, model.embed_texts([documents[0]["text"]]))

    results = retrieve_top_chunks("Explain quantum chemistry", store, model)

    assert results == []


def test_arxiv_id_in_query_pins_legacy_retrieval_to_that_paper() -> None:
    documents = [
        {"source": "2203.02155.pdf", "text": "The KL reward coefficient affects DROP and SQuAD."},
        {"source": "2307.09288.pdf", "text": "The KL reward coefficient affects DROP and SQuAD."},
    ]
    model = EmbeddingModel().fit([str(document["text"]) for document in documents])
    store = VectorStore(dimension=len(model.embed_text(documents[0]["text"])))
    store.add_documents(documents, model.embed_texts([str(document["text"]) for document in documents]))

    results = retrieve_top_chunks(
        "How does the KL reward coefficient affect DROP and SQuAD in 2203.02155?",
        store,
        model,
    )

    assert results
    assert all(document["source"] == "2203.02155.pdf" for _, document in results)
    assert retrieve_top_chunks("Explain the same paper in 9999.99999", store, model) == []


def test_visual_payload_matches_explicit_figure_number_from_manifest(tmp_path) -> None:
    figure_path = tmp_path / "figure.png"
    manifest_path = tmp_path / "figure_manifest.json"
    manifest_path.write_text(
        json.dumps(
            {
                "Figure 3": {
                    "caption": "Vision transformer architecture for dense prediction",
                    "file_path": str(figure_path),
                    "paper_id": "paper-a",
                    "page_number": 2,
                }
            }
        ),
        encoding="utf-8",
    )
    payload = {
        "answer": "Retrieved evidence.",
        "chunks": [
            {"source": "paper-b.pdf", "text": "Figure 3 illustrates a dense prediction architecture."},
            {"source": "paper-a.pdf", "text": "The model improves dense prediction."},
        ],
        "figures": [{"file_path": "stale.png"}],
        "rendered_figure_path": "stale.png",
    }

    filtered = inject_visual_payload(payload, "Show Figure 3 vision transformer", manifest_path)

    assert [figure["paper_id"] for figure in filtered["figures"]] == ["paper-a"]
    assert filtered["matched_figure"]["number"] == "3"
    assert filtered["rendered_figure_path"] == str(figure_path)


def test_visual_payload_matches_figure_number_without_retrieved_passages(tmp_path) -> None:
    figure_path = tmp_path / "figure.png"
    figure_path.write_bytes(b"figure")
    manifest_path = tmp_path / "figure_manifest.json"
    manifest_path.write_text(
        json.dumps(
            {
                "Figure 35": {
                    "caption": "Evaluation on public NLP datasets",
                    "file_path": str(figure_path),
                    "paper_id": "paper-a",
                    "page_number": 12,
                }
            }
        ),
        encoding="utf-8",
    )

    result = inject_visual_payload(
        {"chunks": []},
        "Show Fig. 35: Evaluation on public NLP datasets",
        manifest_path,
    )

    assert len(result["figures"]) == 1
    assert result["matched_figure"]["resolved_path"] == str(figure_path.resolve())


def test_visual_payload_does_not_match_number_when_caption_conflicts_with_query(tmp_path) -> None:
    figure_path = tmp_path / "figure.png"
    manifest_path = tmp_path / "figure_manifest.json"
    manifest_path.write_text(
        json.dumps(
            {
                "Figure 35": {
                    "caption": "Pareto front of mean error and prediction time",
                    "file_path": str(figure_path),
                    "paper_id": "paper-a",
                }
            }
        ),
        encoding="utf-8",
    )

    result = inject_visual_payload(
        {"chunks": []},
        "Figure 35: Evaluation on public NLP datasets",
        manifest_path,
    )

    assert result["figures"] == []
    assert "matched_figure" not in result


def test_visual_payload_keeps_a_figure_referenced_by_retrieved_passage(tmp_path) -> None:
    figure_path = tmp_path / "figure.png"
    manifest_path = tmp_path / "figure_manifest.json"
    manifest_path.write_text(
        json.dumps(
            {
                "Figure 3": {
                    "caption": "Vision transformer architecture for dense prediction",
                    "file_path": str(figure_path),
                    "paper_id": "paper-a",
                    "page_number": 2,
                }
            }
        ),
        encoding="utf-8",
    )
    payload = {
        "answer": "Retrieved evidence.",
        "chunks": [
            {"source": "paper-a.pdf", "text": "Figure 3 illustrates the vision transformer architecture."},
        ],
    }

    filtered = inject_visual_payload(payload, "Show Figure 3 vision transformer", manifest_path)

    assert [figure["paper_id"] for figure in filtered["figures"]] == ["paper-a"]
    assert filtered["rendered_figure_path"] == str(figure_path)
    assert filtered["figures"][0]["resolved_path"] == str(figure_path.resolve())


def test_visual_payload_rejects_manifest_and_chunk_figures_without_chunk_reference(tmp_path) -> None:
    figure_path = tmp_path / "figure.png"
    manifest_path = tmp_path / "figure_manifest.json"
    manifest_path.write_text(
        json.dumps(
            {
                "Figure 5": {
                    "caption": "Figure 5: Vision transformer architecture",
                    "file_path": str(figure_path),
                    "paper_id": "paper-a",
                    "page_number": 2,
                }
            }
        ),
        encoding="utf-8",
    )
    payload = {
        "chunks": [
            {
                "source": "paper-a.pdf",
                "text": "The model improves dense prediction.",
                "figures": [
                    {
                        "caption": "Figure 5: Vision transformer architecture",
                        "file_path": str(figure_path),
                        "paper_id": "paper-a",
                    }
                ],
            }
        ]
    }

    result = inject_visual_payload(payload, "Show the vision transformer figure", manifest_path)

    assert result["figures"] == []
    assert "rendered_figure_path" not in result


def test_visual_payload_accepts_abbreviated_reference_in_same_chunk(tmp_path) -> None:
    figure_path = tmp_path / "figure.png"
    payload = {
        "chunks": [
            {
                "source": "paper-a.pdf",
                "text": "Fig. 5 presents the architecture.",
                "figures": [
                    {
                        "caption": "Figure 5: Vision transformer architecture",
                        "file_path": str(figure_path),
                        "paper_id": "paper-a",
                    }
                ],
            }
        ]
    }

    result = inject_visual_payload(
        payload,
        "Show the vision transformer figure",
        tmp_path / "missing_manifest.json",
    )

    assert [figure["file_path"] for figure in result["figures"]] == [str(figure_path)]


def test_visual_payload_includes_figures_linked_to_retrieved_chunks(tmp_path) -> None:
    figure_path = tmp_path / "figure.png"
    payload = {
        "chunks": [
            {
                "source": "paper-a.pdf",
                "text": "Figure 3 illustrates the architecture.",
                "figures": [
                    {
                        "caption": "Figure 3: Vision transformer architecture",
                        "file_path": str(figure_path),
                        "paper_id": "paper-a",
                    }
                ],
            }
        ]
    }

    result = inject_visual_payload(
        payload,
        "Show the vision transformer figure",
        tmp_path / "missing_manifest.json",
    )

    assert result["figures"][0]["file_path"] == str(figure_path)
    assert result["matched_figure"]["file_path"] == str(figure_path)


def test_visual_payload_resolves_relative_manifest_figure_paths(tmp_path) -> None:
    figures_dir = tmp_path / "figures"
    figures_dir.mkdir()
    figure_path = figures_dir / "figure.png"
    figure_path.write_bytes(b"image")
    manifest_path = figures_dir / "figure_manifest.json"
    manifest_path.write_text(
        json.dumps(
            {
                "Figure 3": {
                    "caption": "Vision transformer architecture",
                    "file_path": "figure.png",
                    "paper_id": "paper-a",
                    "page_number": 2,
                }
            }
        ),
        encoding="utf-8",
    )
    payload = {
        "chunks": [
            {"source": "paper-a.pdf", "text": "Figure 3 illustrates the architecture."},
        ]
    }

    result = inject_visual_payload(payload, "Show Figure 3", manifest_path)

    assert result["figures"][0]["resolved_path"] == str(figure_path.resolve())
    assert result["figures"][0]["file_path"] == str(figure_path.resolve())


def test_retriever_requests_low_relevance_passages_for_fallback() -> None:
    class StubRetriever:
        def __init__(self) -> None:
            self.last_timings_ms = {}
            self.allow_low_relevance_calls = 0
            self.candidate_counts = []

        def retrieve(self, question, embedding_key, top_k, candidate_k, section_type, query_vector, include_low_relevance=False):
            self.candidate_counts.append(candidate_k)
            if include_low_relevance:
                self.allow_low_relevance_calls += 1
                return [{"text": "Adjacent research context.", "score": 0.08}]
            return []

    retriever = StubRetriever()

    chunks, fallback = _retrieve_with_fallback(
        retriever,
        "An unrelated query",
        "e5",
        3,
        None,
        [1.0],
    )

    assert fallback is True
    assert retriever.allow_low_relevance_calls == 1
    assert retriever.candidate_counts == [3, 5]
    assert chunks[0]["score"] == 0.08
