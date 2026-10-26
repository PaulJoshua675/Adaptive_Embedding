from types import SimpleNamespace

from app import _render_answer, _render_performance_metrics, _restore_query_state


def test_render_answer_uses_streamlit_components_for_answer_and_metrics():
    rendered = []
    fake_streamlit = SimpleNamespace(
        caption=lambda value: rendered.append(("caption", value)),
        markdown=lambda value: rendered.append(("markdown", value)),
        image=lambda *args, **kwargs: rendered.append(("image", args, kwargs)),
        info=lambda value: rendered.append(("info", value)),
    )
    result = {
        "answer": "Executive Insights\n\nGrounded result.\n\nTechnical Breakdown\n\nEvidence details.",
        "query_type": "conceptual_general",
        "route_confidence": 0.8,
        "embedding_key": "e5",
        "chunks": [],
        "eval_metrics": {
            "accuracy": 1.0,
            "precision": 1.0,
            "recall": 1.0,
            "f1": 1.0,
        },
    }

    _render_answer(fake_streamlit, result, "Explain the method", __import__("pathlib").Path("figures"))

    markdown_values = [event[1] for event in rendered if event[0] == "markdown"]
    assert result["answer"] in markdown_values
    assert any("Retrieval evaluation" in value for value in markdown_values)
    assert any(kind == "caption" and "conceptual_general" in value for kind, value in rendered)


def test_cached_payload_renders_citations_and_cached_evaluation_metrics(tmp_path):
    rendered = []
    fake_streamlit = SimpleNamespace(
        caption=lambda value: rendered.append(("caption", value)),
        markdown=lambda value: rendered.append(("markdown", value)),
        image=lambda *args, **kwargs: rendered.append(("image", args, kwargs)),
        info=lambda value: rendered.append(("info", value)),
    )
    passages = [{"chunk_id": "paper-1", "source": "paper.pdf", "page_number": 2}]
    result = {
        "answer": "Grounded result.",
        "query_type": "conceptual_general",
        "route_confidence": 0.8,
        "embedding_key": "e5",
        "chunks": passages,
        "retrieved_passages": passages,
        "citations": "[paper.pdf | Page 2 | GENERAL TEXT](file:///paper.pdf)",
        "eval_metrics": {
            "accuracy": 1.0,
            "precision": 1.0,
            "recall": 1.0,
            "f1": 1.0,
        },
        "semantic_cache_hit": True,
    }
    session_state = {}

    _restore_query_state(session_state, "repository", result, 0.003)
    _render_answer(fake_streamlit, session_state["repository_result"], "Question", tmp_path)

    assert session_state["repository_passages"] == passages
    assert session_state["repository_eval_metrics"] == result["eval_metrics"]
    assert session_state["repository_elapsed"] == 0.003
    markdown_values = [event[1] for event in rendered if event[0] == "markdown"]
    assert "### Sources" in markdown_values
    assert result["citations"] in markdown_values
    assert any("Accuracy: **1.000**" in value for value in markdown_values)


def test_render_answer_shows_fallback_badge_and_retrieved_figure(tmp_path):
    rendered = []
    image_path = tmp_path / "figure.png"
    image_path.write_bytes(b"figure")
    fake_streamlit = SimpleNamespace(
        caption=lambda value: rendered.append(("caption", value)),
        markdown=lambda value: rendered.append(("markdown", value)),
        image=lambda *args, **kwargs: rendered.append(("image", args, kwargs)),
        info=lambda value: rendered.append(("info", value)),
    )
    result = {
        "answer": "Related research context.",
        "query_type": "conceptual_general",
        "route_confidence": 0.8,
        "embedding_key": "e5",
        "chunks": [],
        "fallback": True,
        "response_language": "fr",
        "figures": [{"file_path": str(image_path), "caption": "Related figure"}],
        "matched_figure": {
            "file_path": str(image_path),
            "number": "35",
            "caption": "Evaluation on public NLP datasets",
        },
    }

    _render_answer(fake_streamlit, result, "Show a related figure", tmp_path)

    assert any(
        kind == "info" and "Aucune correspondance exacte" in value
        for kind, value in rendered
    )
    markdown_values = [event[1] for event in rendered if event[0] == "markdown"]
    assert markdown_values[-1] == "### 🖼️ Images associées"
    assert any(
        event[0] == "image"
        and event[2].get("caption", "") == "Figure 35: Evaluation on public NLP datasets"
        for event in rendered
        if len(event) == 3
    )


def test_render_answer_displays_image_asset_from_retrieved_figure_chunk(tmp_path):
    rendered = []
    image_path = tmp_path / "truthfulqa.png"
    image_path.write_bytes(b"figure")
    fake_streamlit = SimpleNamespace(
        caption=lambda value: rendered.append(("caption", value)),
        markdown=lambda value: rendered.append(("markdown", value)),
        image=lambda *args, **kwargs: rendered.append(("image", args, kwargs)),
        info=lambda value: rendered.append(("info", value)),
    )
    result = {
        "answer": "The retrieved visual shows the reported dataset results.",
        "query_type": "conceptual_general",
        "route_confidence": 0.8,
        "embedding_key": "e5",
        "chunks": [
            {
                "visual_type": "figure",
                "caption": "Results on the TruthfulQA dataset",
                "figures": [
                    {
                        "caption": "Results on the TruthfulQA dataset",
                        "file_path": str(image_path),
                    }
                ],
            }
        ],
    }

    _render_answer(fake_streamlit, result, "TruthfulQA dataset results", tmp_path)

    assert any(
        event[0] == "image"
        and event[1][0] == str(image_path)
        and event[2]["caption"] == "Results on the TruthfulQA dataset"
        for event in rendered
        if len(event) == 3
    )


def test_render_answer_skips_evaluation_when_no_metrics_are_available(tmp_path):
    rendered = []
    fake_streamlit = SimpleNamespace(
        caption=lambda value: rendered.append(("caption", value)),
        markdown=lambda value: rendered.append(("markdown", value)),
        image=lambda *args, **kwargs: rendered.append(("image", args, kwargs)),
        info=lambda value: rendered.append(("info", value)),
    )

    _render_answer(
        fake_streamlit,
        {
            "answer": "Grounded response.",
            "query_type": "conceptual_general",
            "route_confidence": 0.8,
            "embedding_key": "e5",
            "chunks": [],
        },
        "Question",
        tmp_path,
    )

    markdown_values = [event[1] for event in rendered if event[0] == "markdown"]
    assert not any("Retrieval evaluation" in value for value in markdown_values)


def test_performance_metrics_render_only_available_timings_without_columns():
    rendered = []
    fake_streamlit = SimpleNamespace(
        caption=lambda value: rendered.append(value),
        columns=lambda *_args, **_kwargs: (_ for _ in ()).throw(
            AssertionError("timing metrics should not create columns")
        ),
    )

    _render_performance_metrics(
        fake_streamlit,
        {"profile_times": {"intent_classification_ms": 2.5}},
    )

    assert rendered == ["Timing breakdown (ms): Intent Classification Time (ms): 2.500"]


def test_performance_metrics_skip_timing_breakdown_when_timings_are_missing():
    rendered = []
    fake_streamlit = SimpleNamespace(caption=lambda value: rendered.append(value))

    _render_performance_metrics(fake_streamlit, {"profile_times": {}})

    assert rendered == []


def test_render_answer_localizes_related_image_heading_for_all_supported_languages(tmp_path):
    rendered = []
    image_path = tmp_path / "figure.png"
    image_path.write_bytes(b"figure")
    fake_streamlit = SimpleNamespace(
        caption=lambda value: rendered.append(("caption", value)),
        markdown=lambda value: rendered.append(("markdown", value)),
        image=lambda *args, **kwargs: rendered.append(("image", args, kwargs)),
        info=lambda value: rendered.append(("info", value)),
    )
    headings = {
        "English": "Related Images",
        "Tamil": "தொடர்புடைய படங்கள்",
        "French": "Images associées",
        "Hindi": "संबंधित छवियाँ",
        "Spanish": "Imágenes relacionadas",
    }

    for language, heading in headings.items():
        rendered.clear()
        _render_answer(
            fake_streamlit,
            {
                "answer": "Grounded response.",
                "query_type": "conceptual_general",
                "route_confidence": 0.8,
                "embedding_key": "e5",
                "chunks": [],
                "response_language": language,
                "figures": [{"file_path": str(image_path), "caption": "Related figure"}],
            },
            "Show a related figure",
            tmp_path,
        )

        markdown_values = [event[1] for event in rendered if event[0] == "markdown"]
        assert markdown_values[-1] == f"### 🖼️ {heading}"