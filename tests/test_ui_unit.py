from types import SimpleNamespace

from app import _render_answer


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
    }

    _render_answer(fake_streamlit, result, "Explain the method", __import__("pathlib").Path("figures"))

    markdown_values = [event[1] for event in rendered if event[0] == "markdown"]
    assert result["answer"] in markdown_values
    assert any("Retrieval evaluation" in value for value in markdown_values)
    assert any(kind == "caption" and "conceptual_general" in value for kind, value in rendered)


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