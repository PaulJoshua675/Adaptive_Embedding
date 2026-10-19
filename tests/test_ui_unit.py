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

    markdown_values = [value for kind, value in rendered if kind == "markdown"]
    assert result["answer"] in markdown_values
    assert any("Retrieval evaluation" in value for value in markdown_values)
    assert any(kind == "caption" and "conceptual_general" in value for kind, value in rendered)