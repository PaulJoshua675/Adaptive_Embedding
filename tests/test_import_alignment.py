from __future__ import annotations

import importlib


def test_upload_preprocessing_exports_match_app_imports() -> None:
    preprocessing = importlib.import_module("src.preprocessing")
    app = importlib.import_module("app")

    expected_names = (
        "extract_uploaded_pdf_pages",
        "extract_page_images",
        "query_requests_visuals",
    )
    for name in expected_names:
        assert callable(getattr(preprocessing, name))
        assert callable(getattr(app, name))

    assert preprocessing.__all__
    assert set(expected_names).issubset(preprocessing.__all__)
