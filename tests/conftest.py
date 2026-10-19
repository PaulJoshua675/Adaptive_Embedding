from __future__ import annotations

import os

import pytest


@pytest.fixture(scope="session")
def streamlit_app_url() -> str:
    url = os.getenv("RAG_TEST_URL")
    if not url:
        pytest.skip("Set RAG_TEST_URL to run Streamlit browser integration tests.")
    return url