from __future__ import annotations

import json
import os
import subprocess
import sys
import time
from pathlib import Path
from urllib.request import urlopen

import pytest

pytestmark = pytest.mark.integration

pytest.importorskip("playwright.sync_api")
from playwright.sync_api import Page, sync_playwright

from src.evaluation_suite import retrieval_metrics
from src.retrieval import find_matching_figures

ROOT = Path(__file__).resolve().parents[1]
BASE_URL = os.getenv("RAG_TEST_URL", "http://localhost:8501")
PROOF_PATH = ROOT / "tests" / "evaluation_run_proof.json"


@pytest.fixture(scope="module")
def evaluation_server(streamlit_app_url):
    return streamlit_app_url


def _ask(page: Page, query: str) -> str:
    page.locator('textarea[aria-label="Research question"]').fill(query)
    page.get_by_role("button", name="Retrieve grounded answer", exact=True).click()
    page.get_by_text("Technical Breakdown", exact=True).wait_for(timeout=60_000)
    return page.locator("body").inner_text()


def test_metrics_dashboard_and_figure_alignment(evaluation_server: str) -> None:
    queries = [
        "Explain the transformer architecture and attention mechanism.",
        "What does retrieval mean in this paper?",
        "Show Figure 30",
    ]
    with sync_playwright() as playwright:
        browser = playwright.chromium.launch(headless=True)
        page = browser.new_page(viewport={"width": 1440, "height": 1000})
        page.goto(evaluation_server, wait_until="domcontentloaded")
        page.get_by_text("Browse Central Repository", exact=True).wait_for()

        body_text = ""
        for query in queries:
            body_text = _ask(page, query)
            assert "Precision" in body_text
            assert "Recall" in body_text
            assert "F1-Score" in body_text

        manifest_path = ROOT / "vector_store" / "figures" / "figure_manifest.json"
        matches = find_matching_figures("Show Figure 30", manifest_path)
        assert matches, "Figure 30 must pass dual verification"
        expected_path = str(matches[0]["file_path"])
        assert Path(expected_path).name.endswith("_fig_30.png")
        figure = page.locator("img").last
        figure.wait_for(timeout=60_000)

        proof = {
            "url": evaluation_server,
            "queries": queries,
            "metrics_visible": {name: name in body_text for name in ("Precision", "Recall", "F1-Score")},
            "retrieval_metrics": retrieval_metrics(["figure-30"], ["figure-30"], k=3),
            "verified_figure_path": expected_path,
            "figure_loaded": bool(figure.get_attribute("src")),
        }
        PROOF_PATH.write_text(json.dumps(proof, indent=2), encoding="utf-8")
        assert all(proof["metrics_visible"].values())
        assert proof["figure_loaded"]
        assert PROOF_PATH.exists()
        browser.close()
