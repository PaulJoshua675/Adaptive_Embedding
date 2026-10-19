from __future__ import annotations

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

ROOT = Path(__file__).resolve().parents[1]
BASE_URL = os.getenv("RAG_TEST_URL", "http://localhost:8501")
REPORT_PATH = ROOT / "tests" / "ultra_fast_report.md"


@pytest.fixture(scope="module")
def live_server(streamlit_app_url):
    return streamlit_app_url


def test_targeted_query_is_sub_second(live_server: str) -> None:
    query = "Show the methodology section and explain the attention score formula and retrieval ranking details."
    with sync_playwright() as playwright:
        browser = playwright.chromium.launch(headless=True)
        page: Page = browser.new_page(viewport={"width": 1440, "height": 1000})
        page.goto(live_server, wait_until="domcontentloaded")
        page.get_by_text("Browse Central Repository", exact=True).wait_for(timeout=30_000)
        page.locator('textarea[aria-label="Research question"]').fill(query)
        started = time.perf_counter()
        page.get_by_role("button", name="Retrieve grounded answer", exact=True).click()
        page.get_by_text("Technical Breakdown", exact=True).wait_for(timeout=60_000)
        elapsed_seconds = time.perf_counter() - started
        response_text = page.locator("body").inner_text()
        assert "Executive Insights" in response_text
        assert "Technical Breakdown" in response_text
        assert elapsed_seconds < 1.0
        REPORT_PATH.write_text(
            "# Ultra-Fast Retrieval Report\n\n"
            f"- URL: `{live_server}`\n"
            f"- Query latency: `{elapsed_seconds:.4f}` seconds\n"
            "- SLA threshold: `1.0000` seconds\n"
            f"- Result: `{'PASS' if elapsed_seconds < 1.0 else 'FAIL'}`\n",
            encoding="utf-8",
        )
        assert REPORT_PATH.exists()
        browser.close()
