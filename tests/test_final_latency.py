from __future__ import annotations

import os
import subprocess
import sys
import time
from pathlib import Path
from urllib.request import urlopen

import pytest

pytest.importorskip("playwright.sync_api")
from playwright.sync_api import Page, sync_playwright


ROOT = Path(__file__).resolve().parents[1]
BASE_URL = os.getenv("RAG_TEST_URL", "http://localhost:8501")
SLA_SECONDS = 1.5


@pytest.fixture(scope="module")
def live_server():
    try:
        with urlopen(BASE_URL, timeout=2):
            yield BASE_URL
            return
    except Exception:
        pass
    environment = os.environ.copy()
    environment.update({"PYTHONUTF8": "1", "TRANSFORMERS_OFFLINE": "1", "TOKENIZERS_PARALLELISM": "false"})
    port = BASE_URL.rsplit(":", 1)[-1]
    process = subprocess.Popen(
        [sys.executable, "-m", "streamlit", "run", "app.py", "--server.headless", "true", "--server.port", port, "--", "--streamlit"],
        cwd=ROOT,
        env=environment,
        stdout=subprocess.PIPE,
        stderr=subprocess.STDOUT,
        text=True,
    )
    deadline = time.time() + 45
    while time.time() < deadline:
        try:
            with urlopen(BASE_URL, timeout=2):
                break
        except Exception:
            time.sleep(0.5)
    else:
        process.terminate()
        raise RuntimeError("Streamlit did not start")
    yield BASE_URL
    process.terminate()
    process.wait(timeout=10)


def test_dense_query_completes_under_latency_sla(live_server: str) -> None:
    query = (
        "Explain transformer attention using query key value projections, scaled dot product softmax, "
        "residual connections, and the reported evaluation metrics."
    )
    with sync_playwright() as playwright:
        browser = playwright.chromium.launch(headless=True)
        page: Page = browser.new_page(viewport={"width": 1440, "height": 1000})
        page.goto(live_server, wait_until="domcontentloaded")
        page.get_by_text("Browse Central Repository", exact=True).wait_for(timeout=60_000)
        page.locator('textarea[aria-label="Research question"]').fill(query)
        started = time.perf_counter()
        page.get_by_role("button", name="Retrieve grounded answer", exact=True).click()
        page.get_by_text("Technical Breakdown", exact=True).wait_for(timeout=60_000)
        elapsed_seconds = time.perf_counter() - started
        assert "Executive Insights" in page.locator("body").inner_text()
        assert elapsed_seconds < SLA_SECONDS, f"query latency was {elapsed_seconds:.4f}s"
        browser.close()