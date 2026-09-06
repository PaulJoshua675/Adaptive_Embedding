from __future__ import annotations

import os
import re
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
LATENCY_LOG = ROOT / "tests" / "latency_bench.txt"


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
        [
            sys.executable,
            "-m",
            "streamlit",
            "run",
            "app.py",
            "--server.headless",
            "true",
            "--server.port",
            port,
            "--",
            "--streamlit",
        ],
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


def test_technical_breakdown_has_hard_bullet_and_latency_limits(live_server: str) -> None:
    query = (
        "Compare the transformer attention mechanism, embedding dimensions, retrieval ranking, "
        "training steps, evaluation metrics, and operational differences in exhaustive technical detail."
    )
    with sync_playwright() as playwright:
        browser = playwright.chromium.launch(headless=True)
        page: Page = browser.new_page(viewport={"width": 1440, "height": 1000})
        page.goto(live_server, wait_until="domcontentloaded")
        page.get_by_text("Browse Central Repository", exact=True).wait_for(timeout=30_000)
        question = page.locator('textarea[aria-label="Research question"]')
        question.fill(query)
        started = time.perf_counter()
        page.get_by_role("button", name="Retrieve grounded answer", exact=True).click()
        page.get_by_text("Technical Breakdown", exact=True).wait_for(timeout=60_000)
        elapsed_seconds = time.perf_counter() - started

        body = page.locator("body").inner_text()
        technical_text = body.split("Technical Breakdown", 1)[1].split("Verified Source Citations", 1)[0]
        point_count = len(re.findall(r"Point\s+\d+", technical_text, flags=re.IGNORECASE))
        bullet_elements = [
            text for text in page.locator("li").all_inner_texts()
            if re.search(r"Point\s+\d+", text, flags=re.IGNORECASE)
        ]
        assert point_count <= 12
        assert len(bullet_elements) <= 12
        assert elapsed_seconds < 2.5

        LATENCY_LOG.write_text(
            f"URL: {live_server}\n"
            f"Elapsed seconds: {elapsed_seconds:.4f}\n"
            f"Technical point count: {point_count}\n"
            f"Technical bullet element count: {len(bullet_elements)}\n",
            encoding="utf-8",
        )
        assert LATENCY_LOG.exists()
        browser.close()
