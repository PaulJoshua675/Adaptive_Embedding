from __future__ import annotations

import os
import subprocess
import sys
import time
from pathlib import Path
from urllib.request import urlopen

import fitz
import pytest

playwright = pytest.importorskip("playwright.sync_api")
from playwright.sync_api import Page, TimeoutError as PlaywrightTimeoutError, sync_playwright


ROOT = Path(__file__).resolve().parents[1]
BASE_URL = "http://localhost:8501"
OUTPUT_LOG = ROOT / "tests" / "test_output.md"
SCREENSHOT = ROOT / "tests" / "review2_proof.png"


def _create_test_pdf(path: Path) -> None:
    document = fitz.open()
    page = document.new_page()
    page.insert_textbox(
        fitz.Rect(60, 70, 540, 760),
        "Adaptive retrieval research paper\n\n"
        "This study evaluates retrieval augmented generation for scientific question answering. "
        "The system uses semantic chunking, multi-model embeddings, and page-level provenance.\n\n"
        "The technical method compares dense retrieval with keyword fallback and routes technical "
        "questions to an embedding model selected by query intent. Evaluation measures retrieval "
        "relevance, answer faithfulness, and citation coverage across research domains.",
        fontsize=12,
    )
    document.save(path)
    document.close()


@pytest.fixture(scope="module")
def streamlit_server():
    environment = os.environ.copy()
    environment.update({"PYTHONUTF8": "1", "TRANSFORMERS_OFFLINE": "1", "TOKENIZERS_PARALLELISM": "false"})
    process = subprocess.Popen(
        [sys.executable, "-m", "streamlit", "run", "app.py", "--server.headless", "true", "--server.port", "8501", "--", "--streamlit"],
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
        output = process.stdout.read() if process.stdout else ""
        raise RuntimeError(f"Streamlit did not start:\n{output}")
    yield BASE_URL
    process.terminate()
    try:
        process.wait(timeout=10)
    except subprocess.TimeoutExpired:
        process.kill()


def test_repository_and_custom_upload_workflows(streamlit_server, tmp_path):
    pdf_path = tmp_path / "adaptive_retrieval_test.pdf"
    _create_test_pdf(pdf_path)

    with sync_playwright() as playwright_context:
        browser = playwright_context.chromium.launch(headless=True)
        page: Page = browser.new_page(viewport={"width": 1440, "height": 1000})
        page.goto(streamlit_server, wait_until="domcontentloaded")
        page.get_by_text("Browse Central Repository", exact=True).wait_for()
        page.get_by_text("Upload Custom Research Paper", exact=True).click()
        page.locator('input[type="file"]').set_input_files(str(pdf_path))
        page.get_by_text("Indexed adaptive_retrieval_test.pdf in memory with page-level provenance.", exact=True).wait_for(timeout=30_000)

        textareas = page.locator("textarea")
        textareas.last.fill("How does the system route technical questions and preserve citations?")
        page.get_by_role("button", name="Query uploaded paper", exact=True).click()
        page.get_by_text("Executive Insights", exact=True).wait_for(timeout=60_000)

        body_text = page.locator("body").inner_text()
        assert "Technical Breakdown" in body_text
        assert len(body_text.split("Executive Insights", 1)[-1].strip().splitlines()) >= 4
        OUTPUT_LOG.write_text(
            "# Playwright verification output\n\n" + body_text + "\n",
            encoding="utf-8",
        )
        page.screenshot(path=str(SCREENSHOT), full_page=True)
        assert OUTPUT_LOG.exists()
        assert SCREENSHOT.exists()
        browser.close()