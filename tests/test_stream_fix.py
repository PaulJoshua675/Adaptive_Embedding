from __future__ import annotations

import os
import subprocess
import sys
import time
from pathlib import Path
from urllib.request import urlopen

import fitz
import pytest

pytest.importorskip("playwright.sync_api")
from playwright.sync_api import Page, sync_playwright

ROOT = Path(__file__).resolve().parents[1]
BASE_URL = os.getenv("RAG_TEST_URL", "http://localhost:8501")


def _create_pdf(path: Path) -> None:
    document = fitz.open()
    page = document.new_page()
    page.insert_text((60, 70), "ABSTRACT", fontsize=16)
    page.insert_textbox(
        fitz.Rect(60, 95, 540, 220),
        "The Helios retrieval system combines page-aware chunking with a local embedding index. "
        "The experiment reports 91 percent recall for satellite image classification.",
        fontsize=11,
    )
    document.save(path)
    document.close()


@pytest.fixture(scope="module")
def streamlit_server():
    try:
        with urlopen(BASE_URL, timeout=2):
            yield BASE_URL
            return
    except Exception:
        pass
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
        raise RuntimeError("Streamlit did not start")
    yield BASE_URL
    process.terminate()
    process.wait(timeout=10)


def test_uploaded_stream_is_reset_and_indexed(streamlit_server: str, tmp_path: Path) -> None:
    pdf_path = tmp_path / "helios.pdf"
    _create_pdf(pdf_path)
    with sync_playwright() as playwright:
        browser = playwright.chromium.launch(headless=True)
        page: Page = browser.new_page(viewport={"width": 1440, "height": 1000})
        page.goto(streamlit_server, wait_until="domcontentloaded")
        page.get_by_text("Upload Custom Research Paper", exact=True).click()
        with page.expect_file_chooser() as chooser_info:
            page.locator('input[type="file"]').click(force=True)
        chooser_info.value.set_files(str(pdf_path))
        page.get_by_text("Indexed helios.pdf in memory with page-level provenance.", exact=True).wait_for(timeout=60_000)
        body_before_query = page.locator("body").inner_text().lower()
        assert "no extractable text or images" not in body_before_query
        page.locator('textarea[aria-label="Research question"]').last.fill("What recall did satellite image classification achieve?")
        page.get_by_role("button", name="Query uploaded paper", exact=True).click()
        page.get_by_text("Executive Insights", exact=True).wait_for(timeout=60_000)
        body = page.locator("body").inner_text()
        assert "Technical Breakdown" in body
        assert "Custom Uploaded Document" in body
        assert "Page 1" in body
        executive = body.split("Executive Insights", 1)[1].split("Technical Breakdown", 1)[0].strip()
        technical = body.split("Technical Breakdown", 1)[1].split("Verified Source Citations", 1)[0].strip()
        assert executive and technical and executive != technical
        browser.close()
