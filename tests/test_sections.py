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
BASE_URL = "http://localhost:8511"
ABSTRACT_LOG = ROOT / "tests" / "abstract_test.txt"
FIGURE_PROOF = ROOT / "tests" / "figure30_rendered_proof.png"


def _create_fixture_dataset(root: Path) -> Path:
    dataset = root / "dataset"
    dataset.mkdir()
    pdf_path = dataset / "2401.12345.pdf"
    document = fitz.open()
    page = document.new_page()
    page.insert_text((60, 70), "ABSTRACT", fontsize=16)
    page.insert_textbox(
        fitz.Rect(60, 90, 540, 180),
        "We study adaptive retrieval for research question answering. Our method selects an embedding model by query intent and improves citation coverage.",
        fontsize=11,
    )
    page.insert_text((60, 230), "I. INTRODUCTION", fontsize=16)
    page.insert_textbox(fitz.Rect(60, 250, 540, 340), "Introduction context describes the research motivation and benchmark setup.", fontsize=11)
    page.insert_text((60, 390), "Figure 30: Metadata ratings as a function of model type and model size", fontsize=11)
    image = fitz.Pixmap(fitz.csRGB, fitz.IRect(0, 0, 320, 160), False)
    image.clear_with(0x2D6A73)
    page.insert_image(fitz.Rect(100, 420, 420, 580), pixmap=image)
    page.insert_text((60, 650), "CONCLUSION", fontsize=16)
    page.insert_textbox(fitz.Rect(60, 670, 540, 760), "The conclusion reports that adaptive routing improves grounded answers and maintains provenance.", fontsize=11)
    document.save(pdf_path)
    document.close()
    return pdf_path


@pytest.fixture(scope="module")
def streamlit_server(tmp_path_factory):
    workspace = tmp_path_factory.mktemp("section_fixture")
    _create_fixture_dataset(workspace)
    vector_store = workspace / "vector_store"
    static_dir = workspace / "static"
    environment = os.environ.copy()
    environment.update({"PYTHONUTF8": "1", "TRANSFORMERS_OFFLINE": "1", "TOKENIZERS_PARALLELISM": "false"})
    command = [
        sys.executable, "-m", "streamlit", "run", "app.py",
        "--server.headless", "true", "--server.port", "8511", "--",
        "--streamlit", "--dataset-dir", str(workspace / "dataset"),
        "--vector-store-dir", str(vector_store),
    ]
    environment["RAG_STATIC_DIR"] = str(static_dir)
    process = subprocess.Popen(command, cwd=ROOT, env=environment, stdout=subprocess.PIPE, stderr=subprocess.STDOUT, text=True)
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


def _open_repository(page: Page, url: str) -> None:
    page.goto(url, wait_until="domcontentloaded")
    page.get_by_text("Browse Central Repository", exact=True).wait_for()


def _ask(page: Page, query: str) -> None:
    page.locator('textarea[aria-label="Research question"]').fill(query)
    page.get_by_role("button", name="Retrieve grounded answer", exact=True).click()
    page.get_by_text("Executive Insights", exact=True).wait_for(timeout=60_000)


def test_section_boundary_filtering(streamlit_server):
    with sync_playwright() as playwright:
        browser = playwright.chromium.launch(headless=True)
        page = browser.new_page(viewport={"width": 1440, "height": 1000})
        _open_repository(page, streamlit_server)
        _ask(page, "Give me the abstract of the paper")
        response = page.locator("body").inner_text()
        assert "adaptive retrieval" in response.lower()
        assert "conclusion reports" not in response.lower()
        ABSTRACT_LOG.write_text(response, encoding="utf-8")
        assert ABSTRACT_LOG.exists()
        browser.close()


def test_caption_to_figure_retrieval(streamlit_server):
    with sync_playwright() as playwright:
        browser = playwright.chromium.launch(headless=True)
        page = browser.new_page(viewport={"width": 1440, "height": 1000})
        _open_repository(page, streamlit_server)
        _ask(page, "Show the diagram for Metadata ratings as a function of model type and model size")
        page.get_by_text("Matched figures", exact=True).wait_for(timeout=30_000)
        page.locator("img").last.wait_for(timeout=30_000)
        assert page.locator("img").count() > 0
        page.locator("img").last.screenshot(path=str(FIGURE_PROOF))
        assert FIGURE_PROOF.exists()
        browser.close()
