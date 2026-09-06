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
BASE_URL = "http://localhost:8501"


def _create_sample_pdf(path: Path) -> None:
    document = fitz.open()
    page = document.new_page()
    page.insert_text((60, 70), "ABSTRACT", fontsize=16)
    page.insert_textbox(
        fitz.Rect(60, 95, 540, 195),
        "This academic study evaluates a disk-backed retrieval architecture. "
        "The system reports 93 percent citation coverage and uses page-level provenance.",
        fontsize=11,
    )
    page.insert_text((60, 245), "METHODOLOGY", fontsize=16)
    page.insert_textbox(
        fitz.Rect(60, 270, 540, 400),
        "The ingestion procedure serializes the PDF to a deterministic path, extracts structural blocks, "
        "stores figure snapshots, and builds an isolated multi-embedding index. "
        "The retrieval stage selects the custom workspace for upload queries.",
        fontsize=11,
    )
    page.insert_text((60, 455), "Figure 1: Disk-backed indexing architecture", fontsize=11)
    page.draw_rect(fitz.Rect(100, 480, 420, 650), color=(0.1, 0.35, 0.55), fill=(0.75, 0.85, 0.95))
    document.save(path)
    document.close()


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
            "8501",
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


def test_disk_backed_upload_renders_distinct_analysis(live_server: str, tmp_path: Path) -> None:
    pdf_path = tmp_path / "disk_backed_academic_paper.pdf"
    _create_sample_pdf(pdf_path)
    with sync_playwright() as playwright:
        browser = playwright.chromium.launch(headless=True)
        page: Page = browser.new_page(viewport={"width": 1440, "height": 1000})
        page.goto(live_server, wait_until="domcontentloaded")
        page.get_by_text("Upload Custom Research Paper", exact=True).click()
        page.locator('input[type="file"]').set_input_files(str(pdf_path))
        page.get_by_text("Indexed disk_backed_academic_paper.pdf from disk with page-level provenance.", exact=True).wait_for(timeout=60_000)

        page.locator('textarea[aria-label="Research question"]').last.fill("What citation coverage did the system report?")
        page.get_by_role("button", name="Query uploaded paper", exact=True).click()
        page.get_by_text("Technical Breakdown", exact=True).wait_for(timeout=60_000)

        body = page.locator("body").inner_text()
        assert "Executive Insights" in body
        assert "Technical Breakdown" in body
        assert "Verified Source Citations" in body
        assert "Accuracy" in body
        assert "Precision" in body
        assert "Recall" in body
        assert "F1-Score" in body
        assert "UNSECTIONED" not in body.upper()

        executive = body.split("Executive Insights", 1)[1].split("Technical Breakdown", 1)[0].strip()
        technical = body.split("Technical Breakdown", 1)[1].split("Verified Source Citations", 1)[0].strip()
        assert executive
        assert technical
        assert executive != technical
        browser.close()
