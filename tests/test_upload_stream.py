from __future__ import annotations

import os
import subprocess
import sys
import time
from pathlib import Path
from urllib.request import urlopen

import fitz
import pytest

pytestmark = pytest.mark.integration

pytest.importorskip("playwright.sync_api")
from playwright.sync_api import Page, sync_playwright

ROOT = Path(__file__).resolve().parents[1]
BASE_URL = os.getenv("RAG_TEST_URL", "http://localhost:8501")


def _create_academic_pdf(path: Path) -> None:
    document = fitz.open()
    page = document.new_page()
    page.insert_text((60, 70), "ABSTRACT", fontsize=16)
    page.insert_textbox(
        fitz.Rect(60, 95, 540, 190),
        "This study evaluates a resilient retrieval pipeline for academic documents. "
        "The proposed index reports 93 percent citation coverage.",
        fontsize=11,
    )
    page.insert_text((60, 240), "CONCLUSION", fontsize=16)
    page.insert_textbox(
        fitz.Rect(60, 265, 540, 360),
        "The conclusion confirms that page-level provenance improves grounded answers.",
        fontsize=11,
    )
    document.save(path)
    document.close()


@pytest.fixture(scope="module")
def live_server(streamlit_app_url):
    return streamlit_app_url


def test_custom_upload_builds_index_and_renders_citations(live_server: str, tmp_path: Path) -> None:
    pdf_path = tmp_path / "resilient_retrieval.pdf"
    _create_academic_pdf(pdf_path)
    with sync_playwright() as playwright:
        browser = playwright.chromium.launch(headless=True)
        page: Page = browser.new_page(viewport={"width": 1440, "height": 1000})
        page.goto(live_server, wait_until="domcontentloaded")
        page.get_by_text("Upload Custom Research Paper", exact=True).click()
        page.locator('input[type="file"]').set_input_files(str(pdf_path))
        page.get_by_text("Indexed resilient_retrieval.pdf from disk with page-level provenance.", exact=True).wait_for(timeout=60_000)
        body = page.locator("body").inner_text().lower()
        assert "contains no extractable text or images" not in body
        page.locator('textarea[aria-label="Research question"]').last.fill("What citation coverage was reported?")
        page.get_by_role("button", name="Query uploaded paper", exact=True).click()
        page.get_by_text("Verified Source Citations", exact=True).wait_for(timeout=60_000)
        assert "93" in page.locator("body").inner_text()
        browser.close()