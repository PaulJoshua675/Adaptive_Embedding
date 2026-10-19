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


def _create_sample_pdf(path: Path) -> None:
    document = fitz.open()
    page = document.new_page()
    page.insert_text((60, 70), "ABSTRACT", fontsize=16)
    page.insert_textbox(
        fitz.Rect(60, 95, 540, 190),
        "The Aurora protocol uses a gated retrieval index to identify climate-risk evidence. "
        "Its evidence score is computed with a temperature parameter of 0.2.",
        fontsize=11,
    )
    page.insert_text((60, 240), "METHODOLOGY", fontsize=16)
    page.insert_textbox(
        fitz.Rect(60, 265, 540, 390),
        "The ingestion procedure extracts page text, splits it into overlapping chunks, and "
        "routes each question to the matching embedding index. The evaluation reports 94% recall "
        "for coastal flood-risk classification and stores provenance on page 1.",
        fontsize=11,
    )
    document.save(path)
    document.close()


@pytest.fixture(scope="module")
def live_server(streamlit_app_url):
    return streamlit_app_url


def test_live_upload_isolated_index_and_provenance(live_server: str, tmp_path: Path) -> None:
    pdf_path = tmp_path / "aurora_research.pdf"
    _create_sample_pdf(pdf_path)
    with sync_playwright() as playwright:
        browser = playwright.chromium.launch(headless=True)
        page: Page = browser.new_page(viewport={"width": 1440, "height": 1000})
        page.goto(live_server, wait_until="domcontentloaded")
        page.get_by_text("Upload Custom Research Paper", exact=True).click()
        page.locator('input[type="file"]').set_input_files(str(pdf_path))
        page.get_by_text("Indexed aurora_research.pdf from disk with page-level provenance.", exact=True).wait_for(timeout=60_000)
        page.locator('textarea[aria-label="Research question"]').last.fill("What recall was reported for coastal flood-risk classification?")
        page.get_by_role("button", name="Query uploaded paper", exact=True).click()
        page.get_by_text("Technical Breakdown", exact=True).wait_for(timeout=60_000)
        body = page.locator("body").inner_text()
        assert "Executive Insights" in body
        assert "Technical Breakdown" in body
        assert "Custom Uploaded Document" in body
        assert "Page 1" in body
        executive = body.split("Executive Insights", 1)[1].split("Technical Breakdown", 1)[0]
        technical = body.split("Technical Breakdown", 1)[1]
        assert executive.strip() != technical.strip()
        assert "94%" in technical or "94" in technical
        browser.close()
