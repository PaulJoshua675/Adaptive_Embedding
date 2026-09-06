from __future__ import annotations

import os
from pathlib import Path

import pytest

pytest.importorskip("playwright.sync_api")
from playwright.sync_api import Page, sync_playwright


BASE_URL = os.getenv("RAG_TEST_URL", "http://localhost:8501")
REPORT_PATH = Path(__file__).with_name("pipeline_report.txt")


def _ask(page: Page, query: str) -> None:
    page.locator('textarea[aria-label="Research question"]').fill(query)
    page.get_by_role("button", name="Retrieve grounded answer", exact=True).click()
    page.get_by_text("Technical Breakdown", exact=True).wait_for(timeout=60_000)


def test_indexing_prompt_and_figure_regression() -> None:
    with sync_playwright() as playwright:
        browser = playwright.chromium.launch(headless=True)
        page = browser.new_page(viewport={"width": 1440, "height": 1000})
        page.goto(BASE_URL, wait_until="domcontentloaded")
        page.get_by_text("Browse Central Repository", exact=True).wait_for()

        _ask(page, "What is the technical breakdown of the methodology section?")
        response_text = page.locator("body").inner_text()
        technical_text = response_text.split("Technical Breakdown", 1)[-1]
        assert "UNSECTIONED" not in technical_text.upper()
        assert "Retrieved and synthesized passages" not in technical_text
        assert len(technical_text.split()) >= 25

        _ask(page, "Show Figure 45")
        figure = page.locator('img[src*="static/extracted_figures/"]').last
        figure.wait_for(timeout=60_000)
        assert figure.get_attribute("src")

        REPORT_PATH.write_text(
            "Pipeline regression report\n\n"
            f"URL: {BASE_URL}\n"
            "UNSECTIONED absent: yes\n"
            "Technical explanation present: yes\n"
            f"Figure image source: {figure.get_attribute('src')}\n",
            encoding="utf-8",
        )
        assert REPORT_PATH.exists()
        browser.close()