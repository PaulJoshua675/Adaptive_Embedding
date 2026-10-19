from __future__ import annotations

import re

import pytest

pytestmark = pytest.mark.integration

pytest.importorskip("playwright.sync_api")
from playwright.sync_api import Page, sync_playwright

from config import DOMAIN_EXAMPLE_QUESTIONS


def _select_option(page: Page, widget_index: int, option_pattern: str) -> None:
    page.get_by_test_id("stSelectbox").nth(widget_index).locator('[role="combobox"]').click()
    page.get_by_role("option", name=re.compile(option_pattern)).click()


def test_domain_example_returns_relevant_answer_without_unlinked_images(streamlit_app_url: str) -> None:
    domain = "cs.CV"
    question = DOMAIN_EXAMPLE_QUESTIONS[domain][0]
    with sync_playwright() as playwright:
        browser = playwright.chromium.launch(headless=True)
        page = browser.new_page(viewport={"width": 1440, "height": 1000})
        page.goto(streamlit_app_url, wait_until="domcontentloaded")
        page.get_by_text("Browse Central Repository", exact=True).wait_for()

        _select_option(page, 0, r"^English$")
        _select_option(page, 1, r"^cs\.CV")
        page.get_by_role("button", name=question, exact=True).click()
        page.get_by_text(
            re.compile(r"Overview & Core Context|Information not found in the corpus"),
            exact=False,
        ).wait_for(timeout=120_000)

        assert page.locator('textarea[aria-label="Research question"]').input_value() == question
        body = page.locator("body").inner_text()
        answer_marker = next(
            marker
            for marker in ("Overview & Core Context", "Information not found in the corpus")
            if marker in body
        )
        answer = (
            answer_marker
            if answer_marker == "Information not found in the corpus"
            else body.split(answer_marker, 1)[1].split("Retrieval evaluation", 1)[0]
        )
        assert answer.strip(), "The answer section must contain generated text."
        relevant_terms = ("vision", "transformer", "dense", "prediction")
        if "Information not found in the corpus" not in answer_marker:
            assert sum(term in answer.lower() for term in relevant_terms) >= 2, (
                "A substantive response should reuse multiple key terms from the selected question."
            )

        images = page.locator('[data-testid="stImage"] img')
        assert images.count() == 0, "A non-visual query must not display unrelated figures."
        browser.close()
