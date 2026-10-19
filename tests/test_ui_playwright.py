from __future__ import annotations

import os
import re
import socket
import subprocess
import sys
import time
from pathlib import Path
from urllib.error import URLError
from urllib.request import urlopen

import pytest

pytestmark = pytest.mark.integration

pytest.importorskip("playwright.sync_api")
from playwright.sync_api import sync_playwright

from config import DOMAIN_EXAMPLE_QUESTIONS


ROOT = Path(__file__).resolve().parents[1]
HEADINGS = (
    "Overview & Core Context",
    "Methodology & Architecture",
    "Experimental Results",
    "Limitations & Challenges",
    "Practical Implications",
)


def _available_port() -> int:
    with socket.socket() as listener:
        listener.bind(("127.0.0.1", 0))
        return int(listener.getsockname()[1])


def _wait_for_server(process: subprocess.Popen, url: str, log_path: Path) -> None:
    deadline = time.monotonic() + 90
    while time.monotonic() < deadline:
        if process.poll() is not None:
            raise RuntimeError(f"Streamlit exited before becoming ready:\n{log_path.read_text(encoding='utf-8')}")
        try:
            with urlopen(url, timeout=1):
                return
        except (OSError, URLError):
            time.sleep(0.5)
    raise RuntimeError(f"Streamlit did not become ready at {url}:\n{log_path.read_text(encoding='utf-8')}")


def test_local_streamlit_returns_five_detailed_corpus_grounded_sections(tmp_path: Path) -> None:
    port = _available_port()
    url = f"http://127.0.0.1:{port}"
    log_path = tmp_path / "streamlit.log"
    environment = os.environ.copy()
    environment["STREAMLIT_BROWSER_GATHER_USAGE_STATS"] = "false"
    with log_path.open("w", encoding="utf-8") as log_file:
        process = subprocess.Popen(
            [
                sys.executable,
                "-m",
                "streamlit",
                "run",
                str(ROOT / "app.py"),
                "--server.address=127.0.0.1",
                f"--server.port={port}",
                "--server.headless=true",
            ],
            cwd=ROOT,
            env=environment,
            stdout=log_file,
            stderr=subprocess.STDOUT,
        )
        try:
            _wait_for_server(process, url, log_path)
            question = DOMAIN_EXAMPLE_QUESTIONS["cs.LG"][1]
            with sync_playwright() as playwright:
                browser = playwright.chromium.launch(headless=True)
                try:
                    page = browser.new_page(viewport={"width": 1440, "height": 1000})
                    page.goto(url, wait_until="domcontentloaded")
                    page.get_by_text("Browse Central Repository", exact=True).wait_for(timeout=30_000)
                    body = page.locator("body").inner_text()
                    for domain_label, description in (
                        ("Machine Learning and Deep Learning", "Transformers, RLHF, optimization, retrieval"),
                        ("Image Processing", "CNNs, diffusion models, image understanding"),
                        ("Hardware Architecture", "Distributed training, edge AI, systems"),
                    ):
                        assert domain_label in body
                        assert description in body
                    assert f"Central corpus: {ROOT / 'dataset'}" in body

                    domain_selector = page.get_by_test_id("stSelectbox").nth(1).locator('[role="combobox"]')
                    domain_selector.click()
                    page.get_by_role("option", name=re.compile(r"^cs\.LG")).click()
                    page.get_by_role("button", name=question, exact=True).click()
                    page.get_by_role("heading", name=re.compile(re.escape(HEADINGS[0]))).wait_for(timeout=60_000)

                    body = page.locator("body").inner_text()
                    answer_start = body.index(HEADINGS[0])
                    answer = body[answer_start:]
                    for heading in HEADINGS:
                        assert answer.count(heading) == 1, f"Expected exactly one {heading!r} heading."
                    assert "Verified Source Citations" in answer
                    assert "2203.02155" in answer
                    answer = answer.split("Verified Source Citations", 1)[0]

                    for index, heading in enumerate(HEADINGS):
                        start = answer.index(heading) + len(heading)
                        end = answer.index(HEADINGS[index + 1], start) if index + 1 < len(HEADINGS) else len(answer)
                        section = answer[start:end]
                        word_count = len(re.findall(r"\b[\w'-]+\b", section))
                        assert word_count >= 100, f"{heading} contains only {word_count} words."
                    assert page.locator("ul li").count() >= len(HEADINGS)

                    assert not re.search(r"\bPoint\s+\d+\s*:", answer, flags=re.I)
                    assert not re.search(
                        r"\b1e-4\s+1e-3\s+1e-2\s+1\s+0\s+20\s+40\s+60\b",
                        answer,
                    )
                    assert question in page.locator('textarea[aria-label="Research question"]').input_value()
                finally:
                    browser.close()
        finally:
            process.terminate()
            try:
                process.wait(timeout=10)
            except subprocess.TimeoutExpired:
                process.kill()
                process.wait(timeout=10)
