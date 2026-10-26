from __future__ import annotations

import sys
import os
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

import re
import socket
import subprocess
import time
from pathlib import Path
from urllib.error import URLError
from urllib.request import urlopen

import pytest

pytestmark = pytest.mark.integration

pytest.importorskip("playwright.sync_api")
from playwright.sync_api import Page, TimeoutError as PlaywrightTimeoutError
from playwright.sync_api import sync_playwright

from config import DOMAIN_EXAMPLE_QUESTIONS


ROOT = Path(__file__).resolve().parents[1]
DEFAULT_STREAMLIT_URL = "http://localhost:8501"
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


def _server_is_available(url: str) -> bool:
    try:
        with urlopen(url, timeout=2):
            return True
    except (OSError, URLError):
        return False


def _response_elements(page: Page) -> list[dict[str, str]]:
    return page.evaluate(
        """() => {
            const nodes = [...document.querySelectorAll('[data-testid="stMarkdownContainer"]')];
            const answerPattern = /Overview & Core Context|Information not found in the corpus/i;
            const answers = nodes.filter(node => answerPattern.test(node.innerText));
            const inspected = answers.length ? answers : nodes.slice(-5);
            return inspected.map(node => ({
                text: node.innerText,
                html: node.outerHTML
            }));
        }"""
    )


def _relevant_server_logs(server_log: str) -> str:
    relevant_lines = [
        line
        for line in server_log.splitlines()
        if re.search(r"retriev|faiss|vector|groq|generat|error|exception|traceback|warning", line, re.I)
    ]
    sanitized = "\n".join(relevant_lines[-50:])
    return re.sub(
        r"(?i)(api[_ -]?key|authorization|bearer)(\s*[:=]\s*|\s+)[^\s,;]+",
        r"\1\2<redacted>",
        sanitized,
    )


def _diagnostic_category(
    server_log: str,
    exception_texts: list[str],
    error_alerts: list[str],
    page_errors: list[str],
    answer_text: str,
    retrieved_count: int | None,
) -> str:
    combined_log = server_log.lower()
    if re.search(
        r"retrieval index is not ready|embedding index is unavailable|unable to retrieve from index|"
        r"(?:faiss|vector store|vector index).{0,80}(?:error|exception|fail|missing|unavailable|not found)",
        combined_log,
    ) or retrieved_count == 0:
        return "Vector index lookup / retrieval"
    if re.search(
        r"groq.{0,100}(?:fail|error|empty|parse|json|completion)|"
        r"(?:parse|json|decode).{0,100}(?:groq|response|completion)",
        combined_log,
    ):
        return "Groq response parsing / generation"
    if exception_texts or error_alerts:
        return "Streamlit/backend exception"
    if page_errors:
        return "UI rendering / answer container"
    if not answer_text:
        return "UI rendering / answer container"
    return "No failure detected"


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


def test_streamlit_query_emits_rag_diagnostics(tmp_path: Path) -> None:
    url = DEFAULT_STREAMLIT_URL
    log_path = tmp_path / "streamlit-diagnostics.log"
    process: subprocess.Popen | None = None

    if not _server_is_available(url):
        environment = os.environ.copy()
        environment["STREAMLIT_BROWSER_GATHER_USAGE_STATS"] = "false"
        log_file = log_path.open("w", encoding="utf-8")
        try:
            process = subprocess.Popen(
                [
                    sys.executable,
                    "-m",
                    "streamlit",
                    "run",
                    str(ROOT / "app.py"),
                    "--server.address=127.0.0.1",
                    "--server.port=8501",
                    "--server.headless=true",
                ],
                cwd=ROOT,
                env=environment,
                stdout=log_file,
                stderr=subprocess.STDOUT,
            )
            _wait_for_server(process, url, log_path)
        except Exception:
            log_file.close()
            if process is not None and process.poll() is None:
                process.terminate()
                process.wait(timeout=10)
            raise
    else:
        log_file = None

    console_logs: list[str] = []
    network_responses: list[str] = []
    failed_requests: list[str] = []
    exception_texts: list[str] = []
    response_elements: list[dict[str, str]] = []
    retrieved_count: int | None = None
    page_errors: list[str] = []
    browser = None
    try:
        with sync_playwright() as playwright:
            browser = playwright.chromium.launch(headless=True)
            page = browser.new_page(viewport={"width": 1440, "height": 1000})
            page.on("console", lambda message: console_logs.append(f"{message.type}: {message.text}"))
            page.on("pageerror", lambda error: page_errors.append(str(error)))
            page.on(
                "response",
                lambda response: network_responses.append(f"{response.status} {response.url}"),
            )
            page.on(
                "requestfailed",
                lambda request: failed_requests.append(
                    f"{request.method} {request.url}: {request.failure}"
                ),
            )

            page.goto(url, wait_until="domcontentloaded", timeout=60_000)
            question = "Summarize the document"
            page.locator('textarea[aria-label="Research question"]').fill(question)
            page.get_by_role("button", name="Retrieve grounded answer", exact=True).click()

            timeout_ms = int(os.getenv("RAG_PLAYWRIGHT_TIMEOUT_MS", "180000"))
            try:
                page.wait_for_function(
                    """() => {
                        const body = document.body.innerText;
                        const hasAnswer = /Overview & Core Context|Information not found in the corpus/i.test(body);
                        const hasException = Boolean(document.querySelector('[data-testid="stException"]'));
                        const hasAlert = Boolean(document.querySelector('[data-testid="stAlert"]'));
                        return hasAnswer || hasException || hasAlert;
                    }""",
                    timeout=timeout_ms,
                )
            except PlaywrightTimeoutError as exc:
                console_logs.append(f"wait timeout: {exc}")

            page.wait_for_timeout(1000)
            response_elements = _response_elements(page)
            exception_texts = page.locator('[data-testid="stException"]').all_inner_texts()
            alert_texts = page.locator('[data-testid="stAlert"]').all_inner_texts()
            error_alerts = [
                text
                for text in alert_texts
                if re.search(r"\b(error|failed|unable|exception|traceback)\b", text, re.I)
            ]
            body_text = page.locator("body").inner_text()
            retrieval_match = re.search(r"Retrieved passages:\s*(\d+)", body_text, re.I)
            if retrieval_match:
                retrieved_count = int(retrieval_match.group(1))

            server_log = log_path.read_text(encoding="utf-8", errors="replace") if log_path.exists() else ""
            answer_text = "\n".join(
                element["text"]
                for element in response_elements
                if re.search(
                    r"Overview & Core Context|Information not found in the corpus",
                    element["text"],
                    re.I,
                )
            ).strip()
            category = _diagnostic_category(
                _relevant_server_logs(server_log),
                exception_texts,
                error_alerts,
                page_errors,
                answer_text,
                retrieved_count,
            )
            print(
                "\nRAG Playwright diagnostic report\n"
                f"URL: {url}\n"
                f"Query submitted: {question}\n"
                f"Likely failure area: {category}\n"
                f"Retrieved passages: {retrieved_count if retrieved_count is not None else 'not displayed'}\n"
                f"Answer text: {answer_text or '<empty>'}\n"
                f"Streamlit errors / exceptions: {exception_texts or '<none>'}\n"
                f"Streamlit alert banners: {alert_texts or '<none>'}\n"
                f"Streamlit error alerts: {error_alerts or '<none>'}\n"
                f"Browser page errors: {page_errors or '<none>'}\n"
                f"Browser console: {console_logs or '<none>'}\n"
                f"Network responses ({len(network_responses)}): {network_responses}\n"
                f"Failed requests: {failed_requests or '<none>'}\n"
                f"Server log diagnostics: {_relevant_server_logs(server_log) or '<unavailable; connected to existing server>'}\n"
                "Answer DOM (HTML):\n"
                + (
                    "\n".join(element["html"] for element in response_elements)
                    if response_elements
                    else "<no answer markdown container found>"
                )
            )

            assert not exception_texts, "Streamlit displayed an error or exception banner."
            assert not error_alerts, "Streamlit displayed an error banner."
            assert not page_errors, "The browser raised a page-level JavaScript error."
            assert retrieved_count is not None, "The retrieval count was not rendered in the UI."
            assert retrieved_count > 0, (
                "The query completed without retrieving any passages."
            )
            assert answer_text, "The answer container remained empty after submitting the query."
    finally:
        if browser is not None:
            browser.close()
        if process is not None and process.poll() is None:
            process.terminate()
            try:
                process.wait(timeout=10)
            except subprocess.TimeoutExpired:
                process.kill()
                process.wait(timeout=10)
        if log_file is not None:
            log_file.close()
