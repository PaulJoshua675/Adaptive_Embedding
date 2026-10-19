from __future__ import annotations

import argparse
from concurrent.futures import ThreadPoolExecutor, as_completed
import json
import logging
import random
import re
import time
from pathlib import Path
from typing import Iterable, List

import requests

try:
    import fitz  # type: ignore
except ImportError:  # pragma: no cover
    fitz = None

try:
    import arxiv  # type: ignore
except ImportError as exc:  # pragma: no cover
    raise SystemExit("Install the arxiv package with: pip install arxiv") from exc

LOGGER = logging.getLogger("arxiv_corpus")
DEFAULT_CATEGORIES = ("cs.LG", "cs.CV", "cs.CL", "cs.AI", "cs.IR")


def _valid_pdf(path: Path) -> bool:
    if not path.exists() or path.stat().st_size < 1024:
        return False
    try:
        with path.open("rb") as handle:
            if handle.read(5) != b"%PDF-":
                return False
        if fitz is not None:
            with fitz.open(path) as document:
                return len(document) > 0
        return True
    except Exception:
        return False


def _safe_name(result: object) -> str:
    identifier = str(getattr(result, "entry_id", "paper")).rstrip("/").rsplit("/", 1)[-1]
    identifier = re.sub(r"v\d+$", "", identifier)
    return f"{identifier}.pdf"


def _paper_metadata(result: object) -> dict[str, object]:
    identifier = str(getattr(result, "entry_id", "")).rstrip("/").rsplit("/", 1)[-1]
    identifier = re.sub(r"v\d+$", "", identifier)
    categories = getattr(result, "categories", ()) or ()
    return {
        "paper_id": identifier,
        "title": str(getattr(result, "title", "")).strip(),
        "categories": sorted({str(category) for category in categories}),
    }


def _write_metadata(target: Path, records: dict[str, dict[str, object]]) -> None:
    metadata_path = target / "arxiv_metadata.json"
    existing: dict[str, dict[str, object]] = {}
    if metadata_path.exists():
        try:
            existing = json.loads(metadata_path.read_text(encoding="utf-8"))
        except (OSError, json.JSONDecodeError):
            LOGGER.warning("Could not read existing metadata manifest at %s", metadata_path)
    existing.update(records)
    metadata_path.write_text(json.dumps(existing, indent=2, ensure_ascii=False) + "\n", encoding="utf-8")


def _download_pdf(url: str, destination: Path, retries: int = 5, timeout: float = 45.0) -> bool:
    for attempt in range(retries):
        try:
            response = requests.get(url, timeout=timeout, headers={"User-Agent": "adaptive-rag-corpus/1.0"})
            response.raise_for_status()
            payload = response.content
            if not payload.startswith(b"%PDF-"):
                raise ValueError("downloaded response is not a PDF")
            temporary = destination.with_suffix(destination.suffix + ".part")
            temporary.write_bytes(payload)
            temporary.replace(destination)
            if _valid_pdf(destination):
                return True
            destination.unlink(missing_ok=True)
            raise ValueError("PDF failed structural validation")
        except Exception as exc:
            if attempt == retries - 1:
                LOGGER.warning("Failed %s: %s", url, exc)
                return False
            delay = min(60.0, 2.0 ** attempt) + random.uniform(0.0, 0.5)
            time.sleep(delay)
    return False


def download_corpus(
    output_dir: str | Path,
    categories: Iterable[str] = DEFAULT_CATEGORIES,
    papers_per_category: int = 50,
    max_results: int | None = 200,
    workers: int = 8,
) -> List[Path]:
    target = Path(output_dir)
    target.mkdir(parents=True, exist_ok=True)
    category_list = tuple(categories)
    client = arxiv.Client(page_size=100, delay_seconds=3.0, num_retries=5)
    candidates: List[tuple[str, Path, str]] = []
    metadata_records: dict[str, dict[str, object]] = {}
    seen: set[str] = set()
    total_limit = max_results or papers_per_category * len(category_list)

    for category in category_list:
        search = arxiv.Search(
            query=f"cat:{category}",
            max_results=papers_per_category,
            sort_by=arxiv.SortCriterion.SubmittedDate,
        )
        try:
            results = client.results(search)
            for result in results:
                identifier = str(getattr(result, "entry_id", ""))
                metadata = _paper_metadata(result)
                if metadata["paper_id"]:
                    paper_id = str(metadata["paper_id"])
                    if paper_id in metadata_records:
                        prior_categories = set(metadata_records[paper_id].get("categories", []))
                        prior_categories.update(metadata["categories"])
                        metadata["categories"] = sorted(prior_categories)
                    metadata_records[paper_id] = metadata
                if identifier in seen:
                    continue
                seen.add(identifier)
                destination = target / _safe_name(result)
                if destination.exists() and _valid_pdf(destination):
                    continue
                pdf_url = str(getattr(result, "pdf_url", ""))
                if pdf_url:
                    candidates.append((pdf_url, destination, category))
        except Exception as exc:
            LOGGER.warning("Category %s failed: %s", category, exc)

    _write_metadata(target, metadata_records)

    downloaded: List[Path] = []
    with ThreadPoolExecutor(max_workers=max(1, workers)) as executor:
        future_categories = {
            executor.submit(_download_pdf, url, destination): (destination, category)
            for url, destination, category in candidates
        }
        for future in as_completed(future_categories):
            destination, category = future_categories[future]
            try:
                if future.result():
                    downloaded.append(destination)
                    LOGGER.info("Downloaded %s (%s)", destination.name, category)
            except Exception as exc:  # Defensive: workers should report failures as False.
                LOGGER.warning("Download failed for %s: %s", destination.name, exc)
    return downloaded[:total_limit]


def main() -> None:
    parser = argparse.ArgumentParser(description="Download a resilient multi-domain arXiv PDF corpus")
    parser.add_argument("--output-dir", default="dataset")
    parser.add_argument("--categories", nargs="+", default=list(DEFAULT_CATEGORIES))
    parser.add_argument("--papers-per-category", type=int, default=50)
    parser.add_argument("--max-results", type=int, default=200)
    parser.add_argument("--workers", type=int, default=8)
    parser.add_argument("--log-level", default="INFO")
    args = parser.parse_args()
    logging.basicConfig(level=getattr(logging, args.log_level.upper(), logging.INFO), format="%(levelname)s %(message)s")
    paths = download_corpus(
        args.output_dir,
        args.categories,
        args.papers_per_category,
        args.max_results,
        args.workers,
    )
    print(f"Validated and downloaded {len(paths)} new paper(s) into {Path(args.output_dir).resolve()}")


if __name__ == "__main__":
    main()
