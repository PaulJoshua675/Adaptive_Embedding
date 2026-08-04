from __future__ import annotations

import re
import ssl
import shutil
from pathlib import Path
from typing import Iterable, List
from urllib.request import urlopen


def normalize_arxiv_id(arxiv_id: str) -> str:
    value = str(arxiv_id).strip()
    if not value:
        raise ValueError("arXiv identifier cannot be empty")

    if value.lower().startswith("arxiv:"):
        value = value[6:]

    if value.startswith(("http://", "https://")):
        match = re.search(r"/(?:abs|pdf)/([^/?#]+)", value)
        if match:
            value = match.group(1)
        else:
            value = value.rstrip("/").rsplit("/", 1)[-1]

    value = value.strip().replace(" ", "")
    value = re.sub(r"v\d+$", "", value)
    return value


def build_arxiv_pdf_url(arxiv_id: str) -> str:
    normalized = normalize_arxiv_id(arxiv_id)
    return f"https://arxiv.org/pdf/{normalized}.pdf"


def download_arxiv_paper(arxiv_id: str, output_dir: str | Path, filename: str | None = None) -> Path:
    output_path = Path(output_dir)
    output_path.mkdir(parents=True, exist_ok=True)

    normalized = normalize_arxiv_id(arxiv_id)
    destination_name = filename or f"{normalized}.pdf"
    destination = output_path / destination_name

    if destination.exists():
        return destination

    urls = [build_arxiv_pdf_url(normalized), f"https://export.arxiv.org/pdf/{normalized}"]
    last_error: Exception | None = None
    context = ssl._create_unverified_context()
    for url in urls:
        try:
            with urlopen(url, context=context) as response, destination.open("wb") as handle:
                shutil.copyfileobj(response, handle)
            return destination
        except Exception as exc:  # pragma: no cover - network-dependent
            last_error = exc

    raise RuntimeError(f"Failed to download arXiv paper {arxiv_id}: {last_error}")


def download_arxiv_papers(arxiv_ids: Iterable[str], output_dir: str | Path) -> List[Path]:
    return [download_arxiv_paper(arxiv_id, output_dir) for arxiv_id in arxiv_ids]
