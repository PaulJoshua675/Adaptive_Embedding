from __future__ import annotations

import re
from pathlib import Path
from typing import Dict, List

try:
    import fitz  # type: ignore
except ImportError:  # pragma: no cover - optional dependency
    fitz = None

try:
    from pypdf import PdfReader  # type: ignore
except ImportError:  # pragma: no cover - optional dependency
    PdfReader = None

from .utils import iter_supported_files, load_text


def extract_text_from_pdf(path: str | Path) -> str:
    if fitz is not None:
        doc = fitz.open(path)
        try:
            return "\n".join(page.get_text() for page in doc)
        finally:
            doc.close()

    if PdfReader is not None:
        reader = PdfReader(str(path))
        return "\n".join(page.extract_text() or "" for page in reader.pages)

    return ""


def _split_multi_document_text(file_path: Path, text: str) -> List[Dict[str, str]]:
    pattern = re.compile(r"(?im)^---\s*PAPER\s*(\d+)\s*---\s*$")
    matches = list(pattern.finditer(text))
    if not matches:
        return [
            {
                "id": str(file_path.stem),
                "source": str(file_path),
                "text": text.strip(),
            }
        ]

    documents: List[Dict[str, str]] = []
    for index, match in enumerate(matches):
        start = match.end()
        end = matches[index + 1].start() if index + 1 < len(matches) else len(text)
        section_text = text[start:end].strip()
        if not section_text:
            continue
        paper_number = match.group(1).zfill(2)
        documents.append(
            {
                "id": f"{file_path.stem}_paper_{paper_number}",
                "source": str(file_path),
                "text": section_text,
            }
        )
    return documents


def extract_documents_from_path(root_dir: str | Path) -> List[Dict[str, str]]:
    documents: List[Dict[str, str]] = []
    for file_path in iter_supported_files(root_dir):
        if file_path.suffix.lower() == ".pdf":
            text = extract_text_from_pdf(file_path)
        else:
            text = load_text(file_path)

        if not text.strip():
            continue

        documents.extend(_split_multi_document_text(file_path, text))
    return documents
