"""PDF and corpus ingestion entry points used by the indexing pipeline."""

from __future__ import annotations

from pathlib import Path
from typing import Dict, List

from .preprocessing import extract_documents_with_pages, process_custom_pdf_path


def ingest_corpus(
    root_dir: str | Path,
    figures_dir: str | Path | None = None,
) -> List[Dict[str, object]]:
    """Extract corpus documents with page-scoped figure metadata."""
    return extract_documents_with_pages(root_dir, figures_dir)


def ingest_pdf(
    file_path: str | Path,
    paper_id: str,
    figures_dir: str | Path,
) -> List[Dict[str, object]]:
    """Extract one uploaded PDF and attach its figure metadata to page records."""
    return process_custom_pdf_path(str(file_path), paper_id, figures_dir)
