from __future__ import annotations

import json
import os
import re
from pathlib import Path
from typing import Dict, List, Tuple

try:
    import fitz  # type: ignore
except ImportError:  # pragma: no cover
    fitz = None

try:
    import pytesseract  # type: ignore
except ImportError:  # pragma: no cover
    pytesseract = None

from .utils import iter_supported_files, load_text

__all__ = ["process_custom_pdf_path", "extract_page_images", "query_requests_visuals", "extract_pdf_pages"]

_VISUAL_QUERY = re.compile(r"\b(?:image|figure|fig\.?|chart|plot|graph|diagram|architecture|visual|table|picture)\b", re.I)
_FIGURE_CAPTION = re.compile(
    r"(?im)^\s*(Figure\s+\d+|Fig\.\s*\d+)\s*[:.]?\s*([^\r\n]*)"
)
_SECTION_NAMES = (
    "ABSTRACT|INTRODUCTION|BACKGROUND|METHODOLOGY|RELATED WORK|EVALUATION|"
    "EXPERIMENTAL RESULTS|CONCLUSION|FUTURE WORK|REFERENCES"
)
_SECTION_HEADING = re.compile(
    rf"^\s*(?:(?:\d+(?:\.\d+)*|[IVX]+)[.)]?\s+)?({_SECTION_NAMES})\s*[:.]?\s*$",
    re.I,
)
_QUERY_STOP_WORDS = frozenset(
    "a an and are can could find for from get how i me of please retrieve show tell the to what which with you your".split()
)
_SECTION_QUERY = re.compile(
    r"\b(?:show|find|get|retrieve|explain|summarize|give me)?\s*"
    r"(?:the\s+)?(abstract|introduction|background|related work|methods?|methodology|experiments?|results?|discussion|conclusions?|references)\b",
    re.I,
)


def _normalise(value: str) -> str:
    return " ".join(value.split())


def trim_query_tokens(query: str) -> str:
    """Remove common grammatical filler before local vector encoding."""
    tokens = re.findall(r"[A-Za-z0-9]+(?:[-'][A-Za-z0-9]+)?", query.lower())
    retained = [token for token in tokens if token not in _QUERY_STOP_WORDS]
    return " ".join(retained) or _normalise(query)


def section_query_name(query: str) -> str | None:
    """Return a normalized section name for direct metadata retrieval."""
    match = _SECTION_QUERY.search(_normalise(query))
    if not match:
        return None
    value = match.group(1).upper()
    aliases = {"METHOD": "METHODOLOGY", "METHODS": "METHODOLOGY", "CONCLUSIONS": "CONCLUSION"}
    return aliases.get(value, value)


def _title_reference(text: str, fallback: str) -> str:
    for line in text.splitlines():
        candidate = _normalise(line)
        if len(candidate) >= 8:
            return candidate[:240]
    return fallback.replace("_", " ").replace("-", " ").strip() or "Untitled research paper"


def _section_heading(line: str) -> str | None:
    candidate = _normalise(line).strip(" .:")
    if not candidate or len(candidate) > 120:
        return None
    match = _SECTION_HEADING.match(candidate)
    return match.group(1).upper() if match else None


def _split_text_by_sections(text: str, current_section: str = "ABSTRACT") -> Tuple[List[Dict[str, str]], str]:
    """Split one layout block and return its final section for the next block."""
    sections: List[Dict[str, str]] = []
    current_type = current_section
    current_lines: List[str] = []
    for line in text.splitlines():
        heading = _section_heading(line)
        if heading and current_lines:
            content = _normalise(" ".join(current_lines))
            if content:
                sections.append({"section_type": current_type, "text": content})
            current_lines = []
        if heading:
            current_type = heading
        else:
            current_lines.append(line)
    content = _normalise(" ".join(current_lines))
    if content:
        sections.append({"section_type": current_type, "text": content})
    if not sections and _normalise(text):
        sections.append({"section_type": current_type, "text": _normalise(text)})
    return sections, current_type


def split_text_by_sections(text: str, current_section: str = "ABSTRACT") -> List[Dict[str, str]]:
    """Split text at standard academic headings with inherited sections."""
    sections, _ = _split_text_by_sections(text, current_section)
    return sections


def _write_manifest(manifest: Dict[str, Dict[str, object]], figures_dir: Path) -> None:
    figures_dir.mkdir(parents=True, exist_ok=True)
    manifest_path = figures_dir / "figure_manifest.json"
    existing: Dict[str, Dict[str, object]] = {}
    if manifest_path.exists():
        try:
            loaded = json.loads(manifest_path.read_text(encoding="utf-8"))
            if isinstance(loaded, dict):
                existing = {
                    str(figure_id): dict(entry)
                    for figure_id, entry in loaded.items()
                    if isinstance(entry, dict)
                }
        except json.JSONDecodeError:
            existing = {}
    existing.update(manifest)
    manifest_path.write_text(json.dumps(existing, indent=2, ensure_ascii=True), encoding="utf-8")


def _caption_rect(page, figure_match: re.Match[str]) -> "fitz.Rect":
    """Find the exact text span rectangle for a caption label."""
    figure_label = _normalise(figure_match.group(1)).lower()
    for block in page.get_text("dict").get("blocks", []):
        for line in block.get("lines", []):
            spans = line.get("spans", [])
            line_text = _normalise(" ".join(str(span.get("text", "")) for span in spans))
            if figure_label in line_text.lower() and spans:
                line_rect = fitz.Rect(spans[0]["bbox"])
                for span in spans[1:]:
                    line_rect |= fitz.Rect(span["bbox"])
                return line_rect
    return fitz.Rect(0, 0, page.rect.width, min(page.rect.height, 1))


def _save_page_capture(page, caption_rect: "fitz.Rect", figure_path: Path) -> None:
    clip = fitz.Rect(0, 0, page.rect.width, min(page.rect.height, caption_rect.y1 + 20.0))
    page.get_pixmap(matrix=fitz.Matrix(2, 2), clip=clip, alpha=False).save(str(figure_path))


def _catalog_figures(document, pdf_path: Path, figures_dir: Path) -> Dict[str, Dict[str, object]]:
    manifest: Dict[str, Dict[str, object]] = {}
    figures_dir.mkdir(parents=True, exist_ok=True)
    for page in document:
        page_text = page.get_text("text") or ""
        captions = list(_FIGURE_CAPTION.finditer(page_text))
        if not captions:
            continue
        for figure_match in captions:
            figure_id = _normalise(figure_match.group(1))
            caption = _normalise(figure_match.group(2))
            if not caption:
                continue
            figure_number = re.search(r"\d+", figure_id)
            if not figure_number:
                continue
            figure_path = figures_dir / f"{pdf_path.stem}_fig_{figure_number.group(0)}.png"
            _save_page_capture(page, _caption_rect(page, figure_match), figure_path)
            manifest[figure_id] = {
                "caption": caption,
                "file_path": figure_path.as_posix(),
                "paper_id": pdf_path.stem,
                "page_number": int(page.number + 1),
            }
    return manifest


def _page_blocks_text(page) -> str:
    blocks = page.get_text("blocks") or []
    ordered_blocks = sorted(blocks, key=lambda block: (float(block[1]), float(block[0])))
    return "\n".join(
        str(block[4]).strip()
        for block in ordered_blocks
        if len(block) >= 5 and isinstance(block[4], str) and str(block[4]).strip()
    )


def _write_custom_page_records(
    document,
    file_path: Path,
    paper_id: str,
    figures_dir: Path,
) -> List[Dict[str, object]]:
    records: List[Dict[str, object]] = []
    current_section = "ABSTRACT"
    for page in document:
        page_number = int(page.number + 1)
        page_text = _page_blocks_text(page)
        sections, current_section = _split_text_by_sections(page_text, current_section)
        image_count = len(page.get_images(full=True))
        for section_index, section in enumerate(sections):
            text = str(section.get("text", "")).strip()
            if not text:
                continue
            records.append({
                "id": f"{paper_id}_page_{page_number}_section_{section_index}",
                "source": str(file_path),
                "text": text,
                "page_number": page_number,
                "section_type": str(section.get("section_type") or current_section),
                "title": _title_reference(page_text, paper_id),
                "image_count": image_count,
                "source_type": "upload",
            })
    return records


def process_custom_pdf_path(
    file_path: str,
    paper_id: str,
    figures_dir: str | Path = "vector_store/figures",
) -> List[Dict[str, object]]:
    """Process a serialized custom PDF from disk and return section-scoped records."""
    if fitz is None:
        raise RuntimeError("PyMuPDF is required to process PDF uploads.")
    source_path = Path(file_path)
    if not source_path.is_file():
        raise FileNotFoundError(f"Custom PDF does not exist: {source_path}")
    output_dir = Path(figures_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    try:
        doc = fitz.open(file_path)
    except Exception as exc:
        raise ValueError("The custom PDF is empty or corrupt.") from exc
    if len(doc) == 0:
        doc.close()
        raise ValueError("The custom PDF contains zero pages.")
    try:
        manifest: Dict[str, Dict[str, object]] = {}
        for page in doc:
            page_number = int(page.number + 1)
            page_text = page.get_text("text") or ""
            captions = list(_FIGURE_CAPTION.finditer(page_text))
            if not captions:
                continue
            snapshot_path = output_dir / f"{paper_id}_fig_{page_number}.png"
            page.get_pixmap(matrix=fitz.Matrix(2, 2), alpha=False).save(str(snapshot_path))
            for caption_match in captions:
                figure_id = _normalise(caption_match.group(1))
                caption = _normalise(caption_match.group(2))
                manifest[figure_id] = {
                    "caption": caption,
                    "file_path": snapshot_path.as_posix(),
                    "paper_id": paper_id,
                    "page_number": page_number,
                    "source": str(source_path),
                    "source_type": "upload",
                }
        _write_manifest(manifest, output_dir)
        return _write_custom_page_records(doc, source_path, paper_id, output_dir)
    finally:
        doc.close()


def _render_page_canvas(page, scan_path: Path) -> object:
    pixmap = page.get_pixmap(matrix=fitz.Matrix(2, 2), alpha=False)
    scan_path.parent.mkdir(parents=True, exist_ok=True)
    pixmap.save(str(scan_path))
    return pixmap


def _ocr_pixmap(pixmap: object) -> str:
    if pytesseract is not None:
        try:
            from PIL import Image  # type: ignore
            from io import BytesIO

            return pytesseract.image_to_string(Image.open(BytesIO(pixmap.tobytes("png"))))
        except Exception:
            pass
    return "[Scanned page visual content extracted from high-resolution page pixels.]"


def _extract_page_text(page, scan_path: Path | None = None) -> str:
    text = page.get_text("text") or ""
    if text.strip():
        return text
    pixmap = _render_page_canvas(page, scan_path) if scan_path is not None else page.get_pixmap(matrix=fitz.Matrix(2, 2), alpha=False)
    return _ocr_pixmap(pixmap)


def _extract_document_pages(document, pdf_path: Path, output_dir: Path, source_type: str) -> List[Dict[str, object]]:
    pages: List[Dict[str, object]] = []
    current_section = "DOCUMENT"
    for page in document:
        scan_path = output_dir / f"{pdf_path.stem}_page_{page.number + 1}_scan.png"
        raw_text = _extract_page_text(page, scan_path)
        image_count = len(page.get_images(full=True))
        page_sections, current_section = _split_text_by_sections(raw_text, current_section)
        for section_index, section in enumerate(page_sections):
            if section["text"] or image_count:
                pages.append({
                    "id": f"{pdf_path.stem}_page_{page.number + 1}_section_{section_index}",
                    "source": str(pdf_path),
                    "text": section["text"],
                    "page_number": int(page.number + 1),
                    "section_type": section["section_type"],
                    "title": _title_reference(raw_text, pdf_path.stem),
                    "image_count": image_count,
                    "source_type": source_type,
                })
    return [page for page in pages if str(page.get("text", "")).strip()]


def extract_pdf_pages(
    path: str | Path,
    figures_dir: str | Path | None = None,
    source_type: str = "central",
) -> List[Dict[str, object]]:
    """Extract section-scoped page records and build a caption-aware figure catalog."""
    if fitz is None:
        return []
    pdf_path = Path(path)
    configured_output = os.getenv("RAG_FIGURES_DIR")
    legacy_output = os.getenv("RAG_STATIC_DIR")
    default_output = (
        configured_output
        or (str(Path(legacy_output) / "extracted_figures") if legacy_output else "vector_store/figures")
    )
    output_dir = Path(figures_dir or default_output)
    pages: List[Dict[str, object]] = []
    try:
        with fitz.open(pdf_path) as document:
            _write_manifest(_catalog_figures(document, pdf_path, output_dir), output_dir)
            pages = _extract_document_pages(document, pdf_path, output_dir, source_type)
    except Exception:
        return []
    return [page for page in pages if str(page.get("text", "")).strip()]


def extract_documents_with_pages(root_dir: str | Path, figures_dir: str | Path | None = None) -> List[Dict[str, object]]:
    root_path = Path(root_dir)
    documents: List[Dict[str, object]] = []
    for file_path in iter_supported_files(root_path):
        if file_path.suffix.lower() == ".pdf":
            pages = extract_pdf_pages(file_path, figures_dir)
            if pages:
                documents.extend(pages)
                continue
            fallback = root_path / "extracted_text" / f"{file_path.stem}.txt"
            if not fallback.exists():
                fallback = root_path.parent / "extracted_text" / f"{file_path.stem}.txt"
            if fallback.exists():
                text = load_text(fallback)
                for index, section in enumerate(split_text_by_sections(text)):
                    documents.append({"id": f"{file_path.stem}_section_{index}", "source": str(file_path), "text": section["text"], "page_number": 1, "section_type": section["section_type"], "title": _title_reference(text, file_path.stem), "image_count": 0})
            continue
        text = load_text(file_path)
        for index, section in enumerate(split_text_by_sections(text)):
            if section["text"]:
                documents.append({"id": f"{file_path.stem}_section_{index}", "source": str(file_path), "text": section["text"], "page_number": 1, "section_type": section["section_type"], "title": _title_reference(text, file_path.stem), "image_count": 0})
    return documents


def query_requests_visuals(query: str) -> bool:
    return bool(_VISUAL_QUERY.search(query))


def extract_page_images(
    page_obj: object,
    paper_id: str,
    page_num: int,
    output_dir: str | Path = "vector_store/figures",
) -> List[Path]:
    """Render embedded images from a PyMuPDF page into the figure store.

    A PDF path is also accepted as ``page_obj`` so existing callers can use
    the same public function without owning a PyMuPDF document lifecycle.
    """
    if fitz is None or page_num < 1:
        return []
    target_dir = Path(output_dir)
    target_dir.mkdir(parents=True, exist_ok=True)
    outputs: List[Path] = []
    document = None
    page = page_obj
    if isinstance(page_obj, (str, Path)):
        source_path = Path(page_obj)
        if source_path.suffix.lower() != ".pdf" or not source_path.exists():
            return []
        document = fitz.open(source_path)
        if page_num > len(document):
            document.close()
            return []
        page = document[page_num - 1]
    if not hasattr(page, "get_images"):
        if document is not None:
            document.close()
        return []
    try:
        for image_index, image_info in enumerate(page.get_images(full=True), start=1):
            output_path = target_dir / f"{paper_id}_page_{page_num}_fig_{image_index}.png"
            if not output_path.exists():
                if document is not None:
                    fitz.Pixmap(document, int(image_info[0])).save(str(output_path))
                else:
                    pixmap = page.get_pixmap(matrix=fitz.Matrix(2, 2), alpha=False)
                    pixmap.save(str(output_path))
            outputs.append(output_path)
    finally:
        if document is not None:
            document.close()
    return outputs
