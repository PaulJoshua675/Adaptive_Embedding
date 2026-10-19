from __future__ import annotations

import re
from pathlib import Path
from typing import Dict, List

try:
    import fitz  # type: ignore
except ImportError:  # pragma: no cover - optional dependency
    fitz = None

try:
    from PIL import Image  # type: ignore
except ImportError:  # pragma: no cover - optional dependency
    Image = None

try:
    import pytesseract  # type: ignore
except ImportError:  # pragma: no cover - optional dependency
    pytesseract = None

try:
    from pypdf import PdfReader  # type: ignore
except ImportError:  # pragma: no cover - optional dependency
    PdfReader = None

from .utils import iter_supported_files, load_text


_NUMBER = re.compile(r"(?<![A-Za-z])[-+]?(?:\d+(?:\.\d*)?|\.\d+)(?:[eE][-+]?\d+)?%?(?![A-Za-z])")
_AXIS_SEQUENCE = re.compile(
    r"(?<!\S)(?:[-+]?(?:\d+(?:\.\d*)?|\.\d+)(?:[eE][-+]?\d+)?%?(?:\s+|,\s*)){2}"
    r"[-+]?(?:\d+(?:\.\d*)?|\.\d+)(?:[eE][-+]?\d+)?%?(?!\S)"
)
_FIGURE_LABEL = re.compile(r"^(?:figure|fig\.?)\s*\d+\s*[:.]?$", re.I)
_PLOT_TERMS = re.compile(r"\b(?:axis|validation|training|reward|epoch|iteration|precision|recall|accuracy|f1|bleu|loss)\b", re.I)


def _is_noise_line(line: str) -> bool:
    stripped = line.strip()
    if not stripped:
        return False
    if _FIGURE_LABEL.fullmatch(stripped):
        return True

    tokens = re.findall(r"\S+", stripped)
    numbers = _NUMBER.findall(stripped)
    if len(numbers) >= 3:
        text_tokens = [token for token in tokens if not _NUMBER.fullmatch(token.strip("()[]{}.,:;"))]
        if len(text_tokens) <= 2:
            return True
        if _PLOT_TERMS.search(stripped) and len(numbers) >= 4:
            return True
        if len(numbers) / len(tokens) >= 0.55:
            return True
    return bool(re.fullmatch(r"[-+]?(?:\d+(?:\.\d*)?|\.\d+)(?:[eE][-+]?\d+)?", stripped))


def clean_extracted_text(text: str) -> str:
    """Remove isolated plot labels, numeric axes, and OCR-only table rows."""
    cleaned_lines = []
    for source_line in text.splitlines():
        if _is_noise_line(source_line):
            continue
        line = _AXIS_SEQUENCE.sub("", source_line).strip()
        if line and not _is_noise_line(line):
            cleaned_lines.append(line)
    return re.sub(r"\s+", " ", " ".join(cleaned_lines)).strip()


def _extract_image_ocr_text(doc: object, page: object, page_number: int) -> str:
    if pytesseract is None or Image is None:
        return ""

    image_texts: List[str] = []
    for image_index, image_info in enumerate(page.get_images(full=True), start=1):
        try:
            xref = image_info[0]
            pix = fitz.Pixmap(doc, xref)
            if pix.n - pix.alpha >= 4:
                mode = "RGB"
            elif pix.n == 1:
                mode = "L"
            else:
                mode = "RGB"
            image = Image.frombytes(mode, [pix.width, pix.height], pix.samples)
            extracted = pytesseract.image_to_string(image).strip()
            if extracted:
                image_texts.append(f"[Image {image_index} on page {page_number}] {extracted}")
            pix = None
        except Exception:
            continue

    return "\n".join(image_texts)


def extract_text_from_pdf(path: str | Path) -> str:
    if fitz is not None:
        doc = fitz.open(path)
        try:
            sections: List[str] = []
            for page_number, page in enumerate(doc, start=1):
                text = (page.get_text() or "").strip()
                image_text = _extract_image_ocr_text(doc, page, page_number)
                page_parts = [part for part in [text, image_text] if part]
                if page_parts:
                    sections.append("\n".join(page_parts))
            return clean_extracted_text("\n\n".join(sections))
        finally:
            doc.close()

    if PdfReader is not None:
        reader = PdfReader(str(path))
        return clean_extracted_text("\n".join(page.extract_text() or "" for page in reader.pages))

    return ""


def _load_pdf_fallback_text(file_path: Path, root_dir: Path) -> str:
    candidates = [
        root_dir / "extracted_text" / f"{file_path.stem}.txt",
        root_dir.parent / "extracted_text" / f"{file_path.stem}.txt",
    ]
    for candidate in candidates:
        if candidate.exists():
            return load_text(candidate)
    return ""


def _split_multi_document_text(file_path: Path, text: str) -> List[Dict[str, str]]:
    pattern = re.compile(r"(?im)^---\s*PAPER\s*(\d+)\s*---\s*$")
    matches = list(pattern.finditer(text))
    if not matches:
        return [
            {
                "id": str(file_path.stem),
                "source": str(file_path),
                "text": clean_extracted_text(text),
            }
        ]

    documents: List[Dict[str, str]] = []
    for index, match in enumerate(matches):
        start = match.end()
        end = matches[index + 1].start() if index + 1 < len(matches) else len(text)
        section_text = clean_extracted_text(text[start:end])
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
    root_path = Path(root_dir)
    for file_path in iter_supported_files(root_path):
        if file_path.suffix.lower() == ".pdf":
            text = extract_text_from_pdf(file_path)
            text = clean_extracted_text(text)
            if not text:
                text = _load_pdf_fallback_text(file_path, root_path)
        else:
            text = load_text(file_path)

        if not text.strip():
            continue

        documents.extend(_split_multi_document_text(file_path, text))
    return documents
