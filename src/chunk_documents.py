from __future__ import annotations

import re
from typing import Dict, List

_DATASET_NAMES = (
    "TruthfulQA",
    "GSM8K",
    "MMLU",
    "GPQA",
    "ARC-Challenge",
    "ARC-Easy",
    "HellaSwag",
    "WinoGrande",
    "HumanEval",
    "MBPP",
    "BIG-bench",
    "SQuAD",
    "GLUE",
    "SuperGLUE",
    "ImageNet",
    "COCO",
    "CIFAR-10",
    "CIFAR-100",
    "WikiText",
    "WMT",
    "XSum",
)
_DATASET_PATTERN = re.compile(
    r"\b(?:" + "|".join(re.escape(name) for name in _DATASET_NAMES) + r")\b",
    re.I,
)
_INLINE_VISUAL_CAPTION = re.compile(
    r"\b(Figure|Fig\.?|Table|Tab\.?)\s*(\d+[A-Za-z]?)\s*[:.]\s*([^.!?\n]{1,200})",
    re.I,
)


def normalize_text(text: str) -> str:
    return " ".join(text.split())


def _dataset_names(text: str) -> List[str]:
    found = {match.group(0).casefold(): match.group(0) for match in _DATASET_PATTERN.finditer(text)}
    canonical = {name.casefold(): name for name in _DATASET_NAMES}
    return sorted((canonical[key] for key in found), key=str.casefold)


def _visual_context(text: str, caption: str, visual_label: str) -> str:
    sentences = [
        sentence.strip()
        for sentence in re.split(r"(?<=[.!?])\s+|\n+", normalize_text(text))
        if sentence.strip()
    ]
    match_index = next(
        (
            index
            for index, sentence in enumerate(sentences)
            if caption.casefold() in sentence.casefold()
        ),
        None,
    )
    if match_index is None:
        match_index = next(
            (
                index
                for index, sentence in enumerate(sentences)
                if visual_label.casefold() in sentence.casefold()
            ),
            None,
        )
    if match_index is None:
        return caption
    start = max(0, match_index - 1)
    end = min(len(sentences), match_index + 2)
    return " ".join(sentences[start:end])


def _visual_records(document: Dict[str, object], text: str) -> List[Dict[str, object]]:
    records: List[Dict[str, object]] = []
    for key, visual_type in (("figures", "figure"), ("tables", "table")):
        values = document.get(key)
        if not isinstance(values, list):
            continue
        for value in values:
            if isinstance(value, dict) and str(value.get("caption", "")).strip():
                records.append({**value, "visual_type": str(value.get("visual_type", visual_type))})

    known_labels = set()
    known_captions = set()
    for record in records:
        visual_type = str(record.get("visual_type", "figure"))
        label = str(record.get("figure_id", record.get("table_id", "")))
        label_number = re.search(r"\d+[A-Za-z]?", label)
        if label_number:
            known_labels.add((visual_type, label_number.group(0).casefold()))
        caption = normalize_text(str(record.get("caption", ""))).rstrip(".")
        if caption:
            known_captions.add(caption.casefold())
    for match in _INLINE_VISUAL_CAPTION.finditer(text):
        visual_type = "table" if match.group(1).lower().startswith("tab") else "figure"
        label = f"{'Table' if visual_type == 'table' else 'Figure'} {match.group(2)}"
        caption = normalize_text(match.group(3))
        label_key = (visual_type, match.group(2).casefold())
        if label_key in known_labels or caption.rstrip(".").casefold() in known_captions:
            continue
        records.append({
            f"{visual_type}_id": label,
            "visual_type": visual_type,
            "caption": caption,
        })
        known_labels.add(label_key)
        known_captions.add(caption.rstrip(".").casefold())
    return records


def split_text_into_chunks(
    text: str,
    chunk_size: int = 220,
    overlap: int = 40,
    metadata: Dict[str, object] | None = None,
) -> List[Dict[str, object]]:
    normalized = normalize_text(text)
    words = normalized.split()
    if not words:
        return []

    chunks: List[Dict[str, object]] = []
    start = 0
    while start < len(words):
        end = min(len(words), start + chunk_size)
        chunk_words = words[start:end]
        chunk_text = " ".join(chunk_words)
        chunk = {"text": chunk_text, "start": start, "end": end}
        if metadata:
            chunk.update(metadata)
        chunks.append(chunk)
        if end == len(words):
            break
        start = max(0, end - overlap)
    return chunks


def chunk_documents(documents: List[Dict[str, object]], chunk_size: int = 220, overlap: int = 40) -> List[Dict[str, object]]:
    chunks: List[Dict[str, object]] = []
    grouped_documents: Dict[str, List[Dict[str, object]]] = {}
    for document in documents:
        grouped_documents.setdefault(str(document["id"]).split("_page_", 1)[0], []).append(document)
        document_text = str(document.get("text", ""))
        visual_records = _visual_records(document, document_text)
        dataset_names = _dataset_names(document_text)
        visual_labels = [
            str(record.get("figure_id", record.get("table_id", ""))).strip()
            for record in visual_records
        ]
        doc_metadata: Dict[str, object] = {
            "document_id": str(document["id"]),
            "source": str(document["source"]),
            "page_number": int(document.get("page_number", 1)),
            "title": str(document.get("title", document.get("id", "Untitled research paper"))),
            "section_type": str(document.get("section_type", "ABSTRACT")),
            "keywords": dataset_names + [label for label in visual_labels if label],
            "semantic_tags": [f"dataset:{name.casefold()}" for name in dataset_names],
        }
        if "image_count" in document:
            doc_metadata["image_count"] = int(document.get("image_count", 0))
        if "figures" in document:
            figures = document.get("figures")
            if isinstance(figures, list):
                doc_metadata["figures"] = [dict(figure) for figure in figures if isinstance(figure, dict)]
        if "tables" in document:
            tables = document.get("tables")
            if isinstance(tables, list):
                doc_metadata["tables"] = [dict(table) for table in tables if isinstance(table, dict)]
        if "source_type" in document:
            doc_metadata["source_type"] = str(document.get("source_type", "central"))
        chunks.extend(split_text_into_chunks(document_text, chunk_size=chunk_size, overlap=overlap, metadata=doc_metadata))

        for record in visual_records:
            visual_type = str(record.get("visual_type", "figure"))
            visual_label = str(
                record.get("figure_id", record.get("table_id", ""))
                or f"{visual_type.title()} on page {doc_metadata['page_number']}"
            )
            caption = normalize_text(str(record.get("caption", "")))
            associated_text = _visual_context(document_text, caption, visual_label)
            associated_text = associated_text or caption
            visual_datasets = _dataset_names(f"{caption} {associated_text}")
            keywords = [visual_label, *visual_datasets]
            semantic_tags = [visual_type, *(f"dataset:{name.casefold()}" for name in visual_datasets)]
            visual_chunk: Dict[str, object] = {
                "text": (
                    f"{visual_label}: {caption}\n"
                    f"Associated analysis: {associated_text}\n"
                    + (f"Datasets/benchmarks: {', '.join(visual_datasets)}" if visual_datasets else "")
                ).strip(),
                "start": 0,
                "end": len(associated_text.split()),
                **{
                    key: value
                    for key, value in doc_metadata.items()
                    if key not in {"figures", "tables", "keywords", "semantic_tags"}
                },
                "document_id": f"{document['id']}_{visual_label.lower().replace(' ', '_')}",
                "visual_type": visual_type,
                "visual_label": visual_label,
                "caption": caption,
                "keywords": keywords,
                "semantic_tags": semantic_tags,
            }
            if visual_type == "figure":
                visual_chunk["figures"] = [dict(record)]
            else:
                visual_chunk["tables"] = [dict(record)]
            chunks.append(visual_chunk)

    for document_id, pages in grouped_documents.items():
        page_numbers = {
            int(page.get("page_number", 1))
            for page in pages
            if str(page.get("id", "")).find("_page_") >= 0
        }
        if len(page_numbers) < 50:
            continue
        ordered_pages = sorted(pages, key=lambda page: int(page.get("page_number", 1)))
        sample_count = min(12, len(ordered_pages))
        sample_indices = {
            round(index * (len(ordered_pages) - 1) / max(sample_count - 1, 1))
            for index in range(sample_count)
        }
        summary_sentences: List[str] = []
        for page_index in sorted(sample_indices):
            page_text = normalize_text(str(ordered_pages[page_index].get("text", "")))
            sentence = next(
                (sentence.strip() for sentence in re.split(r"(?<=[.!?])\s+", page_text) if sentence.strip()),
                "",
            )
            if sentence:
                summary_sentences.append(sentence)
        summary_text = " ".join(summary_sentences)
        summary_words = summary_text.split()[:chunk_size]
        if not summary_words:
            continue
        first_page = ordered_pages[0]
        chunks.append({
            "text": "Document summary: " + " ".join(summary_words),
            "start": 0,
            "end": len(summary_words),
            "document_id": document_id,
            "source": str(first_page.get("source", document_id)),
            "page_number": 1,
            "title": str(first_page.get("title", document_id)),
            "section_type": "DOCUMENT SUMMARY",
            "source_type": str(first_page.get("source_type", "central")),
            "is_summary": 1,
        })
    return chunks
