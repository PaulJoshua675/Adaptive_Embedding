from __future__ import annotations

import re
from typing import Dict, List


def normalize_text(text: str) -> str:
    return " ".join(text.split())


def split_text_into_chunks(
    text: str,
    chunk_size: int = 220,
    overlap: int = 40,
    metadata: Dict[str, str] | None = None,
) -> List[Dict[str, str | int]]:
    normalized = normalize_text(text)
    words = normalized.split()
    if not words:
        return []

    chunks: List[Dict[str, str | int]] = []
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


def chunk_documents(documents: List[Dict[str, object]], chunk_size: int = 220, overlap: int = 40) -> List[Dict[str, str | int]]:
    chunks: List[Dict[str, str | int]] = []
    grouped_documents: Dict[str, List[Dict[str, object]]] = {}
    for document in documents:
        grouped_documents.setdefault(str(document["id"]).split("_page_", 1)[0], []).append(document)
        doc_metadata: Dict[str, str | int] = {
            "document_id": str(document["id"]),
            "source": str(document["source"]),
            "page_number": int(document.get("page_number", 1)),
            "title": str(document.get("title", document.get("id", "Untitled research paper"))),
            "section_type": str(document.get("section_type", "ABSTRACT")),
        }
        if "image_count" in document:
            doc_metadata["image_count"] = int(document.get("image_count", 0))
        if "source_type" in document:
            doc_metadata["source_type"] = str(document.get("source_type", "central"))
        chunks.extend(split_text_into_chunks(str(document["text"]), chunk_size=chunk_size, overlap=overlap, metadata=doc_metadata))

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
