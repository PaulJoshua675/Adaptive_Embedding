from __future__ import annotations

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
    for document in documents:
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
    return chunks
