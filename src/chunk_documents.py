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


def chunk_documents(documents: List[Dict[str, str]], chunk_size: int = 220, overlap: int = 40) -> List[Dict[str, str | int]]:
    chunks: List[Dict[str, str | int]] = []
    for document in documents:
        doc_metadata = {"document_id": document["id"], "source": document["source"]}
        chunks.extend(split_text_into_chunks(document["text"], chunk_size=chunk_size, overlap=overlap, metadata=doc_metadata))
    return chunks
