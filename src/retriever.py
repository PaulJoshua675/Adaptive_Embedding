from __future__ import annotations

import re
from typing import Dict, List, Tuple

from .create_vector_db import VectorStore
from .embedding_model import EmbeddingModel
from .extract_pdf import clean_extracted_text
from config import DEFAULT_TOP_K, MIN_RETRIEVAL_SIMILARITY

_ARXIV_ID = re.compile(r"(?<!\d)(\d{4}\.\d{4,5})(?:v\d+)?(?!\d)", re.I)


def _paper_id(document: Dict[str, object]) -> str | None:
    source = str(document.get("source", document.get("document_id", "")))
    match = _ARXIV_ID.search(source)
    return match.group(1) if match else None


def _clean_document(document: Dict[str, object]) -> Dict[str, object] | None:
    cleaned_text = clean_extracted_text(str(document.get("text", "")))
    if not cleaned_text:
        return None
    return {**document, "text": cleaned_text}


def retrieve_top_chunks(
    query: str,
    vector_store: VectorStore,
    embedding_model: EmbeddingModel,
    top_k: int = DEFAULT_TOP_K,
    query_vector: List[float] | None = None,
) -> List[Tuple[float, Dict[str, object]]]:
    limit = min(max(int(top_k), 3), 5)
    query_vector = query_vector or embedding_model.embed_text(query)
    requested_paper = _ARXIV_ID.search(query)
    if requested_paper:
        candidates = [
            (
                sum(a * b for a, b in zip(query_vector, vector_store.embeddings[index])),
                document,
            )
            for index, document in enumerate(vector_store.documents)
            if _paper_id(document) == requested_paper.group(1)
        ]
        candidates.sort(key=lambda item: item[0], reverse=True)
        candidates = candidates[:limit]
    else:
        candidates = vector_store.search(query_vector, top_k=limit)
    cleaned_results: List[Tuple[float, Dict[str, object]]] = []
    for score, document in candidates:
        cleaned_document = _clean_document(document)
        if cleaned_document is not None:
            if score >= MIN_RETRIEVAL_SIMILARITY:
                cleaned_results.append((score, cleaned_document))
    if cleaned_results:
        cleaned_results.sort(key=lambda item: item[0], reverse=True)
        return cleaned_results[:limit]
    return []
