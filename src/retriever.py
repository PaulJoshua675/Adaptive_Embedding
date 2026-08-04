from __future__ import annotations

import re
from typing import Dict, List, Tuple

from .create_vector_db import VectorStore
from .embedding_model import EmbeddingModel


def _tokenize(text: str) -> List[str]:
    return re.findall(r"[a-z0-9]+", text.lower())


def _keyword_score(query: str, document: Dict[str, object]) -> float:
    query_tokens = set(_tokenize(query))
    if not query_tokens:
        return 0.0

    chunk_text = str(document.get("text", ""))
    chunk_tokens = set(_tokenize(chunk_text))
    if not chunk_tokens:
        return 0.0

    overlap = len(query_tokens & chunk_tokens)
    return overlap / max(len(query_tokens), 1)


def retrieve_top_chunks(
    query: str,
    vector_store: VectorStore,
    embedding_model: EmbeddingModel,
    top_k: int = 5,
) -> List[Tuple[float, Dict[str, object]]]:
    query_vector = embedding_model.embed_text(query)
    results = vector_store.search(query_vector, top_k=top_k)

    if results and any(score > 0 for score, _ in results):
        return results[:top_k]

    scored_documents = []
    for document in vector_store.documents:
        score = _keyword_score(query, document)
        if score > 0:
            scored_documents.append((score, document))

    if scored_documents:
        scored_documents.sort(key=lambda item: item[0], reverse=True)
        return scored_documents[:top_k]

    if results:
        return results[:top_k]

    fallback_documents = [(0.0, document) for document in vector_store.documents[:top_k]]
    return fallback_documents
