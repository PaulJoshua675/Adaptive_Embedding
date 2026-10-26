from __future__ import annotations

import re
from functools import lru_cache
from typing import Dict

try:
    import streamlit as st
except ImportError:  # pragma: no cover - optional runtime dependency
    st = None

from config import EMBEDDING_MODELS


def _data_cache(function):
    if st is None:
        return lru_cache(maxsize=500)(function)
    return st.cache_data(show_spinner=False, max_entries=500)(function)


_DIRECT_RESPONSE = re.compile(
    r"^\s*(?:(?P<greeting>hi|hello|hey|good morning|good afternoon|good evening|"
    r"hola|buenos días|buenas tardes|bonjour|salut|namaste|नमस्ते|வணக்கம்)"
    r"|(?P<thanks>thanks|thank you)|(?P<help>help|what can you do))[\s.!?,؟。！]*$",
    re.IGNORECASE,
)
_TECHNICAL_QUERY = re.compile(
    r"[=+\-*/^]|\b(?:algorithm|equation|loss|gradient|derivative|complexity|"
    r"code|python|api|embedding|matrix|formula|optimization)\b",
    re.IGNORECASE,
)
_CONCEPTUAL_TERMS = frozenset(
    {
        "what",
        "why",
        "how",
        "explain",
        "define",
        "overview",
        "purpose",
        "difference",
        "concept",
        "architecture",
    }
)
_QUERY_WORD = re.compile(r"[\w'-]+", re.UNICODE)
_EMBEDDING_BY_TYPE = {
    "technical_mathematical": "bge",
    "conceptual_general": "e5",
    "short_keyword_specific_entity": "minilm",
}


@_data_cache
def route_query(query: str) -> Dict[str, object]:
    """Classify a query with compiled patterns and no model initialization."""
    normalized = query.strip()
    direct_match = _DIRECT_RESPONSE.fullmatch(normalized)
    if direct_match:
        return {
            "route": "direct_response",
            "query_type": "direct_response",
            "direct_kind": direct_match.lastgroup or "greeting",
            "confidence": 1.0,
            "embedding_key": "none",
            "embedding_model": "none",
            "classifier_model": "compiled-regex",
        }

    words = _QUERY_WORD.findall(normalized.casefold())
    if _TECHNICAL_QUERY.search(normalized):
        query_type = "technical_mathematical"
    elif len(words) <= 4 and not (_CONCEPTUAL_TERMS.intersection(words)):
        query_type = "short_keyword_specific_entity"
    else:
        query_type = "conceptual_general"

    embedding_key = _EMBEDDING_BY_TYPE[query_type]
    return {
        "route": "rag_retrieval",
        "query_type": query_type,
        "confidence": 1.0,
        "embedding_key": embedding_key,
        "embedding_model": EMBEDDING_MODELS[embedding_key],
        "classifier_model": "compiled-regex",
    }
