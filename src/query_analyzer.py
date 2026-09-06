from __future__ import annotations

import re
from enum import Enum


class QueryType(str, Enum):
    TECHNICAL = "technical_mathematical"
    CONCEPTUAL = "conceptual_general"
    ENTITY = "short_keyword_specific_entity"


TECHNICAL_TERMS = {"algorithm", "equation", "loss", "gradient", "derivative", "complexity", "code", "python", "api", "embedding", "matrix", "formula"}
CONCEPTUAL_TERMS = {"what", "why", "how", "explain", "define", "overview", "purpose", "difference", "concept", "architecture"}


def analyze_query(query: str) -> QueryType:
    normalized = query.strip().lower()
    words = re.findall(r"[a-z0-9_]+", normalized)
    if not words:
        return QueryType.CONCEPTUAL
    if len(words) <= 4 and not (set(words) & CONCEPTUAL_TERMS):
        return QueryType.ENTITY
    if re.search(r"[=+\-*/^]|\b(?:algorithm|equation|gradient|derivative|complexity|code|python|matrix|formula)\b", normalized):
        return QueryType.TECHNICAL
    if set(words) & TECHNICAL_TERMS:
        return QueryType.TECHNICAL
    return QueryType.CONCEPTUAL