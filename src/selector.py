from __future__ import annotations

from typing import Dict

from .classifier import classify_query
from .query_analyzer import QueryType

ROUTE_BY_TYPE: Dict[QueryType, str] = {
    QueryType.TECHNICAL: "bge",
    QueryType.CONCEPTUAL: "e5",
    QueryType.ENTITY: "minilm",
}


def select_embedding(query: str) -> str:
    return str(classify_query(query)["embedding_key"])


def select_embedding_details(query: str) -> tuple[QueryType, str]:
    prediction = classify_query(query)
    query_type = QueryType(str(prediction["query_type"]))
    return query_type, str(prediction["embedding_key"])