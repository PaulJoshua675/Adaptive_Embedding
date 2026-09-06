from __future__ import annotations

from typing import Dict

from .query_analyzer import QueryType, analyze_query

ROUTE_BY_TYPE: Dict[QueryType, str] = {
    QueryType.TECHNICAL: "bge",
    QueryType.CONCEPTUAL: "e5",
    QueryType.ENTITY: "minilm",
}


def select_embedding(query: str) -> str:
    return ROUTE_BY_TYPE[analyze_query(query)]


def select_embedding_details(query: str) -> tuple[QueryType, str]:
    query_type = analyze_query(query)
    return query_type, ROUTE_BY_TYPE[query_type]