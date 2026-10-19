"""Cross-encoder reranking for retrieved research passages."""

from __future__ import annotations

import os
import logging
from functools import lru_cache
from typing import Dict, List

from config import RERANKER_MODEL

try:
    import streamlit as st
except ImportError:  # pragma: no cover - optional runtime dependency
    st = None

DEFAULT_CROSS_ENCODER = "cross-encoder/ms-marco-MiniLM-L-6-v2"
LOGGER = logging.getLogger(__name__)


def _resource_cache(function):
    if st is None:
        return lru_cache(maxsize=2)(function)
    return st.cache_resource(show_spinner=False)(function)


@_resource_cache
def _load_cross_encoder(model_name: str, local_only: bool):
    from sentence_transformers import CrossEncoder

    return CrossEncoder(model_name, local_files_only=local_only)


class PassageReranker:
    """Re-rank a small candidate set with a lightweight cross-encoder."""

    def __init__(self, model_name: str | None = None) -> None:
        self.model_name = model_name or os.getenv("RERANKER_MODEL", DEFAULT_CROSS_ENCODER or RERANKER_MODEL)
        self.model = None
        try:
            from sentence_transformers import CrossEncoder  # noqa: F401

            local_only = os.getenv("RERANKER_LOCAL_ONLY", "1").lower() not in {"0", "false", "no"}
            self.model = _load_cross_encoder(self.model_name, local_only)
        except Exception as exc:
            LOGGER.warning("Cross-encoder %s is unavailable; preserving retrieval order: %s", self.model_name, exc)
            self.model = None

    def rerank(
        self,
        query: str,
        passages: List[Dict[str, object]],
        top_k: int = 5,
        candidate_k: int = 15,
    ) -> List[Dict[str, object]]:
        """Return the top passages by cross-encoder score, or keep retrieval order."""
        candidates = passages[:candidate_k]
        if not candidates or self.model is None:
            return candidates[:top_k]
        try:
            pairs = [(query, str(item.get("text", ""))) for item in candidates]
            scores = self.model.predict(pairs)
            ranked = [dict(item, rerank_score=float(score)) for item, score in zip(candidates, scores)]
            ranked.sort(key=lambda item: float(item["rerank_score"]), reverse=True)
            return ranked[:top_k]
        except Exception as exc:
            LOGGER.warning("Cross-encoder reranking failed; preserving retrieval order: %s", exc)
            return candidates[:top_k]