from __future__ import annotations

import copy
import math
import re
from collections import OrderedDict
from dataclasses import dataclass
from threading import RLock
from typing import Dict, Sequence

try:
    import numpy as np
except ImportError:  # pragma: no cover - optional acceleration
    np = None


def _normalized_query(query: str) -> str:
    return re.sub(r"\s+", " ", query).strip().casefold()


def _normalized_vector(vector: Sequence[float]) -> tuple[float, ...]:
    values = tuple(float(value) for value in vector)
    norm = math.sqrt(sum(value * value for value in values))
    if not values or norm == 0 or not math.isfinite(norm):
        raise ValueError("Cache embeddings must be non-empty finite vectors")
    return tuple(value / norm for value in values)


@dataclass(frozen=True)
class _CacheEntry:
    query: str
    embedding: tuple[float, ...]
    response: Dict[str, object]


class SemanticResponseCache:
    """Bounded, process-local cache for complete RAG response payloads."""

    def __init__(self, max_entries: int = 256, similarity_threshold: float = 0.92) -> None:
        if max_entries < 1:
            raise ValueError("max_entries must be positive")
        if not 0.0 <= similarity_threshold <= 1.0:
            raise ValueError("similarity_threshold must be between 0 and 1")
        self.max_entries = max_entries
        self.similarity_threshold = similarity_threshold
        self._entries: OrderedDict[tuple[str, str], _CacheEntry] = OrderedDict()
        self._lock = RLock()

    def get(
        self,
        query: str,
        embedding: Sequence[float],
        namespace: str = "",
    ) -> Dict[str, object] | None:
        normalized_query = _normalized_query(query)
        normalized_embedding = _normalized_vector(embedding)
        key = (namespace, normalized_query)
        with self._lock:
            exact = self._entries.get(key)
            if exact is not None:
                self._entries.move_to_end(key)
                return dict(exact.response)

            best_key = None
            best_similarity = self.similarity_threshold
            candidates = [
                (entry_key, entry)
                for entry_key, entry in self._entries.items()
                if entry_key[0] == namespace and len(entry.embedding) == len(normalized_embedding)
            ]
            if np is not None and candidates:
                matrix = np.asarray([entry.embedding for _, entry in candidates], dtype=np.float64)
                similarities = matrix @ np.asarray(normalized_embedding, dtype=np.float64)
                scored = zip(candidates, similarities.tolist())
            else:
                scored = (
                    ((entry_key, entry), sum(a * b for a, b in zip(normalized_embedding, entry.embedding)))
                    for entry_key, entry in candidates
                )
            for (entry_key, _), similarity in scored:
                if similarity > best_similarity:
                    best_key = entry_key
                    best_similarity = similarity
            if best_key is None:
                return None
            self._entries.move_to_end(best_key)
            return dict(self._entries[best_key].response)

    def get_exact(self, query: str, namespace_prefix: str) -> Dict[str, object] | None:
        normalized_query = _normalized_query(query)
        with self._lock:
            for key, entry in reversed(self._entries.items()):
                if key[1] == normalized_query and key[0].startswith(namespace_prefix):
                    self._entries.move_to_end(key)
                    return dict(entry.response)
        return None

    def set(
        self,
        query: str,
        embedding: Sequence[float],
        response: Dict[str, object],
        namespace: str = "",
    ) -> None:
        normalized_query = _normalized_query(query)
        if not normalized_query:
            return
        entry = _CacheEntry(normalized_query, _normalized_vector(embedding), copy.deepcopy(response))
        key = (namespace, normalized_query)
        with self._lock:
            self._entries[key] = entry
            self._entries.move_to_end(key)
            while len(self._entries) > self.max_entries:
                self._entries.popitem(last=False)

    def clear(self) -> None:
        with self._lock:
            self._entries.clear()


response_cache = SemanticResponseCache()
