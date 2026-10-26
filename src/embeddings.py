from __future__ import annotations

import hashlib
import os
import re
from dataclasses import dataclass
from functools import lru_cache
from typing import Dict, Iterable, List, Sequence

try:
    from sentence_transformers import SentenceTransformer  # type: ignore
except ImportError:  # pragma: no cover - optional runtime dependency
    SentenceTransformer = None

from config import EMBEDDING_MODELS

try:
    import streamlit as st
except ImportError:  # pragma: no cover - optional runtime dependency
    st = None


def _resource_cache(function):
    if st is None:
        return lru_cache(maxsize=8)(function)
    return st.cache_resource(show_spinner=False)(function)


def _data_cache(function):
    if st is None:
        return lru_cache(maxsize=2048)(function)
    return st.cache_data(show_spinner=False, max_entries=2048)(function)


def _tokens(text: str) -> List[str]:
    return re.findall(r"[a-z0-9]+", text.lower())


@dataclass
class EmbeddingEncoder:
    key: str
    model_name: str
    dimension: int = 384

    def __post_init__(self) -> None:
        self._model = None
        use_transformers = os.getenv("EMBEDDING_USE_TRANSFORMERS", "0").lower() in {"1", "true", "yes"}
        if SentenceTransformer is not None and use_transformers:
            local_only = os.getenv("EMBEDDING_LOCAL_ONLY", "1").lower() not in {"0", "false", "no"}
            try:
                kwargs = {"local_files_only": True} if local_only else {}
                self._model = SentenceTransformer(self.model_name, model_kwargs=kwargs)
                self.dimension = int(self._model.get_sentence_embedding_dimension())
            except Exception:
                self._model = None

    @property
    def using_transformer(self) -> bool:
        return self._model is not None

    def encode(self, texts: str | Sequence[str]) -> List[List[float]]:
        values = [texts] if isinstance(texts, str) else list(texts)
        if self._model is not None:
            encoded = self._model.encode(values, normalize_embeddings=True, show_progress_bar=False)
            return encoded.tolist() if hasattr(encoded, "tolist") else [list(row) for row in encoded]

        # Stable feature hashing keeps indexing and querying usable without model downloads.
        vectors: List[List[float]] = []
        for text in values:
            vector = [0.0] * self.dimension
            for token in _tokens(text):
                digest = hashlib.blake2b(f"{self.key}:{token}".encode(), digest_size=8).digest()
                index = int.from_bytes(digest[:4], "little") % self.dimension
                sign = 1.0 if digest[4] & 1 else -1.0
                vector[index] += sign
            norm = sum(value * value for value in vector) ** 0.5
            vectors.append([value / norm for value in vector] if norm else vector)
        return vectors

    def encode_query(self, query: str) -> List[float]:
        return list(_cached_query_vector(self.key, self.model_name, query))

    def _encode_query_uncached(self, query: str) -> List[float]:
        prefix = "query: " if self.key == "e5" and self.using_transformer else ""
        return self.encode(prefix + query)[0]


@_resource_cache
def _load_cached_encoder(key: str, model_name: str) -> EmbeddingEncoder:
    return EmbeddingEncoder(key, model_name)


@_data_cache
def _cached_query_vector(key: str, model_name: str, query: str) -> List[float]:
    return _load_cached_encoder(key, model_name)._encode_query_uncached(query)


class EmbeddingRegistry:
    def __init__(self, models: Dict[str, str] | None = None):
        configured = models or EMBEDDING_MODELS
        self.encoders = {
            key: _load_cached_encoder(key, name)
            for key, name in configured.items()
        }

    def __getitem__(self, key: str) -> EmbeddingEncoder:
        try:
            return self.encoders[key]
        except KeyError as exc:
            raise ValueError(f"Unknown embedding key {key!r}; choose from {sorted(self.encoders)}") from exc

    def encode_documents(self, chunks: Iterable[Dict[str, object]]) -> Dict[str, List[List[float]]]:
        texts = [str(chunk.get("text", "")) for chunk in chunks]
        return {
            key: encoder.encode(
                [f"passage: {text}" for text in texts]
                if key == "e5" and encoder.using_transformer
                else texts
            )
            for key, encoder in self.encoders.items()
        }