from __future__ import annotations

import hashlib
import os
import re
from dataclasses import dataclass
from typing import Dict, Iterable, List, Sequence

try:
    from sentence_transformers import SentenceTransformer  # type: ignore
except ImportError:  # pragma: no cover - optional runtime dependency
    SentenceTransformer = None

from config import EMBEDDING_MODELS


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
        prefix = "query: " if self.key == "e5" else ""
        return self.encode(prefix + query)[0]


class EmbeddingRegistry:
    def __init__(self, models: Dict[str, str] | None = None):
        configured = models or EMBEDDING_MODELS
        self.encoders = {key: EmbeddingEncoder(key, name) for key, name in configured.items()}

    def __getitem__(self, key: str) -> EmbeddingEncoder:
        try:
            return self.encoders[key]
        except KeyError as exc:
            raise ValueError(f"Unknown embedding key {key!r}; choose from {sorted(self.encoders)}") from exc

    def encode_documents(self, chunks: Iterable[Dict[str, object]]) -> Dict[str, List[List[float]]]:
        texts = [str(chunk.get("text", "")) for chunk in chunks]
        return {
            key: encoder.encode([f"passage: {text}" for text in texts] if key == "e5" else texts)
            for key, encoder in self.encoders.items()
        }