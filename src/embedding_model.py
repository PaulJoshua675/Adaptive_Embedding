from __future__ import annotations

import math
import re
from typing import List

try:
    from sentence_transformers import SentenceTransformer  # type: ignore
except ImportError:  # pragma: no cover - optional dependency
    SentenceTransformer = None


class EmbeddingModel:
    def __init__(self, model_name: str = "simple-bow"):
        self.model_name = model_name
        self.vocabulary = {}
        self._built = False
        self.backend = SentenceTransformer(model_name) if SentenceTransformer is not None else None

    def _tokenize(self, text: str) -> List[str]:
        return re.findall(r"[a-z0-9]+", text.lower())

    def _build_vocabulary(self, texts: List[str]) -> None:
        vocabulary = {}
        for text in texts:
            for token in self._tokenize(text):
                if token not in vocabulary:
                    vocabulary[token] = len(vocabulary)
        self.vocabulary = vocabulary
        self._built = True

    def _embed_single(self, text: str) -> List[float]:
        if not self._built:
            self._build_vocabulary([text])

        counts = {}
        for token in self._tokenize(text):
            counts[token] = counts.get(token, 0) + 1

        vector = [0.0] * len(self.vocabulary)
        for token, count in counts.items():
            if token in self.vocabulary:
                vector[self.vocabulary[token]] = float(count)
        return vector

    def fit(self, texts: List[str]) -> "EmbeddingModel":
        self._build_vocabulary(texts)
        return self

    def embed_text(self, text: str) -> List[float]:
        if self.backend is not None:
            return list(self.backend.encode(text))
        if not self._built:
            self._build_vocabulary([text])
        return self._embed_single(text)

    def embed_texts(self, texts: List[str]) -> List[List[float]]:
        if self.backend is not None:
            return [list(vector) for vector in self.backend.encode(texts)]
        if not self._built:
            self._build_vocabulary(texts)
        return [self._embed_single(text) for text in texts]

    def similarity(self, left: List[float], right: List[float]) -> float:
        if not left or not right:
            return 0.0

        dot = sum(a * b for a, b in zip(left, right))
        left_norm = math.sqrt(sum(value * value for value in left))
        right_norm = math.sqrt(sum(value * value for value in right))
        if left_norm == 0 or right_norm == 0:
            return 0.0
        return dot / (left_norm * right_norm)
