from __future__ import annotations

from typing import List, Dict, Tuple

try:
    import faiss  # type: ignore
    import numpy as np  # type: ignore
except ImportError:  # pragma: no cover - optional dependency
    faiss = None
    np = None


class VectorStore:
    def __init__(self, dimension: int):
        self.dimension = dimension
        self.documents: List[Dict[str, object]] = []
        self.embeddings: List[List[float]] = []
        self.index = None
        if faiss is not None and np is not None:
            self.index = faiss.IndexFlatIP(self.dimension)

    @staticmethod
    def _normalize(embedding: List[float]) -> List[float]:
        norm = sum(value * value for value in embedding) ** 0.5
        return [value / norm for value in embedding] if norm else list(embedding)

    def add_documents(self, documents: List[Dict[str, object]], embeddings: List[List[float]]) -> None:
        if not documents:
            return

        self.documents.extend(documents)
        normalized_embeddings = [self._normalize(embedding) for embedding in embeddings]
        self.embeddings.extend(normalized_embeddings)
        if self.index is not None and np is not None and embeddings:
            matrix = np.asarray(normalized_embeddings, dtype=np.float32)
            self.index.add(matrix)

    def search(self, query_embedding: List[float], top_k: int = 5) -> List[Tuple[float, Dict[str, object]]]:
        if not self.documents:
            return []

        if self.index is not None and np is not None:
            query_matrix = np.asarray([self._normalize(query_embedding)], dtype=np.float32)
            scores, indices = self.index.search(query_matrix, min(top_k, len(self.documents)))
            results = []
            for score, index in zip(scores[0], indices[0]):
                if index < 0:
                    continue
                results.append((float(score), self.documents[int(index)]))
            return results

        scored_documents = []
        normalized_query = self._normalize(query_embedding)
        for index, document in enumerate(self.documents):
            embedding = self.embeddings[index]
            score = sum(a * b for a, b in zip(normalized_query, embedding))
            scored_documents.append((score, document))

        scored_documents.sort(key=lambda item: item[0], reverse=True)
        return scored_documents[:top_k]
