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
            self.index = faiss.IndexFlatIP(dimension)

    def add_documents(self, documents: List[Dict[str, object]], embeddings: List[List[float]]) -> None:
        self.documents.extend(documents)
        self.embeddings.extend(embeddings)
        if self.index is not None and np is not None:
            matrix = np.array(embeddings, dtype=np.float32)
            self.index.add(matrix)

    def search(self, query_embedding: List[float], top_k: int = 5) -> List[Tuple[float, Dict[str, object]]]:
        if not self.documents:
            return []

        if self.index is not None and np is not None:
            query_matrix = np.array([query_embedding], dtype=np.float32)
            scores, indices = self.index.search(query_matrix, min(top_k, len(self.documents)))
            results = []
            for score, index in zip(scores[0], indices[0]):
                if index < 0:
                    continue
                results.append((float(score), self.documents[int(index)]))
            return results

        scored_documents = []
        for index, document in enumerate(self.documents):
            embedding = self.embeddings[index]
            dot = sum(a * b for a, b in zip(query_embedding, embedding))
            norm_query = sum(value * value for value in query_embedding) ** 0.5
            norm_embedding = sum(value * value for value in embedding) ** 0.5
            if norm_query == 0 or norm_embedding == 0:
                score = 0.0
            else:
                score = dot / (norm_query * norm_embedding)
            scored_documents.append((score, document))

        scored_documents.sort(key=lambda item: item[0], reverse=True)
        return scored_documents[:top_k]
