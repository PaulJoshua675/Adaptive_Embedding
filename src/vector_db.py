from __future__ import annotations

import json

from pathlib import Path
from typing import Dict, List, Tuple

try:
    import faiss  # type: ignore
    import numpy as np  # type: ignore
except ImportError:  # pragma: no cover - optional runtime dependency
    faiss = None
    np = None


FAISS_THREAD_COUNT = 4


def configure_faiss_threads(thread_count: int = FAISS_THREAD_COUNT) -> None:
    if faiss is not None:
        faiss.omp_set_num_threads(thread_count)


_configure_faiss_threads = configure_faiss_threads


def _new_index(dimension: int, vector_count: int):
    _configure_faiss_threads()
    del vector_count
    return faiss.IndexIDMap(faiss.IndexFlatIP(dimension))


def _normalize_embedding(embedding: List[float]) -> List[float]:
    norm = sum(value * value for value in embedding) ** 0.5
    return [value / norm for value in embedding] if norm else list(embedding)


class FaissVectorStore:
    def __init__(self, dimension: int):
        self.dimension = dimension
        self.documents: List[Dict[str, object]] = []
        self.chunk_payloads: Dict[int, Dict[str, object]] = {}
        self.embeddings: List[List[float]] = []
        self.index = _new_index(dimension, 0) if faiss is not None else None

    def add(self, documents: List[Dict[str, object]], embeddings: List[List[float]]) -> None:
        if len(documents) != len(embeddings):
            raise ValueError("The number of documents must match the number of embeddings")
        existing_count = len(self.documents)
        self.documents.extend(documents)
        normalized_embeddings = [_normalize_embedding(embedding) for embedding in embeddings]
        self.embeddings.extend(normalized_embeddings)
        self.chunk_payloads = {index: document for index, document in enumerate(self.documents)}
        if self.index is not None and normalized_embeddings:
            if existing_count and self.index.is_trained:
                self.index.add_with_ids(
                    np.asarray(normalized_embeddings, dtype="float32"),
                    np.arange(existing_count, len(self.documents), dtype="int64"),
                )
                return
            self.index = _new_index(self.dimension, len(self.embeddings))
            self.index.add_with_ids(
                np.asarray(self.embeddings, dtype="float32"),
                np.arange(len(self.embeddings), dtype="int64"),
            )

    def search(self, query: List[float], top_k: int = 5) -> List[Tuple[float, Dict[str, object]]]:
        if not self.documents or top_k <= 0:
            return []
        limit = min(top_k, len(self.documents))
        if self.index is not None:
            normalized_query = _normalize_embedding(query)
            scores, indices = self.index.search(np.asarray([normalized_query], dtype="float32"), limit)
            return [
                (float(score), self.chunk_payloads[int(index)])
                for score, index in zip(scores[0], indices[0])
                if int(index) in self.chunk_payloads
            ]
        scored = []
        normalized_query = _normalize_embedding(query)
        for embedding, document in zip(self.embeddings, self.documents):
            score = sum(a * b for a, b in zip(normalized_query, embedding))
            scored.append((score, document))
        return sorted(scored, key=lambda item: item[0], reverse=True)[:limit]

    def save(self, directory: str | Path) -> None:
        path = Path(directory)
        path.mkdir(parents=True, exist_ok=True)
        if self.index is not None:
            faiss.write_index(self.index, str(path / "index.faiss"))
        (path / "documents.json").write_text(json.dumps(self.documents, ensure_ascii=True), encoding="utf-8")
        (path / "embeddings.json").write_text(json.dumps(self.embeddings), encoding="utf-8")

    @classmethod
    def load(cls, directory: str | Path, index_loader=None) -> "FaissVectorStore":
        _configure_faiss_threads()
        path = Path(directory)
        documents = json.loads((path / "documents.json").read_text(encoding="utf-8"))
        embeddings = json.loads((path / "embeddings.json").read_text(encoding="utf-8"))
        dimension = len(embeddings[0]) if embeddings else 1
        store = cls(dimension)
        store.documents = documents
        store.chunk_payloads = {index: document for index, document in enumerate(documents)}
        store.embeddings = [_normalize_embedding(embedding) for embedding in embeddings]
        if faiss is not None and (path / "index.faiss").exists():
            if index_loader is None:
                _configure_faiss_threads()
                loaded_index = faiss.read_index(str(path / "index.faiss"))
            else:
                loaded_index = index_loader(str(path / "index.faiss"))
            if getattr(loaded_index, "metric_type", None) == faiss.METRIC_INNER_PRODUCT:
                store.index = loaded_index
            else:
                store.index = _new_index(dimension, len(store.embeddings))
                if store.embeddings:
                    store.index.add_with_ids(
                        np.asarray(store.embeddings, dtype="float32"),
                        np.arange(len(store.embeddings), dtype="int64"),
                    )
        elif store.index is not None and store.embeddings:
            store.index.add_with_ids(
                np.asarray(store.embeddings, dtype="float32"),
                np.arange(len(store.embeddings), dtype="int64"),
            )
        return store


class MultiVectorStore:
    def __init__(self, stores: Dict[str, FaissVectorStore]):
        self.stores = stores

    def save(self, root: str | Path) -> None:
        root_path = Path(root)
        for key, store in self.stores.items():
            store.save(root_path / f"faiss_{key}")

    @classmethod
    def load(cls, root: str | Path, keys: List[str], index_loader=None) -> "MultiVectorStore":
        root_path = Path(root)
        return cls({key: FaissVectorStore.load(root_path / f"faiss_{key}", index_loader=index_loader) for key in keys})