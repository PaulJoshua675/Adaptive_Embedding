from __future__ import annotations

from typing import Dict, List, Tuple

from .create_vector_db import VectorStore
from .embedding_model import EmbeddingModel


def retrieve_top_chunks(
    query: str,
    vector_store: VectorStore,
    embedding_model: EmbeddingModel,
    top_k: int = 5,
) -> List[Tuple[float, Dict[str, object]]]:
    query_vector = embedding_model.embed_text(query)
    return vector_store.search(query_vector, top_k=top_k)
