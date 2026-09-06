from __future__ import annotations

import re
from collections.abc import Iterable, Sequence
from typing import Any, Dict, List, Mapping, Set

_TOKEN_PATTERN = re.compile(r"[a-z0-9]+(?:'[a-z0-9]+)?", re.I)
CONFIGURATIONS = ("BGE baseline", "E5 baseline", "MiniLM baseline", "Adaptive RAG")


def _safe_divide(numerator: float, denominator: float) -> float:
    return numerator / denominator if denominator else 0.0


def _identifier(item: object) -> str:
    if isinstance(item, Mapping):
        for key in ("id", "document_id", "chunk_id", "figure_id", "source"):
            if item.get(key) is not None:
                return str(item[key])
        return str(item.get("text", ""))
    return str(item)


def retrieval_metrics(
    retrieved_ids: Sequence[object],
    relevant_ids: Iterable[object],
    k: int = 3,
) -> Dict[str, float]:
    """Calculate Hit@K, precision, recall, and F1 for retrieved IDs.

    Precision and recall are calculated over the top-k retrieved items and unique IDs.
    """
    if k < 1:
        raise ValueError("k must be at least 1")
    retrieved = [_identifier(item) for item in list(retrieved_ids)[:k]]
    relevant: Set[str] = {_identifier(item) for item in relevant_ids}
    retrieved_set = set(retrieved)
    true_positive = len(retrieved_set & relevant)
    false_positive = len(retrieved_set - relevant)
    false_negative = len(relevant - retrieved_set)
    precision = _safe_divide(true_positive, true_positive + false_positive)
    recall = _safe_divide(true_positive, true_positive + false_negative)
    f1 = _safe_divide(2 * precision * recall, precision + recall)
    return {
        f"hit@{k}": float(bool(retrieved_set & relevant)),
        "accuracy": float(bool(retrieved_set & relevant)),
        "precision": precision,
        "recall": recall,
        "f1": f1,
        "true_positives": float(true_positive),
        "false_positives": float(false_positive),
        "false_negatives": float(false_negative),
    }


def calculate_retrieval_metrics(retrieved_ids: Sequence[object], relevant_ids: Iterable[object], k: int = 3) -> Dict[str, float]:
    return retrieval_metrics(retrieved_ids, relevant_ids, k=k)


def hit_at_k(retrieved_ids: Sequence[object], relevant_ids: Iterable[object], k: int = 3) -> float:
    return retrieval_metrics(retrieved_ids, relevant_ids, k=k)[f"hit@{k}"]


def precision_score(retrieved_ids: Sequence[object], relevant_ids: Iterable[object], k: int = 3) -> float:
    return retrieval_metrics(retrieved_ids, relevant_ids, k=k)["precision"]


def recall_score(retrieved_ids: Sequence[object], relevant_ids: Iterable[object], k: int = 3) -> float:
    return retrieval_metrics(retrieved_ids, relevant_ids, k=k)["recall"]


def f1_score(retrieved_ids: Sequence[object], relevant_ids: Iterable[object], k: int = 3) -> float:
    return retrieval_metrics(retrieved_ids, relevant_ids, k=k)["f1"]


def _tokens(value: str) -> List[str]:
    return [token.lower() for token in _TOKEN_PATTERN.findall(value)]


def token_level_f1(prediction: str, reference: str) -> float:
    prediction_tokens = _tokens(prediction)
    reference_tokens = _tokens(reference)
    if not prediction_tokens or not reference_tokens:
        return float(not prediction_tokens and not reference_tokens)
    prediction_counts: Dict[str, int] = {}
    reference_counts: Dict[str, int] = {}
    for token in prediction_tokens:
        prediction_counts[token] = prediction_counts.get(token, 0) + 1
    for token in reference_tokens:
        reference_counts[token] = reference_counts.get(token, 0) + 1
    overlap = sum(min(count, reference_counts.get(token, 0)) for token, count in prediction_counts.items())
    precision = _safe_divide(overlap, len(prediction_tokens))
    recall = _safe_divide(overlap, len(reference_tokens))
    return _safe_divide(2 * precision * recall, precision + recall)


def token_f1_score(prediction: str, reference: str) -> float:
    return token_level_f1(prediction, reference)


def exact_match(prediction: str, reference: str) -> float:
    return float(" ".join(prediction.split()).casefold() == " ".join(reference.split()).casefold())


def exact_match_score(prediction: str, reference: str) -> float:
    return exact_match(prediction, reference)


def generation_metrics(prediction: str, reference: str) -> Dict[str, float]:
    return {"token_f1": token_level_f1(prediction, reference), "exact_match": exact_match(prediction, reference)}


def evaluate_configuration(
    retrieved_ids: Sequence[object],
    relevant_ids: Iterable[object],
    prediction: str,
    reference: str,
    k: int = 3,
) -> Dict[str, float]:
    metrics = retrieval_metrics(retrieved_ids, relevant_ids, k=k)
    metrics.update(generation_metrics(prediction, reference))
    return metrics


def generate_benchmark_report(results_dict: Mapping[str, Mapping[str, Any]]) -> str:
    """Render benchmark results as a stable Markdown comparison table."""
    headers = ["Configuration", "Hit@3", "Precision", "Recall", "F1-Score", "Token F1", "Exact Match"]
    lines = ["| " + " | ".join(headers) + " |", "|" + "|".join("---" for _ in headers) + "|"]
    names = list(CONFIGURATIONS)
    for name in names:
        metrics = results_dict.get(name, results_dict.get(name.lower(), {}))
        hit = metrics.get("hit@3", metrics.get("accuracy", 0.0))
        lines.append(
            "| " + " | ".join([
                str(name),
                f"{float(hit):.3f}",
                f"{float(metrics.get('precision', 0.0)):.3f}",
                f"{float(metrics.get('recall', 0.0)):.3f}",
                f"{float(metrics.get('f1', metrics.get('retrieval_f1', 0.0))):.3f}",
                f"{float(metrics.get('token_f1', 0.0)):.3f}",
                f"{float(metrics.get('exact_match', metrics.get('em', 0.0))):.3f}",
            ]) + " |"
        )
    return "\n".join(lines)


__all__ = [
    "CONFIGURATIONS",
    "calculate_retrieval_metrics",
    "exact_match",
    "exact_match_score",
    "evaluate_configuration",
    "generate_benchmark_report",
    "generation_metrics",
    "f1_score",
    "hit_at_k",
    "precision_score",
    "recall_score",
    "retrieval_metrics",
    "token_level_f1",
    "token_f1_score",
]
