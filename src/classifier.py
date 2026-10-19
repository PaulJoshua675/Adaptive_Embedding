"""Model-backed query intent classification for embedding selection."""

from __future__ import annotations

import os
from functools import lru_cache
from typing import Dict, Tuple

from config import DOMAIN_EXAMPLE_QUESTIONS, EMBEDDING_MODELS

ROUTE_LABELS = (
    "technical_mathematical",
    "conceptual_general",
    "short_keyword_specific_entity",
)
ROUTE_TO_EMBEDDING = {
    "technical_mathematical": "bge",
    "conceptual_general": "e5",
    "short_keyword_specific_entity": "minilm",
}
ZERO_SHOT_MODEL = "valhalla/distilbart-mnli-12-3"


class QueryClassifier:
    """Classify query intent with zero-shot NLI and a trained local fallback."""

    def __init__(self, model_name: str = ZERO_SHOT_MODEL) -> None:
        self.model_name = model_name
        self._zero_shot = None
        self._fallback = None

    def _load_zero_shot(self):
        if self._zero_shot is None:
            try:
                from transformers import pipeline

                offline = os.getenv("TRANSFORMERS_OFFLINE", "0").lower() in {"1", "true", "yes"}
                local_only = os.getenv("CLASSIFIER_LOCAL_ONLY", "1" if offline else "0").lower() not in {"0", "false", "no"}
                pipeline_kwargs = {"model_kwargs": {"local_files_only": True}} if local_only else {}
                self._zero_shot = pipeline(
                    "zero-shot-classification",
                    model=self.model_name,
                    **pipeline_kwargs,
                )
            except Exception:
                self._zero_shot = False
        return self._zero_shot

    def _load_fallback(self):
        if self._fallback is None:
            from sklearn.feature_extraction.text import TfidfVectorizer
            from sklearn.linear_model import LogisticRegression
            from sklearn.pipeline import make_pipeline

            examples = {
                "technical_mathematical": [
                    "derive the gradient equation for the loss function",
                    "compare matrix complexity and algorithm runtime",
                    "what is the mathematical formula and parameter update",
                    "prove the convergence bound for this optimization method",
                    DOMAIN_EXAMPLE_QUESTIONS["cs.LG"][0],
                ],
                "conceptual_general": [
                    "explain the purpose of retrieval augmented generation",
                    "how does this architecture work and why",
                    "summarize the main idea and describe the approach",
                    "what are the advantages and limitations of the method",
                    DOMAIN_EXAMPLE_QUESTIONS["cs.CV"][0],
                    DOMAIN_EXAMPLE_QUESTIONS["cs.CL"][0],
                    DOMAIN_EXAMPLE_QUESTIONS["cs.AI"][0],
                    DOMAIN_EXAMPLE_QUESTIONS["cs.IR"][0],
                ],
                "short_keyword_specific_entity": [
                    "BERT",
                    "ResNet 50",
                    "Figure 3",
                    "Attention Is All You Need",
                ],
            }
            texts, labels = [], []
            for label, samples in examples.items():
                texts.extend(samples)
                labels.extend([label] * len(samples))
            self._fallback = make_pipeline(
                TfidfVectorizer(ngram_range=(1, 2), analyzer="char_wb", min_df=1),
                LogisticRegression(max_iter=500, class_weight="balanced"),
            ).fit(texts, labels)
        return self._fallback

    def predict(self, query: str) -> Dict[str, object]:
        """Return the predicted intent, confidence, embedding key, and model."""
        clean_query = query.strip()
        if not clean_query:
            return {
                "query_type": "conceptual_general",
                "confidence": 0.0,
                "embedding_key": "e5",
                "classifier_model": "empty-query-default",
            }

        zero_shot = self._load_zero_shot()
        if zero_shot:
            try:
                result = zero_shot(clean_query, candidate_labels=list(ROUTE_LABELS), multi_label=False)
                label = str(result["labels"][0])
                confidence = float(result["scores"][0])
                model_name = self.model_name
            except Exception:
                label, confidence = self._predict_fallback(clean_query)
                model_name = "tfidf-logistic-regression-fallback"
        else:
            label, confidence = self._predict_fallback(clean_query)
            model_name = "tfidf-logistic-regression-fallback"

        key = ROUTE_TO_EMBEDDING[label]
        return {
            "query_type": label,
            "confidence": confidence,
            "embedding_key": key,
            "embedding_model": EMBEDDING_MODELS[key],
            "classifier_model": model_name,
        }

    def _predict_fallback(self, query: str) -> Tuple[str, float]:
        model = self._load_fallback()
        probabilities = model.predict_proba([query])[0]
        classes = list(model.classes_)
        best_index = int(probabilities.argmax())
        return str(classes[best_index]), float(probabilities[best_index])


@lru_cache(maxsize=1)
def get_query_classifier() -> QueryClassifier:
    """Return the process-wide query classifier."""
    return QueryClassifier()


def classify_query(query: str) -> Dict[str, object]:
    """Classify a query using the cached model-backed router."""
    return get_query_classifier().predict(query)