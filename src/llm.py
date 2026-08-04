from __future__ import annotations

import re
from typing import Dict, List, Optional

try:
    from transformers import pipeline  # type: ignore
except ImportError:  # pragma: no cover - optional dependency
    pipeline = None


class SimpleLLM:
    def __init__(self, model_name: Optional[str] = None):
        self.model_name = model_name
        self._generator = None
        if pipeline is not None:
            try:
                self._generator = pipeline("text2text-generation", model=model_name or "google/flan-t5-small")
            except Exception:  # pragma: no cover - model download may fail offline
                self._generator = None

    def _extract_answer(self, question: str, context_text: str) -> str:
        if not context_text:
            return "No relevant chunks were retrieved from the current dataset."

        sentences = [sentence.strip() for sentence in re.split(r"(?<=[.!?])\s+", context_text) if sentence.strip()]
        if not sentences:
            sentences = [context_text.strip()]

        question_tokens = {
            token
            for token in re.findall(r"[a-z0-9]+", question.lower())
            if token not in {"what", "which", "when", "where", "who", "why", "how", "does", "do", "did", "can", "could", "would", "should", "the", "a", "an", "to", "of", "for", "use", "uses", "used"}
        }

        best_sentence = sentences[0]
        best_score = 0
        for sentence in sentences:
            sentence_tokens = set(re.findall(r"[a-z0-9]+", sentence.lower()))
            overlap = len(question_tokens & sentence_tokens)
            if overlap > best_score:
                best_score = overlap
                best_sentence = sentence

        if best_score == 0 and len(best_sentence) > 220:
            return best_sentence[:220].rstrip() + "..."
        return best_sentence

    def generate_answer(self, question: str, context_chunks: List[Dict[str, object]]) -> str:
        if not context_chunks:
            return "No relevant chunks were retrieved from the current dataset."

        context_text = "\n".join(str(chunk.get("text", "")) for chunk in context_chunks[:3])

        if self._generator is not None:
            try:
                prompt = (
                    "Answer the user's question using the provided context. "
                    "Keep the answer concise and factual.\n"
                    f"Question: {question}\nContext: {context_text}"
                )
                response = self._generator(prompt, max_length=120)[0]["generated_text"]
                response_text = response.strip()
                if response_text:
                    return response_text
            except Exception:
                pass

        return self._extract_answer(question, context_text)
