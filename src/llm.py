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
            return "Information not found in the corpus."

        sentences = [sentence.strip() for sentence in re.split(r"(?<=[.!?])\s+", context_text) if sentence.strip()]
        if not sentences:
            sentences = [context_text.strip()]

        question_tokens = {
            token
            for token in re.findall(r"[a-z0-9]+", question.lower())
            if token not in {"what", "which", "when", "where", "who", "why", "how", "does", "do", "did", "can", "could", "would", "should", "the", "a", "an", "to", "of", "for", "use", "uses", "used"}
        }

        ranked_sentences = []
        for sentence in sentences:
            sentence_tokens = set(re.findall(r"[a-z0-9]+", sentence.lower()))
            overlap = len(question_tokens & sentence_tokens)
            if overlap:
                ranked_sentences.append((overlap, sentence))

        if not ranked_sentences:
            return " ".join(sentences[:3])
        ranked_sentences.sort(key=lambda item: item[0], reverse=True)
        selected = [sentence for _, sentence in ranked_sentences[:6]]
        return " ".join(selected)

    def generate_answer(self, question: str, context_chunks: List[Dict[str, object]], instructions: str | None = None) -> str:
        if not context_chunks:
            return "Information not found in the corpus."

        context_text = "\n".join(str(chunk.get("text", "")) for chunk in context_chunks[:3])

        if self._generator is not None:
            try:
                prompt = (
                    (instructions or "You are an academic research analyst producing a rigorous long-form answer.") + "\n"
                    "Answer the user query using ONLY the provided verified research contexts.\n"
                    "Do not use external knowledge, assumptions, or unstated implications.\n"
                    "If the contexts do not contain the answer, reply exactly: Information not found in the corpus.\n"
                    "Do not give a one-line answer or a short summary unless the user explicitly requests one.\n"
                    "Return Markdown with an Executive Insights section containing a bold opening paragraph, "
                    "followed by a Technical Breakdown section containing at least 2 detailed paragraphs or a "
                    "markdown table with concrete methods, equations, comparisons, or data points.\n\n"
                    f"USER QUERY:\n{question}\n\nVERIFIED RESEARCH CONTEXTS:\n{context_text}"
                )
                response = self._generator(prompt, max_length=700, min_length=180)[0]["generated_text"]
                response_text = response.strip()
                if response_text:
                    return response_text
            except Exception:
                pass

        return self._extract_answer(question, context_text)
