from __future__ import annotations

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

    def generate_answer(self, question: str, context_chunks: List[Dict[str, object]]) -> str:
        if not context_chunks:
            return "No relevant chunks were retrieved."

        if self._generator is not None:
            try:
                context_text = "\n".join(str(chunk.get("text", "")) for chunk in context_chunks[:3])
                prompt = f"Question: {question}\nContext: {context_text}"
                response = self._generator(prompt, max_length=120)[0]["generated_text"]
                return response.strip()
            except Exception:
                pass

        top_context = " ".join(str(chunk.get("text", "")) for chunk in context_chunks[:3])
        return (
            "Based on the retrieved context, the answer is: "
            f"{top_context[:700]}"
        )
