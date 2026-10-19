"""Utilities for selecting the answer language independently of source language."""

from __future__ import annotations

import re

SUPPORTED_LANGUAGES = ("English", "Tamil", "Hindi", "Spanish", "French")
LANGUAGE_ALIASES = {
    "en": "English",
    "english": "English",
    "ta": "Tamil",
    "tamil": "Tamil",
    "hi": "Hindi",
    "hindi": "Hindi",
    "es": "Spanish",
    "spanish": "Spanish",
    "fr": "French",
    "french": "French",
}


def detect_language(text: str) -> str:
    """Infer one supported language from script and common function words."""
    if re.search(r"[\u0B80-\u0BFF]", text):
        return "Tamil"
    if re.search(r"[\u0900-\u097F]", text):
        return "Hindi"
    lowered = f" {text.lower()} "
    if re.search(r"[áéíóúñ¿¡]", lowered) or any(
        token in lowered for token in (" el ", " la ", " que ", " cómo ", " para ")
    ):
        return "Spanish"
    if re.search(r"[àâçéèêëîïôûùüÿœ]", lowered) or any(
        token in lowered for token in (" le ", " les ", " des ", " est ", " pourquoi ")
    ):
        return "French"
    return "English"


def resolve_response_language(query: str, selection: str | None = None) -> str:
    """Resolve a user-selected language or fall back to query-language detection."""
    if selection:
        normalized = LANGUAGE_ALIASES.get(selection.strip().lower())
        if normalized:
            return normalized
        if selection.strip().lower() not in {"auto", "detect", "detect from query"}:
            raise ValueError(f"Unsupported response language: {selection}")
    return detect_language(query)