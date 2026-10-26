from __future__ import annotations

import os
import re
from pathlib import Path
import hashlib
import time
from collections import OrderedDict
from threading import RLock
from typing import Callable, Dict, Iterable, List

from .llm import (
    MAX_GENERATION_TOKENS,
    SimpleLLM,
    _enforce_response_structure,
    localized_fallback_message,
)

ACADEMIC_SYNTHESIS_INSTRUCTIONS = (
    "You are an academic research analyst. Answer using ONLY the provided retrieved passages. Do not rely on external general knowledge, "
    "unreferenced research concepts, assumptions, or unstated implications. "
    "Do not describe the retrieval process, or invent absent methods, values, metrics, limitations, or conclusions. "
    "State evidence boundaries clearly and do not claim that retrieved passages support absent facts. "
    "Translate both the structural subheadings AND all detailed body content, explanations, and key takeaways into the target language requested by the user. "
    "CRITICAL: Translate every single structural subheading, section title, and body paragraph into the requested target language. "
    "Do not leave any structural headers, template text, or explanatory content in English. "
    "Translate all detailed body content, explanations, and key takeaways; never revert to English midway through the response. "
    "For sections 1, 2, 3, 4, and 5 individually, write every bullet and evidence-boundary statement in the requested target language. "
    "Keep English only for technical paper titles, source citations, and established proper names. "
    "Return exactly five numbered Markdown subheadings with at most three detailed, evidence-based bullets under each. "
    "Each supported section must contain at least 100 words of substantive synthesis from the retrieved passages. "
    "Explain what the evidence says, how the method or result works, and why it matters, using mathematical foundations, model components, "
    "training setups, empirical rationale, gains, or trade-offs only when explicitly supported. Never use paper titles as bullets. "
    "Do not leave headings empty: if a section has no retrieved evidence, state that limitation clearly without padding or inventing facts. "
    "Do not add paragraphs or text outside bullets. Keep citations out of bullets; add them only in the final Verified Source Citations block."
)


def format_sources(chunks: Iterable[Dict[str, object]]) -> str:
    chunks = list(chunks)
    lines = []
    for chunk in chunks:
        source = chunk.get("source", chunk.get("document_id", "unknown"))
        page = chunk.get("page_number", "page unavailable")
        source_name = Path(str(source)).name
        paper_id_match = re.search(r"(?<!\d)(\d{4}\.\d{4,5})(?!\d)", source_name)
        if paper_id_match:
            paper_url = f"https://arxiv.org/abs/{paper_id_match.group(1)}"
        else:
            paper_url = Path(str(source)).resolve().as_uri() if source_name != "unknown" else "#"
        section = _resolved_section(chunk, chunks)
        paper_id = paper_id_match.group(1) if paper_id_match else source_name
        if str(chunk.get("source_type", "")).lower() == "upload":
            lines.append(f"- [Custom Uploaded Document | Page {page} | Section: {section}]({paper_url})")
        else:
            lines.append(f"- [{paper_id} | Page {page} | Section: {section}]({paper_url})")
    return " | ".join(line[2:] if line.startswith("- ") else line for line in lines)


def _resolved_section(chunk: Dict[str, object], chunks: Iterable[Dict[str, object]]) -> str:
    section = str(chunk.get("section_type", "")).strip()
    if section and section.upper() != "UNSECTIONED":
        return section
    sections = [
        str(item.get("section_type", "")).strip()
        for item in chunks
        if str(item.get("section_type", "")).strip().upper() not in {"", "UNSECTIONED"}
    ]
    return sections[0] if sections else "GENERAL TEXT"


def _format_answer(
    answer: str,
    chunks: List[Dict[str, object]],
    context_label: str,
    response_language: str = "English",
    source_citations: str | None = None,
) -> str:
    cleaned = re.sub(r"\bUNSECTIONED\b", context_label, answer.strip(), flags=re.I)
    if cleaned.casefold().strip(" .*!\n") == "information not found in the corpus":
        return f"**{localized_fallback_message(response_language)}**"
    if cleaned == localized_fallback_message(response_language):
        return f"**{cleaned}**"
    return _enforce_response_structure(cleaned, response_language, source_citations, chunks)


def _formatted_missing_answer(response_language: str) -> str:
    return f"**{localized_fallback_message(response_language)}**"


def _generation_instructions(context_label: str, fallback: bool, response_language: str) -> str:
    instructions = (
        f"{ACADEMIC_SYNTHESIS_INSTRUCTIONS} The closest preceding major section is '{context_label}'. "
        "Use that label for any unclassified source context. "
        f"CRITICAL: Translate ALL 5 structural subheadings into the requested target language ({response_language}). "
        "Do not leave any heading, section marker, or overview label in English."
    )
    if fallback:
        instructions += (
            " The exact answer to this query is not explicitly detailed in the corpus. However, based on topically "
            "related arXiv papers in the index, provide relevant context, related research themes, and adjacent findings."
        )
    return instructions


_ANSWER_CACHE_TTL_SECONDS = 3600
_ANSWER_CACHE_MAX_ENTRIES = 512
_answer_cache: OrderedDict[tuple[str, ...], tuple[float, str]] = OrderedDict()
_answer_cache_lock = RLock()


def _answer_cache_key(
    query: str,
    chunks: List[Dict[str, object]],
    response_language: str,
    fallback: bool,
    model_name: str | None,
) -> tuple[str, ...]:
    chunk_hashes = hashlib.sha256()
    for chunk in chunks:
        text = str(chunk.get("text", ""))
        source = str(chunk.get("source", chunk.get("document_id", "")))
        section = str(chunk.get("section_type", ""))
        for value in (text, source, section):
            chunk_hashes.update(value.encode("utf-8"))
            chunk_hashes.update(b"\0")
    backend = "hosted" if os.getenv("HF_TOKEN") or os.getenv("GROQ_API_KEY") else "local"
    return (
        query,
        chunk_hashes.hexdigest(),
        response_language,
        str(fallback),
        model_name or os.getenv("LOCAL_LLM_MODEL", ""),
        backend,
        str(MAX_GENERATION_TOKENS),
    )


def _get_cached_answer(key: tuple[str, ...]) -> str | None:
    now = time.monotonic()
    with _answer_cache_lock:
        cached = _answer_cache.get(key)
        if cached is None:
            return None
        expires_at, answer = cached
        if expires_at <= now:
            del _answer_cache[key]
            return None
        _answer_cache.move_to_end(key)
        return answer


def _cache_answer(key: tuple[str, ...], answer: str) -> None:
    expires_at = time.monotonic() + _ANSWER_CACHE_TTL_SECONDS
    with _answer_cache_lock:
        _answer_cache[key] = (expires_at, answer)
        _answer_cache.move_to_end(key)
        while len(_answer_cache) > _ANSWER_CACHE_MAX_ENTRIES:
            _answer_cache.popitem(last=False)


class AnswerGenerator:
    def __init__(self, model_name: str | None = None):
        self.llm = SimpleLLM(model_name=model_name or os.getenv("LOCAL_LLM_MODEL"))

    def generate(
        self,
        query: str,
        chunks: List[Dict[str, object]],
        response_language: str = "English",
        fallback: bool = False,
    ) -> str:
        cache_key = _answer_cache_key(
            query,
            chunks,
            response_language,
            fallback,
            self.llm.model_name,
        )
        cached = _get_cached_answer(cache_key)
        if cached is not None:
            return cached
        answer = self._generate_uncached(query, chunks, response_language, fallback)
        _cache_answer(cache_key, answer)
        return answer

    def _generate_uncached(
        self,
        query: str,
        chunks: List[Dict[str, object]],
        response_language: str,
        fallback: bool,
    ) -> str:
        if not chunks:
            return _formatted_missing_answer(response_language)
        chunks = SimpleLLM._relevant_chunks(query, chunks, include_low_relevance=fallback)
        if not chunks:
            return _formatted_missing_answer(response_language)
        resolved_sections = [_resolved_section(chunk, chunks) for chunk in chunks]
        context_label = resolved_sections[0] if resolved_sections else "GENERAL TEXT"
        instructions = _generation_instructions(context_label, fallback, response_language)
        answer = self.llm.generate_answer(
            query,
            chunks,
            instructions=instructions,
            response_language=response_language,
            fallback=fallback,
        )
        answer_text = _format_answer(
            answer,
            chunks,
            context_label,
            response_language,
            format_sources(chunks),
        )
        if fallback and answer_text == "**Information not found in the corpus.**":
            context_text = "\n\n".join(str(chunk.get("text", "")) for chunk in chunks)
            answer_text = _format_answer(
                self.llm._extract_answer(query, context_text, fallback=True),
                chunks,
                context_label,
                response_language,
                format_sources(chunks),
            )
        if answer_text == "**Information not found in the corpus.**":
            return answer_text
        return answer_text

    def generate_fast(
        self,
        query: str,
        chunks: List[Dict[str, object]],
        response_language: str = "English",
        fallback: bool = False,
    ) -> str:
        """Synthesize a grounded response from retrieved sentences without loading an LLM."""
        if not chunks:
            return _formatted_missing_answer(response_language)
        chunks = SimpleLLM._relevant_chunks(query, chunks, include_low_relevance=fallback)
        if not chunks:
            return _formatted_missing_answer(response_language)
        resolved_sections = [_resolved_section(chunk, chunks) for chunk in chunks]
        context_label = resolved_sections[0] if resolved_sections else "GENERAL TEXT"
        context_text = "\n\n".join(str(chunk.get("text", "")) for chunk in chunks)
        answer = self.llm._extract_answer(query, context_text, fallback=fallback)
        return _format_answer(
            answer,
            chunks,
            context_label,
            response_language,
            format_sources(chunks),
        )

    def generate_stream(
        self,
        query: str,
        chunks: List[Dict[str, object]],
        stream_consumer: Callable[[Iterable[str]], str],
        response_language: str = "English",
        fallback: bool = False,
    ) -> str:
        if not chunks:
            return _formatted_missing_answer(response_language)
        cache_key = _answer_cache_key(
            query,
            chunks,
            response_language,
            fallback,
            self.llm.model_name,
        )
        cached = _get_cached_answer(cache_key)
        if cached is not None:
            stream_consumer(iter((cached,)))
            return cached
        chunks = SimpleLLM._relevant_chunks(query, chunks, include_low_relevance=fallback)
        if not chunks:
            return _formatted_missing_answer(response_language)
        resolved_sections = [_resolved_section(chunk, chunks) for chunk in chunks]
        context_label = resolved_sections[0] if resolved_sections else "GENERAL TEXT"
        instructions = _generation_instructions(context_label, fallback, response_language)
        streamed_answer = stream_consumer(
            self.llm.generate_answer_stream(
                query,
                chunks,
                instructions=instructions,
                response_language=response_language,
                fallback=fallback,
            )
        )
        answer_text = _format_answer(
            streamed_answer,
            chunks,
            context_label,
            response_language,
            format_sources(chunks),
        )
        if fallback and answer_text == "**Information not found in the corpus.**":
            context_text = "\n\n".join(str(chunk.get("text", "")) for chunk in chunks)
            answer_text = _format_answer(
                self.llm._extract_answer(query, context_text, fallback=True),
                chunks,
                context_label,
                response_language,
                format_sources(chunks),
            )
        _cache_answer(cache_key, answer_text)
        return answer_text