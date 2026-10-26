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
    generation_cache_version,
    localized_fallback_message,
)
from .multilingual import resolve_response_language
from .multilingual import MultilingualResponseError

ACADEMIC_SYNTHESIS_INSTRUCTIONS = (
    "Synthesize only the retrieved research evidence. Use relevant figure and table captions, retain reported values, "
    "and explain supported methods, results, limitations, and implications. Clearly mark evidence gaps. "
    "Preserve technical names, equations, source labels, identifiers, and numbers."
)

_EVIDENCE_REFERENCE = re.compile(r"\[Evidence\s+(\d+)(?:\s*\|[^\]]*)?\]")


def _format_source_citation(
    chunk: Dict[str, object],
    chunks: List[Dict[str, object]],
) -> str:
    source = chunk.get("source", chunk.get("document_id", "unknown"))
    page = chunk.get("page_number", "page unavailable")
    source_name = Path(str(source)).name
    if not source_name or source_name == "unknown":
        return ""
    paper_id_match = re.search(r"(?<!\d)(\d{4}\.\d{4,5})(?!\d)", source_name)
    if paper_id_match:
        paper_url = f"https://arxiv.org/abs/{paper_id_match.group(1)}"
    else:
        paper_url = Path(str(source)).resolve().as_uri() if source_name != "unknown" else "#"
    section = _resolved_section(chunk, chunks)
    paper_id = paper_id_match.group(1) if paper_id_match else source_name
    if str(chunk.get("source_type", "")).lower() == "upload":
        label = f"Custom Uploaded Document | Page {page} | Section: {section}"
    else:
        label = f"{paper_id} | Page {page} | Section: {section}"
    return f"[{label}]({paper_url})"


def _attach_evidence_citations(
    answer: str,
    chunks: List[Dict[str, object]],
    response_language: str,
) -> str:
    def replace_reference(match: re.Match[str]) -> str:
        index = int(match.group(1)) - 1
        if not 0 <= index < len(chunks):
            raise MultilingualResponseError(response_language, "invalid_response")
        return _format_source_citation(chunks[index], chunks)

    return _EVIDENCE_REFERENCE.sub(replace_reference, answer)


def format_sources(chunks: Iterable[Dict[str, object]]) -> str:
    chunks = list(chunks)
    return " | ".join(
        citation
        for chunk in chunks
        if (citation := _format_source_citation(chunk, chunks))
    )


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
    target_language: str = "English",
    source_citations: str | None = None,
    fallback: bool = False,
) -> str:
    answer = _attach_evidence_citations(answer, chunks, target_language)
    cleaned = re.sub(r"\bUNSECTIONED\b", context_label, answer.strip(), flags=re.I)
    if cleaned.casefold().strip(" .*!\n") == "information not found in the corpus":
        return f"**{localized_fallback_message(target_language)}**"
    if cleaned == localized_fallback_message(target_language):
        return f"**{cleaned}**"
    return _enforce_response_structure(
        cleaned,
        target_language,
        source_citations,
        chunks,
        fallback=fallback,
    )


def _formatted_missing_answer(target_language: str) -> str:
    return f"**{localized_fallback_message(target_language)}**"


def _generation_instructions(context_label: str, fallback: bool, target_language: str) -> str:
    instructions = (
        f"{ACADEMIC_SYNTHESIS_INSTRUCTIONS} Context label for unclassified evidence: {context_label}. "
        f"Requested target language: {target_language}."
    )
    if fallback:
        instructions += (
            " Direct evidence is absent; identify any adjacent findings as related context, not as a direct answer."
        )
    return instructions


_ANSWER_CACHE_TTL_SECONDS = 3600
_ANSWER_CACHE_MAX_ENTRIES = 512
_answer_cache: OrderedDict[tuple[str, ...], tuple[float, str]] = OrderedDict()
_answer_cache_lock = RLock()


def _answer_cache_key(
    query: str,
    chunks: List[Dict[str, object]],
    target_language: str,
    fallback: bool,
    model_name: str | None,
) -> tuple[str, ...]:
    chunk_hashes = hashlib.sha256()
    for chunk in chunks:
        text = str(chunk.get("text", ""))
        source = str(chunk.get("source", chunk.get("document_id", "")))
        section = str(chunk.get("section_type", ""))
        passage_id = str(
            chunk.get("chunk_id", chunk.get("id", chunk.get("document_id", "")))
        )
        page = str(chunk.get("page_number", ""))
        for value in (passage_id, text, source, page, section):
            chunk_hashes.update(value.encode("utf-8"))
            chunk_hashes.update(b"\0")
    backend = "hosted" if os.getenv("HF_TOKEN") or os.getenv("GROQ_API_KEY") else "local"
    return (
        query,
        chunk_hashes.hexdigest(),
        target_language,
        str(fallback),
        model_name or os.getenv("LOCAL_LLM_MODEL", ""),
        backend,
        str(MAX_GENERATION_TOKENS),
        generation_cache_version(),
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
        target_language = resolve_response_language(query, response_language)
        cache_key = _answer_cache_key(
            query,
            chunks,
            target_language,
            fallback,
            self.llm.model_name,
        )
        cached = _get_cached_answer(cache_key)
        if cached is not None:
            return cached
        answer = self._generate_uncached(query, chunks, target_language, fallback)
        _cache_answer(cache_key, answer)
        return answer

    def _generate_uncached(
        self,
        query: str,
        chunks: List[Dict[str, object]],
        target_language: str,
        fallback: bool,
    ) -> str:
        if not chunks:
            return _formatted_missing_answer(target_language)
        chunks = SimpleLLM._relevant_chunks(query, chunks, include_low_relevance=fallback)
        if not chunks:
            return _formatted_missing_answer(target_language)
        resolved_sections = [_resolved_section(chunk, chunks) for chunk in chunks]
        context_label = resolved_sections[0] if resolved_sections else "GENERAL TEXT"
        instructions = _generation_instructions(context_label, fallback, target_language)
        answer = self.llm.generate_answer(
            query,
            chunks,
            instructions=instructions,
            response_language=target_language,
            fallback=fallback,
        )
        answer_text = _format_answer(
            answer,
            chunks,
            context_label,
            target_language,
            format_sources(chunks),
            fallback=fallback,
        )
        if fallback and answer_text == _formatted_missing_answer(target_language):
            context_text = "\n\n".join(str(chunk.get("text", "")) for chunk in chunks)
            answer_text = _format_answer(
                self.llm._extract_answer(
                    query,
                    context_text,
                    fallback=True,
                    response_language=target_language,
                ),
                chunks,
                context_label,
                target_language,
                format_sources(chunks),
                fallback=fallback,
            )
        if answer_text == _formatted_missing_answer(target_language):
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
        target_language = resolve_response_language(query, response_language)
        if not chunks:
            return _formatted_missing_answer(target_language)
        chunks = SimpleLLM._relevant_chunks(query, chunks, include_low_relevance=fallback)
        if not chunks:
            return _formatted_missing_answer(target_language)
        resolved_sections = [_resolved_section(chunk, chunks) for chunk in chunks]
        context_label = resolved_sections[0] if resolved_sections else "GENERAL TEXT"
        context_text = "\n\n".join(str(chunk.get("text", "")) for chunk in chunks)
        answer = self.llm._extract_answer(
            query,
            context_text,
            fallback=fallback,
            response_language=target_language,
        )
        return _format_answer(
            answer,
            chunks,
            context_label,
            target_language,
            format_sources(chunks),
            fallback=fallback,
        )

    def generate_stream(
        self,
        query: str,
        chunks: List[Dict[str, object]],
        stream_consumer: Callable[[Iterable[str]], str],
        response_language: str = "English",
        fallback: bool = False,
    ) -> str:
        target_language = resolve_response_language(query, response_language)
        if not chunks:
            return _formatted_missing_answer(target_language)
        cache_key = _answer_cache_key(
            query,
            chunks,
            target_language,
            fallback,
            self.llm.model_name,
        )
        cached = _get_cached_answer(cache_key)
        if cached is not None:
            stream_consumer(iter((cached,)))
            return cached
        chunks = SimpleLLM._relevant_chunks(query, chunks, include_low_relevance=fallback)
        if not chunks:
            return _formatted_missing_answer(target_language)
        resolved_sections = [_resolved_section(chunk, chunks) for chunk in chunks]
        context_label = resolved_sections[0] if resolved_sections else "GENERAL TEXT"
        instructions = _generation_instructions(context_label, fallback, target_language)
        streamed_answer = stream_consumer(
            self.llm.generate_answer_stream(
                query,
                chunks,
                instructions=instructions,
                response_language=target_language,
                fallback=fallback,
            )
        )
        answer_text = _format_answer(
            streamed_answer,
            chunks,
            context_label,
            target_language,
            format_sources(chunks),
            fallback=fallback,
        )
        if fallback and answer_text == _formatted_missing_answer(target_language):
            context_text = "\n\n".join(str(chunk.get("text", "")) for chunk in chunks)
            answer_text = _format_answer(
                self.llm._extract_answer(
                    query,
                    context_text,
                    fallback=True,
                    response_language=target_language,
                ),
                chunks,
                context_label,
                target_language,
                format_sources(chunks),
                fallback=fallback,
            )
        _cache_answer(cache_key, answer_text)
        return answer_text