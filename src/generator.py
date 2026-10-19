from __future__ import annotations

import os
import re
from pathlib import Path
from typing import Callable, Dict, Iterable, List

from .llm import SimpleLLM, _enforce_response_structure

ACADEMIC_SYNTHESIS_INSTRUCTIONS = (
    "You are an academic research analyst. Answer using ONLY the provided retrieved passages. Do not rely on external general knowledge, "
    "unreferenced research concepts, assumptions, or unstated implications. "
    "Do not describe the retrieval process, or invent absent methods, values, metrics, limitations, or conclusions. "
    "If the passages do not directly answer the query, return exactly: Information not found in the corpus. "
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
        link = f"[Read Full Paper on arXiv]({paper_url})" if paper_id_match else f"[Open Local Paper]({paper_url})"
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
        return "**Information not found in the corpus.**"
    return _enforce_response_structure(cleaned, response_language, source_citations, chunks)


class AnswerGenerator:
    def __init__(self, model_name: str | None = None):
        self.llm = SimpleLLM(model_name=model_name or os.getenv("LOCAL_LLM_MODEL"))

    def generate(
        self,
        query: str,
        chunks: List[Dict[str, object]],
        response_language: str = "English",
    ) -> str:
        if not chunks:
            return "**Information not found in the corpus.**"
        chunks = SimpleLLM._relevant_chunks(query, chunks)
        if not chunks:
            return "**Information not found in the corpus.**"
        resolved_sections = [_resolved_section(chunk, chunks) for chunk in chunks]
        context_label = resolved_sections[0] if resolved_sections else "GENERAL TEXT"
        instructions = (
            f"{ACADEMIC_SYNTHESIS_INSTRUCTIONS} The closest preceding major section is '{context_label}'. "
            "Use that label for any unclassified source context."
        )
        answer = self.llm.generate_answer(
            query,
            chunks,
            instructions=instructions,
            response_language=response_language,
        )
        answer_text = _format_answer(
            answer,
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
    ) -> str:
        """Synthesize a grounded response from retrieved sentences without loading an LLM."""
        if not chunks:
            return "**Information not found in the corpus.**"
        chunks = SimpleLLM._relevant_chunks(query, chunks)
        if not chunks:
            return "**Information not found in the corpus.**"
        resolved_sections = [_resolved_section(chunk, chunks) for chunk in chunks]
        context_label = resolved_sections[0] if resolved_sections else "GENERAL TEXT"
        context_text = "\n\n".join(str(chunk.get("text", "")) for chunk in chunks)
        answer = self.llm._extract_answer(query, context_text)
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
    ) -> str:
        if not chunks:
            return "**Information not found in the corpus.**"
        chunks = SimpleLLM._relevant_chunks(query, chunks)
        if not chunks:
            return "**Information not found in the corpus.**"
        resolved_sections = [_resolved_section(chunk, chunks) for chunk in chunks]
        context_label = resolved_sections[0] if resolved_sections else "GENERAL TEXT"
        instructions = (
            f"{ACADEMIC_SYNTHESIS_INSTRUCTIONS} The closest preceding major section is '{context_label}'. "
            "Use that label for any unclassified source context."
        )
        streamed_answer = stream_consumer(
            self.llm.generate_answer_stream(
                query,
                chunks,
                instructions=instructions,
                response_language=response_language,
            )
        )
        return _format_answer(
            streamed_answer,
            chunks,
            context_label,
            response_language,
            format_sources(chunks),
        )