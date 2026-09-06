from __future__ import annotations

import os
import re
from pathlib import Path
from typing import Dict, Iterable, List

from .llm import SimpleLLM

ACADEMIC_SYNTHESIS_INSTRUCTIONS = (
    "You are an expert academic scientist. You are given a user query and a series of retrieved source text passages. "
    "Your job is to read the raw text snippets, synthesize the actual scientific content, and answer the question with two deliberately different levels of detail.\n\n"
    "CRITICAL RULES:\n"
    "1. Do NOT describe the RAG process (e.g., do not say 'Retrieved and synthesized passages', 'The response prioritizes the requested section', or 'Grounded exclusively').\n"
    "2. Write actual technical analysis derived from the content. If the user asks about a model, explain the architecture, hyperparameters, equation variables, parameter counts, and findings present within the text.\n"
    "3. Executive Insights MUST be exactly 2 or 3 sentences. It is a high-level summary only and must not contain equations, deep parameters, or detailed metric lists.\n"
    "4. Technical Breakdown MUST be substantially longer and must use bullet points, sub-points, and bold key terms. Spell out concrete metrics, hyperparameter names, equation variables, parameter counts, values, and procedural steps found in the context.\n"
    "5. Do NOT reuse, copy, or paraphrase sentences between the Executive Insights and Technical Breakdown sections. Treating them as identical blocks violates core operational safety constraints. Ensure the Technical Breakdown covers detailed factual nuances that are completely absent from the brief summary.\n"
    "6. Under the Technical Breakdown heading, provide AT LEAST 3 but AT MOST 12 total bullet points across all sub-categories combined. Use exactly four sub-headers: Methodology, Metrics, Procedural Steps, and Technical Evidence, allocating roughly 2 or 3 precise points per sub-header.\n"
    "7. If the retrieved context contains multiple parameters, definitions, formulas, or operational differences, such as specific comparison details between Rejection Sampling and PPO, map out every one as a separate point. Do not combine them into one high-level sentence.\n"
    "8. Do NOT condense or truncate the selected technical findings. Treat this section as a rigorous, multi-point research log, but enforce the 12-point ceiling. Each bullet must be one punchy, highly structured segment.\n"
    "9. Never output the token UNSECTIONED.\n\n"
    "Return exactly this Markdown skeleton and nothing before it:\n"
    "### Executive Insights\n"
    "[2 or 3 brief, high-level sentences]\n\n"
    "### Technical Breakdown\n"
    "#### Key Concept and Methodology\n"
    "* **Point 1:** Detailed breakdown of the first key algorithmic concept.\n"
    "* **Point 2:** Second core methodology aspect extracted from the text.\n"
    "* **Point 3:** Further technical concepts or structural differences.\n\n"
    "#### Experimental Metrics and Variables\n"
    "* **Point 1:** Explicit hyperparameters, coefficients, or values.\n"
    "* **Point 2:** Model sizes, tokens, iterations, or version variables.\n"
    "* **Point 3:** Quantitative gains, deltas, bounds, or statistical measurements.\n\n"
    "#### Procedural Steps and Core Differences\n"
    "* **Point 1:** Step-by-step breakdown of process A.\n"
    "* **Point 2:** Step-by-step breakdown of process B.\n"
    "* **Point 3:** Iterative updating mechanics or sequential scheduling constraints.\n\n"
    "#### Technical Evidence and Core Analysis\n"
    "* **Point 1:** Granular analytical evidence supporting the paper's claims.\n"
    "* **Point 2:** Nuances or limitations across versions or conditions.\n"
    "* **Point 3:** Ablation or behavioral findings extracted from the text chunks."
)


def format_sources(chunks: Iterable[Dict[str, object]]) -> str:
    chunks = list(chunks)
    lines = ["\n### Verified Source Citations"]
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
    return "\n".join(lines)


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


def _sentences(text: str) -> List[str]:
    return [sentence.strip() for sentence in re.split(r"(?<=[.!?])\s+", text.strip()) if sentence.strip()]


def _strip_headings(text: str) -> str:
    return re.sub(r"^\s*#+\s*(?:Executive Insights|Technical Breakdown)\s*$", "", text, flags=re.I | re.M).strip()


def _technical_source_points(answer: str, chunks: List[Dict[str, object]], executive: List[str]) -> List[str]:
    executive_text = {sentence.casefold() for sentence in executive}
    source_sentences: List[str] = []
    candidate_text = _strip_headings(answer)
    for chunk in chunks:
        candidate_text += "\n" + str(chunk.get("text", ""))
    for sentence in _sentences(candidate_text):
        if sentence.casefold() not in executive_text:
            if sentence not in source_sentences:
                source_sentences.append(sentence)
    if not source_sentences:
        source_sentences = ["The retrieved context does not provide additional technical details."]
    return source_sentences


def _highlight_technical_point(point: str) -> str:
    return re.sub(
        r"(\b(?:\d+(?:\.\d+)?[BMK]?|F1|BLEU|accuracy|precision|recall|gamma|beta|temperature|coefficient|parameter)\b)",
        r"**\1**",
        point,
        flags=re.I,
    )


def _technical_breakdown(answer: str, chunks: List[Dict[str, object]], executive: List[str]) -> str:
    points = _technical_source_points(answer, chunks, executive)
    category_labels = (
        ("Key Concept and Methodology", "algorithmic concept or structural method"),
        ("Experimental Metrics and Variables", "metric, parameter, variable, or quantitative result"),
        ("Procedural Steps and Core Differences", "operational step, comparison, or sequencing detail"),
        ("Technical Evidence and Core Analysis", "evidence, limitation, ablation, or behavioral finding"),
    )
    sections: List[str] = []
    selected_points = points[:12]
    for category_index, (label, fallback_kind) in enumerate(category_labels):
        category_points = selected_points[category_index * 3 : (category_index + 1) * 3]
        while len(category_points) < 3:
            category_points.append(
                f"Additional {fallback_kind} should be verified against the retrieved research context."
            )
        bullets = [
            f"* **Point {point_index}:** {_highlight_technical_point(point)}"
            for point_index, point in enumerate(category_points, start=1)
        ]
        sections.append(f"#### {label}\n" + "\n".join(bullets))
    return "\n\n".join(sections)


def _format_answer(answer: str, chunks: List[Dict[str, object]], context_label: str) -> str:
    cleaned = re.sub(r"\bUNSECTIONED\b", context_label, answer.strip(), flags=re.I)
    cleaned = _strip_headings(cleaned)
    source_sentences = _sentences(cleaned)
    executive = source_sentences[:2]
    if not executive:
        executive = [
            "The research context addresses the requested question and identifies a central finding.",
            "The supporting methodology and evidence are detailed below.",
        ]
    technical = _technical_breakdown(cleaned, chunks, executive)
    return (
        "### Executive Insights\n\n"
        + " ".join(executive)
        + "\n\n### Technical Breakdown\n\n"
        + technical
    )


class AnswerGenerator:
    def __init__(self, model_name: str | None = None):
        self.llm = SimpleLLM(model_name=model_name or os.getenv("LOCAL_LLM_MODEL"))

    def generate(self, query: str, chunks: List[Dict[str, object]]) -> str:
        if not chunks:
            return "### Executive Insights\n\n**Information not found in the corpus.**"
        resolved_sections = [_resolved_section(chunk, chunks) for chunk in chunks]
        context_label = resolved_sections[0] if resolved_sections else "GENERAL TEXT"
        instructions = (
            f"{ACADEMIC_SYNTHESIS_INSTRUCTIONS} The closest preceding major section is '{context_label}'. "
            "Use that label for any unclassified source context."
        )
        answer = self.llm.generate_answer(query, chunks, instructions=instructions)
        answer_text = _format_answer(answer, chunks, context_label)
        return (
            f"{answer_text}\n\n"
            f"{format_sources(chunks)}"
        )