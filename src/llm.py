from __future__ import annotations

import re
import logging
from threading import Thread
from functools import lru_cache
from pathlib import Path
from typing import Dict, Iterable, Iterator, List, Optional

from .extract_pdf import clean_extracted_text
from .multilingual import resolve_response_language
import os
from huggingface_hub import InferenceClient

# Initialize InferenceClient using the environment variable
hf_token = os.getenv("HF_TOKEN")
client = InferenceClient(model="google/flan-t5-base", token=hf_token)

def generate_response(prompt: str) -> str:
    try:
        response = client.text_generation(
            prompt, 
            max_new_tokens=1024, 
            temperature=0.3
        )
        return response
    except Exception as e:
        print(f"Error calling Hugging Face Inference API: {e}")
        raise e

try:
    from transformers import pipeline  # type: ignore
except ImportError:  # pragma: no cover - optional dependency
    pipeline = None


try:
    import streamlit as st
except ImportError:  # pragma: no cover - optional runtime dependency
    st = None


MIN_SIMILARITY_SCORE = 0.20
LOGGER = logging.getLogger(__name__)
_PROMPT_PATH = Path(__file__).resolve().parents[1] / "prompts" / "prompt.txt"
_ARXIV_ID = re.compile(r"(?<!\d)(\d{4}\.\d{4,5})(?:v\d+)?(?!\d)", re.I)
_RESPONSE_HEADINGS = {
    "English": (
        "Overview & Core Context",
        "Methodology & Architecture",
        "Experimental Results",
        "Limitations & Challenges",
        "Practical Implications",
    ),
    "Tamil": (
        "மேலோட்டம் மற்றும் அடிப்படைச் சூழல்",
        "முறையியல் மற்றும் கட்டமைப்பு",
        "சோதனை முடிவுகள்",
        "வரம்புகள் மற்றும் சவால்கள்",
        "நடைமுறை விளைவுகள்",
    ),
    "Hindi": (
        "अवलोकन और मूल संदर्भ",
        "कार्यप्रणाली और संरचना",
        "प्रायोगिक परिणाम",
        "सीमाएँ और चुनौतियाँ",
        "व्यावहारिक निहितार्थ",
    ),
    "Spanish": (
        "Panorama y contexto central",
        "Metodología y arquitectura",
        "Resultados experimentales",
        "Limitaciones y desafíos",
        "Implicaciones prácticas",
    ),
    "French": (
        "Vue d’ensemble et contexte central",
        "Méthodologie et architecture",
        "Résultats expérimentaux",
        "Limites et défis",
        "Implications pratiques",
    ),
}
_SECTION_TERMS = (
    re.compile(r"\b(?:overview|context|background|contribution|executive|insights?)\b", re.I),
    re.compile(r"\b(?:method|methodology|architecture|design|technical|approach|model)\b", re.I),
    re.compile(r"\b(?:experiment|result|evaluation|dataset|metric|performance)\b", re.I),
    re.compile(r"\b(?:limitation|challenge|weakness|constraint|drawback)\b", re.I),
    re.compile(r"\b(?:implication|application|practical|deployment|future|source|citation)\b", re.I),
)
_BULLET_PREFIX = re.compile(r"^\s*(?:[-*+]\s+|\d+[.)]\s+)")
_GENERIC_POINT_LABEL = re.compile(r"^(?:\*\*)?(?:Point|Key Concept)\s*\d+\s*:?(?:\*\*)?\s*", re.I)
_SECTION_EVIDENCE_TERMS = (
    {"instructgpt", "instruction", "model", "human", "preference", "training", "alignment", "contribution"},
    {"supervised", "fine", "tuning", "reward", "ppo", "pretraining", "kl", "label", "procedure", "trained"},
    {"evaluation", "result", "score", "dataset", "drop", "squad", "win", "reward", "performance", "accuracy"},
    {"regression", "decrease", "decreases", "tax", "limitation", "challenge", "bias", "risk", "over", "mitigate", "however", "but"},
    {"generalization", "instruction", "deployment", "application", "performance", "pretraining", "task", "summarization", "question"},
)


def _section_index(heading: str) -> int | None:
    numbered = re.match(r"\s*(\d+)[.)]\s+", heading)
    if numbered:
        index = int(numbered.group(1)) - 1
        return index if 0 <= index < 5 else None
    for index, pattern in enumerate(_SECTION_TERMS):
        if pattern.search(heading):
            return index
    return None


def _enforce_response_structure(
    answer: str,
    response_language: str = "English",
    source_citations: str | None = None,
    context_chunks: List[Dict[str, object]] | None = None,
) -> str:
    """Normalize generated prose into five nonempty, grounded sections."""
    if answer.casefold().strip(" .*!\n") == "information not found in the corpus":
        return "Information not found in the corpus."
    answer_lines = []
    skipping_citations = False
    for line in answer.splitlines():
        if re.search(r"\bVerified Source Citations\b", line, flags=re.I):
            if re.match(r"\s*(?:#{1,6}\s*)?\*{0,2}Verified Source Citations\*{0,2}\s*:?\s*$", line, flags=re.I):
                skipping_citations = True
                continue
            line = re.sub(r"\s*\*{0,2}Verified Source Citations\*{0,2}\s*:.*$", "", line, flags=re.I)
        if skipping_citations:
            if re.match(r"\s*#{1,6}\s+", line):
                skipping_citations = False
            else:
                continue
        answer_lines.append(line)
    answer = "\n".join(answer_lines)

    points: list[tuple[int | None, str]] = []
    current_section: int | None = None
    pending_point: str | None = None

    def add_pending() -> None:
        nonlocal pending_point
        if pending_point:
            cleaned = clean_extracted_text(_GENERIC_POINT_LABEL.sub("", pending_point).strip())
            if cleaned:
                points.append((current_section, cleaned))
        pending_point = None

    for line in answer.splitlines():
        stripped = line.strip()
        if not stripped:
            add_pending()
            continue
        heading = re.match(r"^#{1,6}\s+(.+?)\s*#*\s*$", stripped)
        if heading:
            add_pending()
            current_section = _section_index(heading.group(1))
            continue
        bullet = _BULLET_PREFIX.match(stripped)
        if bullet:
            add_pending()
            pending_point = stripped[bullet.end():].strip()
        elif pending_point is not None:
            pending_point = f"{pending_point} {stripped}"
        else:
            plain_text = re.sub(r"^\s*#+\s*", "", stripped)
            for sentence in re.split(r"(?<=[.!?])\s+", plain_text):
                sentence = sentence.strip()
                if sentence:
                    points.append((current_section, clean_extracted_text(sentence)))
    add_pending()

    buckets: list[list[str]] = [[] for _ in range(5)]
    next_section = 0
    seen: set[str] = set()
    for indicated_section, point in points:
        point = point.strip(" -*")
        if not point:
            continue
        key = point.casefold()
        if key in seen:
            continue
        seen.add(key)
        section = indicated_section
        if section is None:
            section = next(
                (index for index, pattern in enumerate(_SECTION_TERMS) if pattern.search(point)),
                next_section,
            )
            if len(buckets[section]) >= 3:
                section = next(
                    (index for index in range(5) if len(buckets[index]) < 3),
                    section,
                )
        elif len(buckets[section]) >= 3:
            continue
        if len(buckets[section]) >= 3:
            continue
        buckets[section].append(point)
        next_section = (section + 1) % 5

    headings = _RESPONSE_HEADINGS.get(response_language, _RESPONSE_HEADINGS["English"])
    sections = []
    used_evidence: set[str] = set()
    for index, (heading, section_points) in enumerate(zip(headings, buckets), start=1):
        if not section_points:
            section_points.append(_grounded_section_fallback(index - 1, context_chunks or []))
        if response_language == "English":
            evidence = _extractive_section_evidence(index - 1, section_points, context_chunks or [], used_evidence)
            if evidence:
                supplemental = "Retrieved evidence: " + " ".join(evidence)
                if len(section_points) < 3:
                    section_points.append(supplemental)
                else:
                    section_points[-1] = f"{section_points[-1]} {supplemental}"
        section = f"### {index}. {heading}"
        section += "\n\n" + "\n".join(f"- {point}" for point in section_points)
        sections.append(section)
    formatted = "\n\n".join(sections)
    if source_citations:
        formatted += f"\n\n**Verified Source Citations**\n\n{source_citations}"
    return formatted


def _grounded_section_fallback(index: int, context_chunks: List[Dict[str, object]]) -> str:
    section_names = (
        "overview and core context",
        "methodology and architecture",
        "experimental results",
        "limitations and challenges",
        "practical implications",
    )
    sources = []
    source_sections = []
    for chunk in context_chunks:
        source = Path(str(chunk.get("source", chunk.get("document_id", "")))).name
        paper_id = re.search(r"(?<!\d)(\d{4}\.\d{4,5})(?!\d)", source)
        if paper_id and paper_id.group(1) not in sources:
            sources.append(paper_id.group(1))
        source_section = str(chunk.get("section_type", "")).strip()
        if source_section and source_section.upper() != "UNSECTIONED" and source_section not in source_sections:
            source_sections.append(source_section)
    source_label = ", ".join(sources) if sources else "the retrieved passages"
    section_label = ", ".join(source_sections) if source_sections else "the available retrieved context"
    return (
        f"The retrieved passages from {source_label} do not contain explicit evidence for {section_names[index]} "
        f"for this query. The available context is limited to material identified as {section_label}, and it does not "
        f"provide directly supported details that can be summarized under {section_names[index]}. No claim about "
        "unretrieved parts of the paper or about results, limitations, or recommendations is made here. "
        "This section is left as an evidence boundary rather than filling the gap with external knowledge or "
        "assumptions; consult additional retrieved passages before drawing conclusions on this aspect."
    )


def _extractive_section_evidence(
    index: int,
    section_points: List[str],
    context_chunks: List[Dict[str, object]],
    used_evidence: set[str],
) -> List[str]:
    current_word_count = sum(len(re.findall(r"\b[\w'-]+\b", point)) for point in section_points)
    if current_word_count >= 100:
        return []

    section_tokens = _SECTION_EVIDENCE_TERMS[index]
    existing_text = " ".join(section_points).casefold()
    candidates: list[tuple[int, int, bool, str]] = []
    for chunk_index, chunk in enumerate(context_chunks):
        text = clean_extracted_text(str(chunk.get("text", "")))
        for sentence in re.split(r"(?<=[.!?])\s+", text):
            sentence = sentence.strip()
            words = re.findall(r"\b[\w'-]+\b", sentence)
            key = sentence.casefold()
            if len(words) < 9 or sentence.casefold() in existing_text:
                continue
            sentence_tokens = set(re.findall(r"[a-z][a-z0-9-]*", sentence.casefold()))
            relevance = len(section_tokens & sentence_tokens)
            candidates.append((relevance, chunk_index, key in used_evidence, sentence))

    candidates.sort(key=lambda candidate: (-candidate[0], candidate[1]))
    selected = []
    selected_keys: set[str] = set()
    for require_section_match in (True, False):
        for reuse_evidence in (False, True):
            for relevance, _, was_used, sentence in candidates:
                key = sentence.casefold()
                if bool(relevance) != require_section_match or was_used != reuse_evidence or key in selected_keys:
                    continue
                if not was_used:
                    used_evidence.add(key)
                selected.append(sentence)
                selected_keys.add(key)
                current_word_count += len(re.findall(r"\b[\w'-]+\b", sentence))
                if current_word_count >= 100:
                    break
            if current_word_count >= 100:
                break
        if current_word_count >= 100:
            break
    return selected


@lru_cache(maxsize=1)
def _load_prompt_template() -> str:
    try:
        return _PROMPT_PATH.read_text(encoding="utf-8")
    except OSError:
        return "Question: {question}\nContext: {context}\nAnswer:"


def _resource_cache(function):
    if st is None:
        return lru_cache(maxsize=4)(function)
    return st.cache_resource(show_spinner=False)(function)


@_resource_cache
def _load_generator(model_name: str):
    if pipeline is None:
        LOGGER.warning("Transformers is unavailable; using extractive answer generation")
        return None
    try:
        return pipeline("text2text-generation", model=model_name)
    except Exception as exc:
        LOGGER.warning("Could not load text-generation model %s: %s", model_name, exc)
        return None


class SimpleLLM:
    def __init__(self, model_name: Optional[str] = None):
        self.model_name = model_name
        self._model_enabled = True
        self._generator = None

    def _get_generator(self):
        if self._generator is None and self._model_enabled:
            self._generator = _load_generator(self.model_name or "google/flan-t5-small")
            self._model_enabled = False
        return self._generator

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
            return "Information not found in the corpus."
        ranked_sentences.sort(key=lambda item: item[0], reverse=True)
        selected = [sentence for _, sentence in ranked_sentences[:6]]
        return " ".join(selected)

    @staticmethod
    def _relevant_chunks(question: str, context_chunks: List[Dict[str, object]]) -> List[Dict[str, object]]:
        relevant = []
        requested_paper = _ARXIV_ID.search(question)
        target_paper_id = requested_paper.group(1) if requested_paper else None
        minimum_score = min(MIN_SIMILARITY_SCORE, 0.125) if target_paper_id is not None else MIN_SIMILARITY_SCORE
        for chunk in context_chunks:
            text = clean_extracted_text(str(chunk.get("text", "")))
            if not text:
                continue
            source = str(chunk.get("source", chunk.get("document_id", "")))
            source_match = _ARXIV_ID.search(source)
            if target_paper_id is not None and (
                source_match is None or source_match.group(1) != target_paper_id
            ):
                continue
            if "score" in chunk:
                try:
                    score = float(chunk["score"])
                except (TypeError, ValueError):
                    continue
            else:
                score = 1.0
            if score >= minimum_score:
                relevant.append({**chunk, "text": text})
        return relevant

    def generate_answer(
        self,
        question: str,
        context_chunks: List[Dict[str, object]],
        instructions: str | None = None,
        response_language: str | None = None,
    ) -> str:
        if response_language is None:
            response_language = "English"
            if st is not None:
                selected_language = st.session_state.get("response_language")
                if selected_language:
                    response_language = str(selected_language)
        response_language = resolve_response_language(question, response_language)
        context_chunks = self._relevant_chunks(question, context_chunks)
        if not context_chunks:
            return "Information not found in the corpus."

        context_text = "\n\n".join(str(chunk.get("text", "")) for chunk in context_chunks)

        if self._get_generator() is not None:
            try:
                template = _load_prompt_template()
                prompt = (
                    (instructions or "You are an academic research analyst.") + "\n"
                    "Answer using ONLY the provided retrieved context. Do not rely on external general knowledge or unreferenced research concepts. "
                    "Use only directly relevant facts supported by the context. If no passage directly answers the question, "
                    "reply exactly: Information not found in the corpus. Never invent missing details.\n"
                    f"CRITICAL: Write the entire five-section response strictly in {response_language}. "
                    "Keep technical paper titles and source citations in English.\n"
                    "Write all answer prose in " + response_language + ". Return exactly five Markdown subheadings, numbered 1 through 5, using the five headings specified in the template. "
                    "Under each heading use no more than three detailed Markdown bullets and provide at least 100 words of substantive synthesis. "
                    "Explain what, how, and why from the context. Never repeat paper titles as bullets or leave a heading empty; for unsupported sections, state the evidence boundary without inventing or padding. "
                    "Do not write paragraphs or text outside bullets. "
                    "Remove raw plot-axis numbers, OCR artifacts, and filler while preserving supported findings. "
                    "Never use generic labels such as Point 1 or Key Concept 1.\n\n"
                    + template.replace("{question}", question).replace("{context}", context_text).replace("{response_language}", response_language)
                )
                response = self._generator(prompt, max_new_tokens=1800)[0]["generated_text"]
                response_text = response.strip()
                if response_text:
                    return _enforce_response_structure(response_text, response_language, context_chunks=context_chunks)
            except Exception as exc:
                LOGGER.warning("Text generation failed; using extractive fallback: %s", exc)

        return _enforce_response_structure(
            self._extract_answer(question, context_text),
            response_language,
            context_chunks=context_chunks,
        )

    def generate_answer_stream(
        self,
        question: str,
        context_chunks: List[Dict[str, object]],
        instructions: str | None = None,
        response_language: str | None = None,
    ) -> Iterator[str]:
        if response_language is None:
            response_language = "English"
            if st is not None:
                selected_language = st.session_state.get("response_language")
                if selected_language:
                    response_language = str(selected_language)
        response_language = resolve_response_language(question, response_language)
        context_chunks = self._relevant_chunks(question, context_chunks)
        if not context_chunks:
            yield "Information not found in the corpus."
            return

        generator = self._get_generator()
        model = getattr(generator, "model", None)
        tokenizer = getattr(generator, "tokenizer", None)
        if model is None or tokenizer is None:
            yield self.generate_answer(
                question,
                context_chunks,
                instructions=instructions,
                response_language=response_language,
            )
            return

        try:
            from transformers import TextIteratorStreamer
        except ImportError as exc:  # pragma: no cover - tied to optional transformers dependency
            raise RuntimeError("Streaming generation requires transformers.TextIteratorStreamer") from exc

        context_text = "\n\n".join(str(chunk.get("text", "")) for chunk in context_chunks)
        template = _load_prompt_template()
        prompt = (
            (instructions or "You are an academic research analyst.") + "\n"
            "Answer using ONLY the provided retrieved context. Do not rely on external general knowledge or unreferenced research concepts. "
            "Use only directly relevant facts supported by the context. If no passage directly answers the question, "
            "reply exactly: Information not found in the corpus. Never invent missing details.\n"
            f"CRITICAL: Write the entire five-section response strictly in {response_language}. "
            "Keep technical paper titles and source citations in English.\n"
            f"Write all answer prose in {response_language}. Return exactly five Markdown subheadings, numbered 1 through 5, "
            "using the five headings specified in the template. Under each heading use no more than three detailed Markdown bullets "
            "and provide at least 100 words of substantive synthesis. Explain what, how, and why from context. "
            "Never repeat titles as bullets or leave headings empty; unsupported sections must state the evidence boundary without inventing or padding. "
            "Do not write paragraphs or text outside bullets. "
            "Remove raw plot-axis numbers, OCR artifacts, and filler while preserving supported findings. "
            "Never use generic labels such as Point 1 or Key Concept 1.\n\n"
            + template.replace("{question}", question).replace("{context}", context_text).replace("{response_language}", response_language)
        )
        encoded = tokenizer(prompt, return_tensors="pt")
        model_device = getattr(model, "device", None)
        if model_device is not None:
            encoded = {key: value.to(model_device) for key, value in encoded.items()}
        streamer = TextIteratorStreamer(tokenizer, skip_prompt=True, skip_special_tokens=True)
        generation_errors: List[BaseException] = []

        def generate() -> None:
            try:
                model.generate(**encoded, streamer=streamer, max_new_tokens=700, do_sample=False)
            except BaseException as exc:
                generation_errors.append(exc)
                streamer.end()

        worker = Thread(target=generate, daemon=True)
        worker.start()
        try:
            yield from streamer
        finally:
            worker.join()
        if generation_errors:
            raise RuntimeError("Streaming text generation failed") from generation_errors[0]
