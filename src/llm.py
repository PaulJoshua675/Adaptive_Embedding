from __future__ import annotations

import logging
import os
import re
from threading import Thread
from functools import lru_cache
from pathlib import Path
from typing import Dict, Iterator, List, Optional

from openai import OpenAI, OpenAIError

from .extract_pdf import clean_extracted_text
from .multilingual import resolve_response_language


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
GROQ_BASE_URL = "https://api.groq.com/openai/v1"
GROQ_MODELS = (
    "openai/gpt-oss-20b",
    "openai/gpt-oss-120b",
    "qwen/qwen3.8-27b",
)
GROQ_MODEL = GROQ_MODELS[0]
HF_BASE_URL = "https://router.huggingface.co/v1"
HF_MODELS = (
    "google/gemma-2-2b-it",
    "HuggingFaceH4/zephyr-7b-beta",
)
HF_MODEL = HF_MODELS[0]
HF_ROUTING_POLICY = "fastest"
MAX_GENERATION_TOKENS = 256
MAX_CONTEXT_CHARS = 500  # Safe limit for 1024-token window
_CONTEXT_TRUNCATION_SUFFIX = "\n[Context truncated to fit model limit...]"
MAX_TOTAL_PROMPT_CHARS = 2500
_PROMPT_TRUNCATION_SUFFIX = "\n[Prompt truncated for model limits]"


def _truncate_context(context_text: str) -> str:
    if len(context_text) <= MAX_CONTEXT_CHARS:
        return context_text
    content_limit = MAX_CONTEXT_CHARS - len(_CONTEXT_TRUNCATION_SUFFIX)
    return context_text[:content_limit] + _CONTEXT_TRUNCATION_SUFFIX


def _truncate_total_prompt(prompt: str) -> str:
    if len(prompt) <= MAX_TOTAL_PROMPT_CHARS:
        return prompt
    content_limit = MAX_TOTAL_PROMPT_CHARS - len(_PROMPT_TRUNCATION_SUFFIX)
    return prompt[:content_limit] + _PROMPT_TRUNCATION_SUFFIX


def query_groq_inference_api(prompt: str) -> str:
    api_key = os.getenv("GROQ_API_KEY")
    if not api_key or not api_key.strip():
        LOGGER.warning("GROQ_API_KEY is missing; skipping Groq inference")
        return ""

    client = OpenAI(api_key=api_key.strip(), base_url=GROQ_BASE_URL)
    for model in GROQ_MODELS:
        try:
            completion = client.chat.completions.create(
                model=model,
                messages=[{"role": "user", "content": prompt}],
                max_tokens=MAX_GENERATION_TOKENS,
                temperature=0.3,
            )
        except (OpenAIError, TimeoutError, ConnectionError) as exc:
            _log_candidate_failure("Groq", model, exc)
            continue
        if not completion.choices:
            LOGGER.warning("Groq model %s returned no chat completion choices; trying next model", model)
            continue
        content = completion.choices[0].message.content
        if not isinstance(content, str) or not content.strip():
            LOGGER.warning("Groq model %s returned an empty chat completion; trying next model", model)
            continue
        return content.strip()

    return ""


def query_huggingface_inference_api(prompt: str) -> str:
    api_key = os.getenv("HF_TOKEN")
    if not api_key or not api_key.strip():
        LOGGER.warning("HF_TOKEN is missing; skipping Hugging Face inference")
        return ""

    client = OpenAI(api_key=api_key.strip(), base_url=HF_BASE_URL)
    for model in HF_MODELS:
        try:
            completion = client.chat.completions.create(
                model=f"{model}:{HF_ROUTING_POLICY}",
                messages=[{"role": "user", "content": prompt}],
                max_tokens=MAX_GENERATION_TOKENS,
                temperature=0.3,
            )
        except (OpenAIError, TimeoutError, ConnectionError) as exc:
            _log_candidate_failure("Hugging Face", model, exc)
            continue
        if not completion.choices:
            LOGGER.warning(
                "Hugging Face model %s returned no chat completion choices; trying next model",
                model,
            )
            continue
        content = completion.choices[0].message.content
        if not isinstance(content, str) or not content.strip():
            LOGGER.warning(
                "Hugging Face model %s returned an empty chat completion; trying next model",
                model,
            )
            continue
        return content.strip()

    return ""


def _log_candidate_failure(provider: str, model: str, exc: Exception) -> None:
    status_code = getattr(exc, "status_code", None)
    error_code = getattr(exc, "code", None)
    error_type = getattr(exc, "type", None)
    rejected_request = (
        status_code in (400, 404)
        or error_code in ("model_not_found", "invalid_request_error")
        or error_type == "invalid_request_error"
    )
    if rejected_request:
        LOGGER.warning(
            "%s model %s rejected (HTTP %s, %s); trying next model",
            provider,
            model,
            status_code or "unknown",
            error_code or error_type or "invalid request",
        )
    else:
        LOGGER.warning(
            "%s inference failed for model %s: %s; trying next model",
            provider,
            model,
            exc,
        )


def generate_response(prompt: str) -> str:
    if os.getenv("HF_TOKEN"):
        response = query_huggingface_inference_api(prompt)
        if response:
            return response
    return query_groq_inference_api(prompt)


def generate_local_fallback(prompt: str, pipeline) -> str:
    """Run local generation with bounded prompt and model-window sizes."""
    final_prompt = _truncate_total_prompt(prompt)
    tokenizer = getattr(pipeline, "tokenizer", None)
    output = pipeline(
        final_prompt,
        max_new_tokens=MAX_GENERATION_TOKENS,
        truncation=True,
        do_sample=False,
        pad_token_id=getattr(tokenizer, "eos_token_id", None),
        return_full_text=False,
    )
    return output[0]["generated_text"]


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
_FALLBACK_COPY = {
    "English": {
        "not_found": "Information not found in the corpus.",
        "section": (
            "The retrieved passages from {source_label} do not contain explicit evidence for {section_name} "
            "for this query. The available context is limited to material identified as {section_label}, and it does not "
            "provide directly supported details that can be summarized under {section_name}. No claim about "
            "unretrieved parts of the paper or about results, limitations, or recommendations is made here. "
            "This section is left as an evidence boundary rather than filling the gap with external knowledge or "
            "assumptions; consult additional retrieved passages before drawing conclusions on this aspect."
        ),
        "section_names": (
            "overview and core context",
            "methodology and architecture",
            "experimental results",
            "limitations and challenges",
            "practical implications",
        ),
        "retrieved": "the retrieved passages",
        "context": "the available retrieved context",
    },
    "French": {
        "not_found": "Les passages récupérés ne contiennent pas de preuve explicite concernant cette question.",
        "section": (
            "Les passages récupérés de {source_label} ne contiennent pas de preuve explicite concernant {section_name} "
            "pour cette question. Le contexte disponible se limite aux éléments identifiés comme {section_label} et "
            "ne fournit pas de détails directement étayés pouvant être résumés dans cette section. Aucune affirmation "
            "n'est faite au sujet des parties non récupérées de l'article, de ses résultats, de ses limites ou de ses "
            "recommandations. Cette section marque la limite des éléments probants, sans combler les lacunes par des "
            "connaissances externes ou des suppositions ; consultez d'autres passages avant de tirer des conclusions."
        ),
        "section_names": (
            "la vue d'ensemble et le contexte central",
            "la méthodologie et l'architecture",
            "les résultats expérimentaux",
            "les limites et les difficultés",
            "les implications pratiques",
        ),
        "retrieved": "passages récupérés",
        "context": "le contexte récupéré disponible",
    },
    "Spanish": {
        "not_found": "Los pasajes recuperados no contienen evidencia explícita sobre esta consulta.",
        "section": (
            "Los pasajes recuperados de {source_label} no contienen evidencia explícita sobre {section_name} "
            "para esta consulta. El contexto disponible se limita al material identificado como {section_label} y "
            "no ofrece detalles respaldados directamente que puedan resumirse en esta sección. No se hacen "
            "afirmaciones sobre partes no recuperadas del artículo ni sobre resultados, limitaciones o "
            "recomendaciones. Esta sección señala el límite de la evidencia, sin completar la información con "
            "conocimientos externos ni suposiciones; consulte otros pasajes antes de sacar conclusiones."
        ),
        "section_names": (
            "la visión general y el contexto central",
            "la metodología y la arquitectura",
            "los resultados experimentales",
            "las limitaciones y los desafíos",
            "las implicaciones prácticas",
        ),
        "retrieved": "los pasajes recuperados",
        "context": "el contexto recuperado disponible",
    },
    "Tamil": {
        "not_found": "மீட்டெடுக்கப்பட்ட பகுதிகளில் இந்தக் கேள்விக்கான வெளிப்படையான ஆதாரம் இல்லை.",
        "section": (
            "{source_label} இலிருந்து மீட்டெடுக்கப்பட்ட பகுதிகளில் இந்தக் கேள்விக்கான {section_name} பற்றிய "
            "வெளிப்படையான ஆதாரம் இல்லை. கிடைக்கும் சூழல் {section_label} என அடையாளம் காணப்பட்ட தகவல்களுக்கு "
            "மட்டுப்படுத்தப்பட்டுள்ளது; இந்தப் பிரிவைச் சுருக்குவதற்கான நேரடியாக ஆதரிக்கப்படும் விவரங்கள் இல்லை. "
            "மீட்டெடுக்கப்படாத கட்டுரைப் பகுதிகள், முடிவுகள், வரம்புகள் அல்லது பரிந்துரைகள் குறித்து இங்கு எந்தக் "
            "கூற்றும் முன்வைக்கப்படவில்லை. வெளிப்புற அறிவையோ ஊகங்களையோ சேர்க்காமல் ஆதார வரம்பாகவே இந்தப் பிரிவு "
            "விடப்பட்டுள்ளது; முடிவெடுப்பதற்கு முன் கூடுதல் பகுதிகளைப் பார்க்கவும்."
        ),
        "section_names": (
            "மேலோட்டமும் அடிப்படைச் சூழலும்",
            "முறையியலும் கட்டமைப்பும்",
            "சோதனை முடிவுகள்",
            "வரம்புகளும் சவால்களும்",
            "நடைமுறை விளைவுகள்",
        ),
        "retrieved": "மீட்டெடுக்கப்பட்ட பகுதிகள்",
        "context": "கிடைக்கும் மீட்டெடுக்கப்பட்ட சூழல்",
    },
    "Hindi": {
        "not_found": "प्राप्त अंशों में इस प्रश्न के बारे में स्पष्ट साक्ष्य नहीं है।",
        "section": (
            "{source_label} से प्राप्त अंशों में इस प्रश्न के लिए {section_name} का स्पष्ट साक्ष्य नहीं है। "
            "उपलब्ध संदर्भ {section_label} के रूप में पहचानी गई सामग्री तक सीमित है और इस अनुभाग के सारांश के "
            "लिए प्रत्यक्ष रूप से समर्थित विवरण प्रदान नहीं करता। लेख के अप्राप्त भागों, परिणामों, सीमाओं या "
            "सिफारिशों के बारे में यहां कोई दावा नहीं किया गया है। बाहरी ज्ञान या अनुमानों से कमी पूरी करने के "
            "बजाय इस अनुभाग को साक्ष्य की सीमा के रूप में रखा गया है; निष्कर्ष निकालने से पहले अतिरिक्त अंश देखें।"
        ),
        "section_names": (
            "अवलोकन और मूल संदर्भ",
            "कार्यप्रणाली और संरचना",
            "प्रायोगिक परिणाम",
            "सीमाएँ और चुनौतियाँ",
            "व्यावहारिक निहितार्थ",
        ),
        "retrieved": "प्राप्त अंश",
        "context": "उपलब्ध प्राप्त संदर्भ",
    },
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


def localized_fallback_message(
    response_language: str,
    section_index: int | None = None,
    source_label: str | None = None,
    section_label: str | None = None,
) -> str:
    """Return a missing-evidence response in the requested supported language."""
    copy = _FALLBACK_COPY.get(response_language, _FALLBACK_COPY["English"])
    if section_index is None:
        return str(copy["not_found"])
    section_names = copy["section_names"]
    section_name = section_names[section_index] if 0 <= section_index < len(section_names) else section_names[0]
    return str(copy["section"]).format(
        source_label=source_label or copy["retrieved"],
        section_label=section_label or copy["context"],
        section_name=section_name,
    )


def _enforce_response_structure(
    answer: str,
    response_language: str = "English",
    source_citations: str | None = None,
    context_chunks: List[Dict[str, object]] | None = None,
) -> str:
    """Normalize generated prose into five nonempty, grounded sections."""
    if answer.casefold().strip(" .*!\n") == "information not found in the corpus":
        return localized_fallback_message(response_language)
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
            section_points.append(
                _grounded_section_fallback(index - 1, context_chunks or [], response_language)
            )
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


def _grounded_section_fallback(
    index: int,
    context_chunks: List[Dict[str, object]],
    response_language: str = "English",
) -> str:
    sources = []
    for chunk in context_chunks:
        source = Path(str(chunk.get("source", chunk.get("document_id", "")))).name
        paper_id = re.search(r"(?<!\d)(\d{4}\.\d{4,5})(?!\d)", source)
        if paper_id and paper_id.group(1) not in sources:
            sources.append(paper_id.group(1))
    source_label = ", ".join(sources) if sources else None
    return localized_fallback_message(
        response_language,
        section_index=index,
        source_label=source_label,
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


def _build_generation_prompt(
    question: str,
    context_text: str,
    instructions: str | None,
    response_language: str,
    fallback: bool,
) -> str:
    answer_scope = (
        "The exact answer to this query is not explicitly detailed in the corpus. However, based on topically related arXiv papers in the index, provide relevant context, related research themes, and adjacent findings. Do not claim that the context directly answers the query."
        if fallback
        else "If no passage directly answers the question, reply exactly: Information not found in the corpus. Never invent missing details."
    )
    template = _load_prompt_template()
    headings = _RESPONSE_HEADINGS.get(response_language, _RESPONSE_HEADINGS["English"])
    prompt_values = {
        "question": question,
        "context": context_text,
        "response_language": response_language,
        "target_language": response_language,
        **{f"heading_{index}": heading for index, heading in enumerate(headings, start=1)},
    }
    visual_instructions = ""
    if re.search(r"\b(?:figure|fig\.?|table|tab\.?)\b", context_text, re.I):
        visual_instructions = (
            "The retrieved context includes figure or table evidence. Explicitly synthesize and narrate its key trends, scores, "
            "comparisons, and takeaways; connect the evidence to the named dataset or task, and retain reported values. "
            "Explain a trend only when supported by the caption or retrieved analysis, and never infer unseen visual details.\n"
        )
    return (
        (instructions or "You are an academic research analyst.") + "\n"
        "Answer using ONLY the provided retrieved context. Do not rely on external general knowledge or unreferenced research concepts. "
        f"Use only facts supported by the context. {answer_scope}\n"
        + visual_instructions
        + f"CRITICAL: Write the entire five-section response strictly in {response_language}. "
        "Keep technical paper titles and source citations in English.\n"
        f"CRITICAL: You must translate ALL 5 structural subheadings into the requested target language ({response_language}). "
        "Do not leave any heading, section marker, or overview label in English.\n"
        "Translate both the structural subheadings AND all detailed body content, explanations, and key takeaways into the target language requested by the user.\n"
        "CRITICAL: Translate every single structural subheading, section title, and body paragraph into the requested target language. "
        "Do not leave any structural headers, template text, or explanatory content in English. Translate all detailed body content, "
        "explanations, and key takeaways; never revert to English midway through the response.\n"
        "Write all answer prose in " + response_language + ". Return exactly five Markdown subheadings, numbered 1 through 5, using the five headings specified in the template. "
        "Under each heading use no more than three detailed Markdown bullets and provide at least 100 words of substantive synthesis. "
        "Explain what, how, and why from the context. Never repeat paper titles as bullets or leave a heading empty; for unsupported sections, state the evidence boundary without inventing or padding. "
        "Do not write paragraphs or text outside bullets. "
        "Remove raw plot-axis numbers, OCR artifacts, and filler while preserving supported findings. "
        "Never use generic labels such as Point 1 or Key Concept 1.\n\n"
        + template.format_map(prompt_values)
    )


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
        return pipeline("text-generation", model=model_name)
    except Exception as exc:
        LOGGER.warning("Could not load text-generation model %s: %s", model_name, exc)
        return None


class SimpleLLM:
    def __init__(self, model_name: Optional[str] = None):
        self.model_name = model_name
        self._model_enabled = True
        self._generator = None

    def generate(self, prompt: str) -> str:
        return generate_response(prompt)

    def _get_generator(self):
        if self._generator is None and self._model_enabled:
            self._generator = _load_generator(self.model_name or "gpt2")
            self._model_enabled = False
        return self._generator

    def _extract_answer(self, question: str, context_text: str, fallback: bool = False) -> str:
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
            if not fallback:
                return "Information not found in the corpus."
            return " ".join(sentences[:6])
        ranked_sentences.sort(key=lambda item: item[0], reverse=True)
        selected = [sentence for _, sentence in ranked_sentences[:6]]
        return " ".join(selected)

    @staticmethod
    def _relevant_chunks(
        question: str,
        context_chunks: List[Dict[str, object]],
        include_low_relevance: bool = False,
    ) -> List[Dict[str, object]]:
        relevant = []
        requested_paper = _ARXIV_ID.search(question)
        target_paper_id = requested_paper.group(1) if requested_paper else None
        minimum_score = (
            -1.0
            if include_low_relevance
            else min(MIN_SIMILARITY_SCORE, 0.125) if target_paper_id is not None else MIN_SIMILARITY_SCORE
        )
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
        fallback: bool = False,
    ) -> str:
        if response_language is None:
            response_language = "English"
            if st is not None:
                selected_language = st.session_state.get("response_language")
                if selected_language:
                    response_language = str(selected_language)
        response_language = resolve_response_language(question, response_language)
        context_chunks = self._relevant_chunks(question, context_chunks, include_low_relevance=fallback)
        if not context_chunks:
            return localized_fallback_message(response_language)

        context_text = _truncate_context(
            "\n\n".join(str(chunk.get("text", "")) for chunk in context_chunks)
        )
        prompt = _build_generation_prompt(
            question,
            context_text,
            instructions,
            response_language,
            fallback,
        )
        hf_token = os.getenv("HF_TOKEN")
        groq_api_key = os.getenv("GROQ_API_KEY")
        if (hf_token and hf_token.strip()) or (groq_api_key and groq_api_key.strip()):
            try:
                response_text = generate_response(prompt).strip()
                if response_text:
                    return _enforce_response_structure(
                        response_text,
                        response_language,
                        context_chunks=context_chunks,
                    )
            except Exception as exc:
                LOGGER.warning("Hosted generation failed: %s", exc)
            return _enforce_response_structure(
                self._extract_answer(question, context_text, fallback=fallback),
                response_language,
                context_chunks=context_chunks,
            )

        if self._get_generator() is not None:
            try:
                response = generate_local_fallback(prompt, self._generator)
                response_text = response.strip()
                if response_text:
                    return _enforce_response_structure(response_text, response_language, context_chunks=context_chunks)
            except Exception as exc:
                LOGGER.warning("Text generation failed; using extractive fallback: %s", exc)

        return _enforce_response_structure(
            self._extract_answer(question, context_text, fallback=fallback),
            response_language,
            context_chunks=context_chunks,
        )

    def generate_answer_stream(
        self,
        question: str,
        context_chunks: List[Dict[str, object]],
        instructions: str | None = None,
        response_language: str | None = None,
        fallback: bool = False,
    ) -> Iterator[str]:
        if response_language is None:
            response_language = "English"
            if st is not None:
                selected_language = st.session_state.get("response_language")
                if selected_language:
                    response_language = str(selected_language)
        response_language = resolve_response_language(question, response_language)
        context_chunks = self._relevant_chunks(question, context_chunks, include_low_relevance=fallback)
        if not context_chunks:
            yield localized_fallback_message(response_language)
            return

        hf_token = os.getenv("HF_TOKEN")
        groq_api_key = os.getenv("GROQ_API_KEY")
        if (hf_token and hf_token.strip()) or (groq_api_key and groq_api_key.strip()):
            yield self.generate_answer(
                question,
                context_chunks,
                instructions=instructions,
                response_language=response_language,
                fallback=fallback,
            )
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
                fallback=fallback,
            )
            return

        try:
            from transformers import TextIteratorStreamer
        except ImportError as exc:  # pragma: no cover - tied to optional transformers dependency
            raise RuntimeError("Streaming generation requires transformers.TextIteratorStreamer") from exc

        context_text = _truncate_context(
            "\n\n".join(str(chunk.get("text", "")) for chunk in context_chunks)
        )
        prompt = _build_generation_prompt(
            question,
            context_text,
            instructions,
            response_language,
            fallback,
        )
        final_prompt = _truncate_total_prompt(prompt)
        encoded = tokenizer(
            final_prompt,
            return_tensors="pt",
            truncation=True,
            max_length=1024,
        )
        model_device = getattr(model, "device", None)
        if model_device is not None:
            encoded = {key: value.to(model_device) for key, value in encoded.items()}
        streamer = TextIteratorStreamer(tokenizer, skip_prompt=True, skip_special_tokens=True)
        generation_errors: List[BaseException] = []

        def generate() -> None:
            try:
                model.generate(
                    **encoded,
                    streamer=streamer,
                    max_new_tokens=MAX_GENERATION_TOKENS,
                    do_sample=False,
                )
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
