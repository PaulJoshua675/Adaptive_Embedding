from __future__ import annotations

import logging
import os
import re
import hashlib
from threading import RLock, Thread
from functools import lru_cache
from pathlib import Path
from typing import Dict, Iterator, List, Optional

from openai import OpenAI, OpenAIError

from .extract_pdf import clean_extracted_text
from .multilingual import (
    LANGUAGE_CONFIG,
    RESPONSE_HEADINGS,
    SUPPORTED_LANGUAGES,
    MultilingualResponseError,
    detect_language,
    localized_response_text,
    resolve_response_language,
    text_matches_language,
)


try:
    from transformers import (  # type: ignore
        AutoModelForSeq2SeqLM,
        AutoTokenizer,
        pipeline,
    )
except ImportError:  # pragma: no cover - optional dependency
    AutoModelForSeq2SeqLM = None
    AutoTokenizer = None
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
MAX_GENERATION_TOKENS = 800
MAX_CONTEXT_CHARS = 6000
_CONTEXT_TRUNCATION_SUFFIX = "\n[Context truncated to fit model limit...]"
MAX_TOTAL_PROMPT_CHARS = 10000
_PROMPT_TRUNCATION_SUFFIX = "\n[Prompt truncated for model limits]"
TRANSLATION_MODEL = os.getenv(
    "MULTILINGUAL_TRANSLATION_MODEL",
    "facebook/nllb-200-distilled-600M",
)


def _truncate_context(context_text: str, max_chars: int = MAX_CONTEXT_CHARS) -> str:
    if len(context_text) <= max_chars:
        return context_text
    content_limit = max_chars - len(_CONTEXT_TRUNCATION_SUFFIX)
    return context_text[:content_limit] + _CONTEXT_TRUNCATION_SUFFIX


def _truncate_total_prompt(prompt: str) -> str:
    if len(prompt) <= MAX_TOTAL_PROMPT_CHARS:
        return prompt
    content_limit = MAX_TOTAL_PROMPT_CHARS - len(_PROMPT_TRUNCATION_SUFFIX)
    return prompt[:content_limit] + _PROMPT_TRUNCATION_SUFFIX


def _chat_messages(prompt: str, system_prompt: str | None = None) -> list[dict[str, str]]:
    if system_prompt:
        return [
            {"role": "system", "content": system_prompt},
            {"role": "user", "content": prompt},
        ]
    return [{"role": "user", "content": prompt}]


def query_groq_inference_api(prompt: str, system_prompt: str | None = None) -> str:
    api_key = os.getenv("GROQ_API_KEY")
    if not api_key or not api_key.strip():
        LOGGER.warning("GROQ_API_KEY is missing; skipping Groq inference")
        return ""

    client = OpenAI(api_key=api_key.strip(), base_url=GROQ_BASE_URL)
    for model in GROQ_MODELS:
        try:
            completion = client.chat.completions.create(
                model=model,
                messages=_chat_messages(prompt, system_prompt),
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


def query_huggingface_inference_api(prompt: str, system_prompt: str | None = None) -> str:
    api_key = os.getenv("HF_TOKEN")
    if not api_key or not api_key.strip():
        LOGGER.warning("HF_TOKEN is missing; skipping Hugging Face inference")
        return ""

    client = OpenAI(api_key=api_key.strip(), base_url=HF_BASE_URL)
    for model in HF_MODELS:
        try:
            completion = client.chat.completions.create(
                model=f"{model}:{HF_ROUTING_POLICY}",
                messages=_chat_messages(prompt, system_prompt),
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


def _stream_hosted_completion(
    client: OpenAI,
    provider: str,
    models: tuple[str, ...],
    prompt: str,
    system_prompt: str | None = None,
    model_suffix: str = "",
) -> Iterator[str]:
    for model in models:
        emitted_content = False
        try:
            stream = client.chat.completions.create(
                model=f"{model}{model_suffix}",
                messages=_chat_messages(prompt, system_prompt),
                max_tokens=MAX_GENERATION_TOKENS,
                temperature=0.3,
                stream=True,
            )
            for event in stream:
                if not event.choices:
                    continue
                content = event.choices[0].delta.content
                if isinstance(content, str) and content:
                    emitted_content = True
                    yield content
        except (OpenAIError, TimeoutError, ConnectionError) as exc:
            if emitted_content:
                raise
            _log_candidate_failure(provider, model, exc)
            continue
        if emitted_content:
            return
        LOGGER.warning("%s model %s returned an empty streamed completion; trying next model", provider, model)


def query_groq_inference_api_stream(
    prompt: str,
    system_prompt: str | None = None,
) -> Iterator[str]:
    api_key = os.getenv("GROQ_API_KEY")
    if not api_key or not api_key.strip():
        LOGGER.warning("GROQ_API_KEY is missing; skipping Groq inference")
        return

    client = OpenAI(api_key=api_key.strip(), base_url=GROQ_BASE_URL)
    yield from _stream_hosted_completion(
        client,
        "Groq",
        GROQ_MODELS,
        prompt,
        system_prompt=system_prompt,
    )


def query_huggingface_inference_api_stream(
    prompt: str,
    system_prompt: str | None = None,
) -> Iterator[str]:
    api_key = os.getenv("HF_TOKEN")
    if not api_key or not api_key.strip():
        LOGGER.warning("HF_TOKEN is missing; skipping Hugging Face inference")
        return

    client = OpenAI(api_key=api_key.strip(), base_url=HF_BASE_URL)
    yield from _stream_hosted_completion(
        client,
        "Hugging Face",
        HF_MODELS,
        prompt,
        system_prompt=system_prompt,
        model_suffix=f":{HF_ROUTING_POLICY}",
    )


def _redact_error(exc: BaseException) -> str:
    message = str(exc)
    for secret_name in ("HF_TOKEN", "GROQ_API_KEY"):
        secret = os.getenv(secret_name)
        if secret:
            message = message.replace(secret, "[REDACTED]")
    return re.sub(r"(?i)\bBearer\s+\S+", "Bearer [REDACTED]", message)


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
            _redact_error(exc),
        )


def generate_response(prompt: str, system_prompt: str | None = None) -> str:
    if os.getenv("HF_TOKEN"):
        response = (
            query_huggingface_inference_api(prompt, system_prompt=system_prompt)
            if system_prompt
            else query_huggingface_inference_api(prompt)
        )
        if response:
            return response
    return (
        query_groq_inference_api(prompt, system_prompt=system_prompt)
        if system_prompt
        else query_groq_inference_api(prompt)
    )


def generate_response_stream(
    prompt: str,
    system_prompt: str | None = None,
) -> Iterator[str]:
    if os.getenv("HF_TOKEN"):
        hf_stream = (
            query_huggingface_inference_api_stream(prompt, system_prompt=system_prompt)
            if system_prompt
            else query_huggingface_inference_api_stream(prompt)
        )
        try:
            first_token = next(hf_stream)
        except StopIteration:
            pass
        else:
            yield first_token
            yield from hf_stream
            return
    if system_prompt:
        yield from query_groq_inference_api_stream(prompt, system_prompt=system_prompt)
    else:
        yield from query_groq_inference_api_stream(prompt)


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
_RESPONSE_HEADINGS = RESPONSE_HEADINGS
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

_REFUSAL_PATTERNS = (
    re.compile(
        r"\b(?:i(?:'m| am) sorry\b.{0,100}\b(?:cannot|can't|won't|unable to)\b|"
        r"i (?:cannot|can't|won't|am unable to)\s+(?:comply|help|assist|provide|answer|fulfill)\b)",
        re.I,
    ),
    re.compile(r"\blo siento\b.{0,100}\bno puedo\b", re.I),
    re.compile(r"\b(?:désolé|je suis désolé)\b.{0,100}\bje ne peux pas\b", re.I),
    re.compile(r"மன்னிக்கவும்.{0,100}(?:முடியாது|செய்ய முடியாது)"),
    re.compile(r"(?:माफ़ी|माफ़ करें).{0,100}(?:नहीं कर सकता|नहीं कर सकती)"),
)
_PROTECTED_TRANSLATION_PATTERN = re.compile(
    r"\[[^\]]+\]\(https?://[^)]+\)|"
    r"https?://\S+|"
    r"\[Evidence\s+\d+(?:\s*\|[^\]]*)?\]|"
    r"(?<!\w)\d{4}\.\d{4,5}(?:v\d+)?(?!\w)|"
    r"`[^`]*`|"
    r"\b[A-Z][A-Za-z0-9]*(?:[-_/][A-Za-z0-9]+)+\b|"
    r"\b[A-Z]{2,}(?:[-_][A-Z0-9]+)*\b|"
    r"(?<!\w)\d+(?:[.,]\d+)*(?:%|[a-zA-Z]+)?(?!\w)"
)
_translation_lock = RLock()


def _is_provider_refusal(answer: str) -> bool:
    return any(pattern.search(answer) for pattern in _REFUSAL_PATTERNS)


@lru_cache(maxsize=2)
def _load_translation_model(model_name: str):
    if AutoModelForSeq2SeqLM is None or AutoTokenizer is None:
        raise RuntimeError("Transformers sequence-to-sequence translation support is unavailable")
    tokenizer = AutoTokenizer.from_pretrained(model_name)
    model = AutoModelForSeq2SeqLM.from_pretrained(model_name)
    model.eval()
    return tokenizer, model


@lru_cache(maxsize=512)
def _translate_text_cached(
    text: str,
    source_language: str,
    target_language: str,
    model_name: str,
) -> str:
    if source_language == target_language:
        return text
    tokenizer, model = _load_translation_model(model_name)
    protected_values: list[str] = []

    def protect(match: re.Match[str]) -> str:
        protected_values.append(match.group(0))
        return f"ZXQPLACEHOLDER{len(protected_values) - 1}QXZ"

    protected_text = _PROTECTED_TRANSLATION_PATTERN.sub(protect, text)
    source_code = LANGUAGE_CONFIG[source_language].translation_code
    target_code = LANGUAGE_CONFIG[target_language].translation_code
    with _translation_lock:
        tokenizer.src_lang = source_code
        target_token = tokenizer.convert_tokens_to_ids(target_code)
        if target_token is None or target_token == tokenizer.unk_token_id:
            raise ValueError(f"Translation model does not support language code {target_code}")
        encoded = tokenizer(
            protected_text,
            return_tensors="pt",
            truncation=True,
            max_length=512,
        )
        generated = model.generate(
            **encoded,
            forced_bos_token_id=target_token,
            max_new_tokens=512,
        )
        translated = tokenizer.batch_decode(generated, skip_special_tokens=True)[0]
    if not isinstance(translated, str) or not translated.strip():
        raise ValueError("Translation model returned an empty output")
    translated = translated.strip()
    for index, value in enumerate(protected_values):
        placeholder = f"ZXQPLACEHOLDER{index}QXZ"
        if placeholder not in translated:
            raise ValueError("Translation model changed a protected technical token")
        translated = translated.replace(placeholder, value)
    return translated


def _translate_point_if_needed(text: str, response_language: str) -> str:
    prose = _PROTECTED_TRANSLATION_PATTERN.sub("", text).strip()
    if not prose or text_matches_language(prose, response_language):
        return text
    source_language = detect_language(prose)
    try:
        translated = _translate_text_cached(
            text,
            source_language,
            response_language,
            TRANSLATION_MODEL,
        )
    except Exception as exc:
        LOGGER.warning("Multilingual translation failed for %s: %s", response_language, exc)
        raise MultilingualResponseError(response_language, "translation_unavailable") from exc
    translated_prose = _PROTECTED_TRANSLATION_PATTERN.sub("", translated).strip()
    if not text_matches_language(translated_prose, response_language):
        LOGGER.warning("Translation output failed language validation for %s", response_language)
        raise MultilingualResponseError(response_language, "translation_unavailable")
    return translated


_CITATION_HEADING_PATTERN = re.compile(
    "|".join(
        re.escape(localized_response_text(language, "verified_source_citations"))
        for language in SUPPORTED_LANGUAGES
    ),
    re.I,
)
_SECTION_TERMS = (
    re.compile(r"\b(?:overview|context|background|contribution|executive|insights?)\b", re.I),
    re.compile(r"\b(?:method|methodology|architecture|design|technical|approach|model)\b", re.I),
    re.compile(r"\b(?:experiment|result|evaluation|dataset|metric|performance)\b", re.I),
    re.compile(r"\b(?:limitation|challenge|weakness|constraint|drawback)\b", re.I),
    re.compile(r"\b(?:implication|application|practical|deployment|future|source|citation)\b", re.I),
)
_EVIDENCE_BOUNDARY_MARKERS = {
    "English": "do not contain explicit evidence",
    "French": "ne contiennent pas de preuve explicite",
    "Spanish": "no contienen evidencia explícita",
    "Tamil": "வெளிப்படையான ஆதாரம் இல்லை",
    "Hindi": "स्पष्ट साक्ष्य नहीं है",
}
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
    response_language = resolve_response_language("", response_language)
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
    fallback: bool = False,
) -> str:
    """Normalize generated prose into five nonempty, grounded sections."""
    response_language = resolve_response_language("", response_language)
    if not isinstance(answer, str) or not answer.strip():
        raise MultilingualResponseError(response_language, "invalid_response")
    if _is_provider_refusal(answer):
        raise MultilingualResponseError(response_language, "provider_refused")
    normalized_answer = answer.casefold().strip(" .*!\n")
    if normalized_answer in {
        str(copy["not_found"]).casefold().strip(" .*!\n")
        for copy in _FALLBACK_COPY.values()
    } | {"information not found in the corpus"}:
        return localized_fallback_message(response_language)
    answer_lines = []
    skipping_citations = False
    for line in answer.splitlines():
        if _CITATION_HEADING_PATTERN.search(line):
            if re.fullmatch(
                rf"\s*(?:#{{1,6}}\s*)?\*{{0,2}}(?:{_CITATION_HEADING_PATTERN.pattern})\*{{0,2}}\s*:?\s*",
                line,
                flags=re.I,
            ):
                skipping_citations = True
                continue
            line = re.sub(
                rf"\s*\*{{0,2}}(?:{_CITATION_HEADING_PATTERN.pattern})\*{{0,2}}\s*:.*$",
                "",
                line,
                flags=re.I,
            )
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
    repeated_sections: dict[str, set[int]] = {}
    for indicated_section, point in points:
        point = point.strip(" -*")
        if not point:
            continue
        key = point.casefold()
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
        repeated_sections.setdefault(key, set()).add(section)
        if key in seen:
            continue
        seen.add(key)
        if len(buckets[section]) >= 3:
            continue
        if len(buckets[section]) >= 3:
            continue
        buckets[section].append(point)
        next_section = (section + 1) % 5

    if any(len(section_indexes) >= 3 for section_indexes in repeated_sections.values()):
        raise MultilingualResponseError(response_language, "invalid_response")

    boundary_marker = _EVIDENCE_BOUNDARY_MARKERS.get(
        response_language,
        _EVIDENCE_BOUNDARY_MARKERS["English"],
    ).casefold()
    grounded_buckets = [
        [point for point in section_points if boundary_marker not in point.casefold()]
        for section_points in buckets
    ]
    headings = _RESPONSE_HEADINGS.get(response_language, _RESPONSE_HEADINGS["English"])
    sections = []
    used_evidence: set[str] = set()
    for index, (heading, section_points) in enumerate(zip(headings, grounded_buckets), start=1):
        if not section_points:
            section_points.append(
                _grounded_section_fallback(index - 1, context_chunks or [], response_language)
            )
        translated_points = [
            _translate_point_if_needed(point, response_language)
            for point in section_points
        ]
        if response_language == "English":
            evidence = _extractive_section_evidence(index - 1, translated_points, context_chunks or [], used_evidence)
            if evidence:
                supplemental = "Retrieved evidence: " + " ".join(evidence)
                if len(translated_points) < 3:
                    translated_points.append(supplemental)
                else:
                    translated_points[-1] = f"{translated_points[-1]} {supplemental}"
        section = f"### {index}. {heading}"
        section += "\n\n" + "\n".join(f"- {point}" for point in translated_points)
        sections.append(section)
    formatted = "\n\n".join(sections)
    if fallback:
        related_notice = localized_response_text(response_language, "related_context_notice")
        if related_notice.casefold() not in formatted.casefold():
            formatted = formatted.replace("\n\n- ", f"\n\n- {related_notice} ", 1)
    if source_citations:
        citation_heading = localized_response_text(response_language, "verified_source_citations")
        formatted += f"\n\n**{citation_heading}**\n\n{source_citations}"
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
        for relevance, _, was_used, sentence in candidates:
            key = sentence.casefold()
            if bool(relevance) != require_section_match or was_used or key in selected_keys:
                continue
            used_evidence.add(key)
            selected.append(sentence)
            selected_keys.add(key)
            current_word_count += len(re.findall(r"\b[\w'-]+\b", sentence))
            if current_word_count >= 100:
                break
        if current_word_count >= 100:
            break
    return selected


def _load_prompt_template() -> str:
    try:
        return _PROMPT_PATH.read_text(encoding="utf-8")
    except OSError:
        return "Question: {question}\nContext: {context}\nAnswer:"


def generation_cache_version() -> str:
    """Fingerprint prompts and language/model settings used to produce cached answers."""
    try:
        prompt_text = _PROMPT_PATH.read_text(encoding="utf-8")
    except OSError:
        prompt_text = ""
    fingerprint = repr(
        (
            prompt_text,
            LANGUAGE_CONFIG,
            GROQ_MODELS,
            HF_MODELS,
            MAX_GENERATION_TOKENS,
            MAX_CONTEXT_CHARS,
            TRANSLATION_MODEL,
        )
    )
    return hashlib.sha256(f"multilingual-v2:{fingerprint}".encode("utf-8")).hexdigest()


def _format_retrieved_evidence(context_chunks: List[Dict[str, object]]) -> str:
    evidence_blocks = []
    for index, chunk in enumerate(context_chunks, start=1):
        text = str(chunk.get("text", "")).strip()
        if not text:
            continue
        source = str(chunk.get("source", chunk.get("document_id", "unknown"))).strip()
        source_name = Path(source).name if source and source != "unknown" else ""
        page = str(chunk.get("page_number", "")).strip()
        section = str(chunk.get("section_type", "")).strip()
        metadata = " | ".join(value for value in (source_name, page, section) if value)
        metadata_label = f" | {metadata}" if metadata else ""
        label = f"[Evidence {index}{metadata_label}]"
        evidence_blocks.append(f"{label}\n{text}")
    return "\n\n".join(evidence_blocks)


def _build_generation_prompt(
    question: str,
    context_text: str,
    instructions: str | None,
    response_language: str,
    fallback: bool,
) -> str:
    del instructions, fallback
    template = _load_prompt_template()
    headings = _RESPONSE_HEADINGS.get(response_language, _RESPONSE_HEADINGS["English"])
    prompt_values = {
        "question": question,
        "context": context_text,
        "response_language": response_language,
        "native_language": LANGUAGE_CONFIG[response_language].native_name,
        **{f"heading_{index}": heading for index, heading in enumerate(headings, start=1)},
    }
    return template.format_map(prompt_values)


def _build_system_prompt(
    response_language: str,
    instructions: str | None,
    fallback: bool,
) -> str:
    language = LANGUAGE_CONFIG[response_language]
    evidence_scope = (
        "Direct evidence is absent. Summarize only closely related evidence, clearly label it as related context, and do not imply that it directly answers the question."
        if fallback
        else "If evidence for a requested section is absent, say so briefly in the requested language; do not fill gaps with outside knowledge."
    )
    custom_instructions = f"\nTask-specific guidance: {instructions}" if instructions else ""
    return (
        "You are an academic research assistant. Answer the user's research question using only the retrieved evidence. "
        f"Write all headings and substantive prose in {language.native_name} ({response_language}). "
        "Be factual, concise, and explicit about uncertainty. Preserve source labels, paper titles, proper names, technical identifiers, "
        "equations, code, and numbers exactly; cite supported claims with the supplied [Evidence N] labels. "
        "When figures or tables are retrieved, synthesize supported trends, comparisons, datasets, and reported values from their captions or accompanying text; never infer unseen visual details. "
        "This is ordinary research summarization: do not substitute generic refusal boilerplate for a supported summary. "
        "Continue to follow applicable safety requirements and refuse only when the request itself requires refusal. "
        f"{evidence_scope} Return exactly five numbered Markdown sections with one or two evidence-based bullets per section, "
        "and no introduction or conclusion outside those bullets."
        f"{custom_instructions}"
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
            self._model_enabled = False
            if self.model_name:
                self._generator = _load_generator(self.model_name)
        return self._generator

    def _extract_answer(
        self,
        question: str,
        context_text: str,
        fallback: bool = False,
        response_language: str = "English",
    ) -> str:
        if not context_text:
            return localized_fallback_message(response_language)

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
            if fallback:
                return " ".join(sentences[:6])
            if response_language == "English":
                return localized_fallback_message(response_language)
            return " ".join(sentences[:6])
        ranked_sentences.sort(key=lambda item: item[0], reverse=True)
        selected = [sentence for _, sentence in ranked_sentences[:6]]
        return " ".join(selected)

    def _validate_or_extract(
        self,
        answer: str,
        question: str,
        context_text: str,
        context_chunks: List[Dict[str, object]],
        response_language: str,
        fallback: bool,
    ) -> str:
        try:
            return _enforce_response_structure(
                answer,
                response_language,
                context_chunks=context_chunks,
                fallback=fallback,
            )
        except MultilingualResponseError as exc:
            if exc.message_key != "invalid_response":
                raise
            LOGGER.warning("Malformed model response; using retrieved evidence for recovery")
            extractive_answer = self._extract_answer(
                question,
                context_text,
                fallback=fallback,
                response_language=response_language,
            )
            return _enforce_response_structure(
                extractive_answer,
                response_language,
                context_chunks=context_chunks,
                fallback=fallback,
            )

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
        prompt_context = _truncate_context(
            _format_retrieved_evidence(context_chunks),
            MAX_CONTEXT_CHARS + 250,
        )
        prompt = _build_generation_prompt(
            question,
            prompt_context,
            instructions,
            response_language,
            fallback,
        )
        system_prompt = _build_system_prompt(response_language, instructions, fallback)
        hf_token = os.getenv("HF_TOKEN")
        groq_api_key = os.getenv("GROQ_API_KEY")
        if (hf_token and hf_token.strip()) or (groq_api_key and groq_api_key.strip()):
            try:
                response_text = generate_response(
                    _truncate_total_prompt(prompt),
                    system_prompt=system_prompt,
                ).strip()
                if response_text:
                    return self._validate_or_extract(
                        response_text,
                        question,
                        context_text,
                        context_chunks,
                        response_language,
                        fallback,
                    )
                LOGGER.warning("Hosted answer generation returned an empty response")
            except MultilingualResponseError:
                raise
            except Exception as exc:
                LOGGER.warning("Hosted generation failed: %s", _redact_error(exc))
            return self._validate_or_extract(
                self._extract_answer(
                    question, context_text, fallback=fallback, response_language=response_language
                ),
                question,
                context_text,
                context_chunks,
                response_language,
                fallback,
            )

        if self._get_generator() is not None:
            try:
                local_prompt = f"System instructions:\n{system_prompt}\n\n{prompt}"
                response = generate_local_fallback(local_prompt, self._generator)
                response_text = response.strip()
                if response_text:
                    return self._validate_or_extract(
                        response_text,
                        question,
                        context_text,
                        context_chunks,
                        response_language,
                        fallback,
                    )
            except MultilingualResponseError:
                raise
            except Exception as exc:
                LOGGER.warning("Text generation failed; using extractive fallback: %s", _redact_error(exc))

        return self._validate_or_extract(
            self._extract_answer(
                question,
                context_text,
                fallback=fallback,
                response_language=response_language,
            ),
            question,
            context_text,
            context_chunks,
            response_language,
            fallback,
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
        context_text = _truncate_context(
            "\n\n".join(str(chunk.get("text", "")) for chunk in context_chunks)
        )

        hf_token = os.getenv("HF_TOKEN")
        groq_api_key = os.getenv("GROQ_API_KEY")
        if (hf_token and hf_token.strip()) or (groq_api_key and groq_api_key.strip()):
            prompt_context = _truncate_context(
                _format_retrieved_evidence(context_chunks),
                MAX_CONTEXT_CHARS + 250,
            )
            prompt = _build_generation_prompt(
                question,
                prompt_context,
                instructions,
                response_language,
                fallback,
            )
            system_prompt = _build_system_prompt(response_language, instructions, fallback)
            try:
                streamed_response = "".join(
                    generate_response_stream(
                        _truncate_total_prompt(prompt),
                        system_prompt=system_prompt,
                    )
                )
            except Exception as exc:
                LOGGER.warning("Hosted streaming generation failed: %s", _redact_error(exc))
                streamed_response = ""
            if not streamed_response.strip():
                yield self.generate_answer(
                    question,
                    context_chunks,
                    instructions=instructions,
                    response_language=response_language,
                    fallback=fallback,
                )
                return
            yield self._validate_or_extract(
                streamed_response,
                question,
                context_text,
                context_chunks,
                response_language,
                fallback,
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

        prompt_context = _truncate_context(
            _format_retrieved_evidence(context_chunks),
            MAX_CONTEXT_CHARS + 250,
        )
        prompt = _build_generation_prompt(
            question,
            prompt_context,
            instructions,
            response_language,
            fallback,
        )
        system_prompt = _build_system_prompt(response_language, instructions, fallback)
        final_prompt = _truncate_total_prompt(
            f"System instructions:\n{system_prompt}\n\n{prompt}"
        )
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
            streamed_response = "".join(streamer)
        finally:
            worker.join()
        if generation_errors:
            LOGGER.warning(
                "Local streaming generation failed: %s",
                _redact_error(generation_errors[0]),
            )
            yield self.generate_answer(
                question,
                context_chunks,
                instructions=instructions,
                response_language=response_language,
                fallback=fallback,
            )
            return
        yield self._validate_or_extract(
            streamed_response,
            question,
            context_text,
            context_chunks,
            response_language,
            fallback,
        )
