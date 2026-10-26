"""Utilities for selecting the answer language independently of source language."""

from __future__ import annotations

import re
from dataclasses import dataclass


@dataclass(frozen=True)
class LanguageConfiguration:
    native_name: str
    translation_code: str
    aliases: tuple[str, ...]
    headings: tuple[str, ...]
    markers: tuple[str, ...] = ()


LANGUAGE_CONFIG = {
    "English": LanguageConfiguration(
        native_name="English",
        translation_code="eng_Latn",
        aliases=("en", "english"),
        headings=(
            "Overview & Core Context",
            "Methodology & Architecture",
            "Experimental Results",
            "Limitations & Challenges",
            "Practical Implications",
        ),
        markers=("the", "and", "with", "from", "evidence", "results"),
    ),
    "Tamil": LanguageConfiguration(
        native_name="தமிழ்",
        translation_code="tam_Taml",
        aliases=("ta", "tamil", "தமிழ்"),
        headings=(
            "மேலோட்டம் மற்றும் அடிப்படைச் சூழல்",
            "முறையியல் மற்றும் கட்டமைப்பு",
            "சோதனை முடிவுகள்",
            "வரம்புகள் மற்றும் சவால்கள்",
            "நடைமுறை விளைவுகள்",
        ),
    ),
    "Hindi": LanguageConfiguration(
        native_name="हिन्दी",
        translation_code="hin_Deva",
        aliases=("hi", "hindi", "हिन्दी", "हिंदी"),
        headings=(
            "अवलोकन और मूल संदर्भ",
            "कार्यप्रणाली और संरचना",
            "प्रायोगिक परिणाम",
            "सीमाएँ और चुनौतियाँ",
            "व्यावहारिक निहितार्थ",
        ),
    ),
    "Spanish": LanguageConfiguration(
        native_name="español",
        translation_code="spa_Latn",
        aliases=("es", "spanish", "español"),
        headings=(
            "Panorama y contexto central",
            "Metodología y arquitectura",
            "Resultados experimentales",
            "Limitaciones y desafíos",
            "Implicaciones prácticas",
        ),
        markers=(
            "el", "los", "las", "una", "unos", "unas", "del", "para", "pero",
            "también", "resultados", "según", "mediante", "utiliza", "evaluación",
            "datos", "método", "limitaciones", "modelo", "resultado",
        ),
    ),
    "French": LanguageConfiguration(
        native_name="français",
        translation_code="fra_Latn",
        aliases=("fr", "french", "français"),
        headings=(
            "Vue d’ensemble et contexte central",
            "Méthodologie et architecture",
            "Résultats expérimentaux",
            "Limites et défis",
            "Implications pratiques",
        ),
        markers=(
            "le", "les", "des", "une", "dans", "pour", "avec", "mais", "sont",
            "cette", "ces", "résultats", "selon", "utilise", "données",
            "méthode", "méthodologie", "évaluation", "limites", "modèle",
            "résultat",
        ),
    ),
}
SUPPORTED_LANGUAGES = tuple(LANGUAGE_CONFIG)
RESPONSE_LANGUAGE_OPTIONS = (*SUPPORTED_LANGUAGES, "Detect from query")
RESPONSE_HEADINGS = {
    language: configuration.headings
    for language, configuration in LANGUAGE_CONFIG.items()
}
LANGUAGE_ALIASES = {
    alias.casefold(): language
    for language, configuration in LANGUAGE_CONFIG.items()
    for alias in (language, configuration.native_name, *configuration.aliases)
}

_RESPONSE_UI_COPY = {
    "English": {
        "ml_route": "ML route",
        "embedding_backend": "Embedding backend",
        "retrieved_passages": "Retrieved passages",
        "sources": "Sources",
        "verified_source_citations": "Verified Source Citations",
        "page": "Page",
        "page_unavailable": "page unavailable",
        "unknown_source": "Unknown source",
        "retrieval_evaluation": "Retrieval evaluation",
        "accuracy": "Accuracy",
        "precision": "Precision",
        "recall": "Recall",
        "f1_score": "F1-Score",
        "intent_classification_time": "Intent Classification Time (ms)",
        "faiss_index_lookup_time": "FAISS Index Lookup Time (ms)",
        "cross_encoder_reranking_time": "Cross-Encoder Re-ranking Time (ms)",
        "answer_generation_time": "LLM / Multi-Lingual Answer Generation Time (ms)",
        "original_rag_pipeline": "Original RAG pipeline",
        "timing_breakdown": "Timing breakdown (ms)",
        "cached_response_lookup": "Cached response lookup",
        "total_query_round_trip": "Total query round trip",
        "uploaded_index_not_ready": "The uploaded document index is not ready. Upload the document again before querying.",
        "answer_request_failed": "Unable to retrieve a grounded answer.",
        "translation_unavailable": "The multilingual translation model is unavailable; no answer was displayed.",
        "provider_refused": "The language model declined to answer this research request.",
        "invalid_response": "The language model returned an empty or malformed answer. Please try again.",
        "related_context_notice": "Related context only: the retrieved passages do not directly answer the question.",
    },
    "Tamil": {
        "ml_route": "இயந்திரக் கற்றல் வழித்தடம்",
        "embedding_backend": "உட்பொதிவு பின்தளம்",
        "retrieved_passages": "மீட்டெடுக்கப்பட்ட பகுதிகள்",
        "sources": "ஆதாரங்கள்",
        "verified_source_citations": "சரிபார்க்கப்பட்ட மூல மேற்கோள்கள்",
        "page": "பக்கம்",
        "page_unavailable": "பக்கத் தகவல் இல்லை",
        "unknown_source": "அறியப்படாத மூலம்",
        "retrieval_evaluation": "மீட்டெடுப்பு மதிப்பீடு",
        "accuracy": "துல்லியம்",
        "precision": "துல்லிய விகிதம்",
        "recall": "மீட்டெடுப்பு விகிதம்",
        "f1_score": "F1 மதிப்பெண்",
        "intent_classification_time": "நோக்க வகைப்படுத்தல் நேரம் (மி.வி.)",
        "faiss_index_lookup_time": "FAISS குறியீட்டுத் தேடல் நேரம் (மி.வி.)",
        "cross_encoder_reranking_time": "குறுக்கு-குறியாக்கி மறுவரிசை நேரம் (மி.வி.)",
        "answer_generation_time": "பதில் உருவாக்க நேரம் (மி.வி.)",
        "original_rag_pipeline": "அசல் RAG செயல்முறை",
        "timing_breakdown": "நேரப் பிரிப்பு (மி.வி.)",
        "cached_response_lookup": "தற்காலிகச் சேமிப்பு பதில் தேடல்",
        "total_query_round_trip": "மொத்த வினா சுற்று நேரம்",
        "uploaded_index_not_ready": "பதிவேற்றிய ஆவணக் குறியீடு தயாராக இல்லை. வினா கேட்பதற்கு முன் ஆவணத்தை மீண்டும் பதிவேற்றவும்.",
        "answer_request_failed": "ஆதாரங்களுடன் கூடிய பதிலைப் பெற முடியவில்லை.",
        "translation_unavailable": "பன்மொழி மொழிபெயர்ப்பு மாதிரி கிடைக்கவில்லை; பதில் காட்டப்படவில்லை.",
        "provider_refused": "மொழி மாதிரி இந்த ஆய்வுக் கேள்விக்குப் பதிலளிக்க மறுத்தது.",
        "invalid_response": "மொழி மாதிரி காலியான அல்லது தவறான பதிலை வழங்கியது. மீண்டும் முயற்சிக்கவும்.",
        "related_context_notice": "தொடர்புடைய சூழல் மட்டும்: மீட்டெடுக்கப்பட்ட பகுதிகள் கேள்விக்கு நேரடியாகப் பதிலளிக்கவில்லை.",
    },
    "Hindi": {
        "ml_route": "एमएल मार्ग",
        "embedding_backend": "एम्बेडिंग बैकएंड",
        "retrieved_passages": "प्राप्त अंश",
        "sources": "स्रोत",
        "verified_source_citations": "सत्यापित स्रोत उद्धरण",
        "page": "पृष्ठ",
        "page_unavailable": "पृष्ठ उपलब्ध नहीं",
        "unknown_source": "अज्ञात स्रोत",
        "retrieval_evaluation": "पुनर्प्राप्ति मूल्यांकन",
        "accuracy": "सटीकता",
        "precision": "परिशुद्धता",
        "recall": "रिकॉल",
        "f1_score": "F1-स्कोर",
        "intent_classification_time": "इरादा वर्गीकरण समय (ms)",
        "faiss_index_lookup_time": "FAISS इंडेक्स खोज समय (ms)",
        "cross_encoder_reranking_time": "क्रॉस-एनकोडर पुनःक्रमण समय (ms)",
        "answer_generation_time": "उत्तर निर्माण समय (ms)",
        "original_rag_pipeline": "मूल RAG पाइपलाइन",
        "timing_breakdown": "समय विवरण (ms)",
        "cached_response_lookup": "कैश्ड उत्तर खोज",
        "total_query_round_trip": "कुल क्वेरी राउंड ट्रिप",
        "uploaded_index_not_ready": "अपलोड किए गए दस्तावेज़ का इंडेक्स तैयार नहीं है। प्रश्न पूछने से पहले दस्तावेज़ फिर से अपलोड करें।",
        "answer_request_failed": "साक्ष्य-आधारित उत्तर प्राप्त नहीं किया जा सका।",
        "translation_unavailable": "बहुभाषी अनुवाद मॉडल उपलब्ध नहीं है; उत्तर प्रदर्शित नहीं किया गया।",
        "provider_refused": "भाषा मॉडल ने इस शोध प्रश्न का उत्तर देने से इनकार किया।",
        "invalid_response": "भाषा मॉडल ने खाली या अमान्य उत्तर दिया। कृपया फिर से प्रयास करें।",
        "related_context_notice": "केवल संबंधित संदर्भ: प्राप्त अंश प्रश्न का सीधे उत्तर नहीं देते।",
    },
    "Spanish": {
        "ml_route": "Ruta de ML",
        "embedding_backend": "Backend de embeddings",
        "retrieved_passages": "Pasajes recuperados",
        "sources": "Fuentes",
        "verified_source_citations": "Citas de fuentes verificadas",
        "page": "Página",
        "page_unavailable": "página no disponible",
        "unknown_source": "Fuente desconocida",
        "retrieval_evaluation": "Evaluación de recuperación",
        "accuracy": "Exactitud",
        "precision": "Precisión",
        "recall": "Exhaustividad",
        "f1_score": "Puntuación F1",
        "intent_classification_time": "Tiempo de clasificación de intención (ms)",
        "faiss_index_lookup_time": "Tiempo de búsqueda del índice FAISS (ms)",
        "cross_encoder_reranking_time": "Tiempo de reordenación del cross-encoder (ms)",
        "answer_generation_time": "Tiempo de generación de respuesta (ms)",
        "original_rag_pipeline": "Canalización RAG original",
        "timing_breakdown": "Desglose de tiempos (ms)",
        "cached_response_lookup": "Búsqueda de respuesta en caché",
        "total_query_round_trip": "Tiempo total de ida y vuelta de la consulta",
        "uploaded_index_not_ready": "El índice del documento cargado no está listo. Vuelve a cargar el documento antes de consultarlo.",
        "answer_request_failed": "No se pudo recuperar una respuesta fundamentada.",
        "translation_unavailable": "El modelo de traducción multilingüe no está disponible; no se mostró ninguna respuesta.",
        "provider_refused": "El modelo de lenguaje rechazó esta consulta de investigación.",
        "invalid_response": "El modelo de lenguaje devolvió una respuesta vacía o mal formada. Inténtalo de nuevo.",
        "related_context_notice": "Solo contexto relacionado: los pasajes recuperados no responden directamente a la pregunta.",
    },
    "French": {
        "ml_route": "Routage ML",
        "embedding_backend": "Backend d’incorporation",
        "retrieved_passages": "Passages récupérés",
        "sources": "Sources",
        "verified_source_citations": "Citations des sources vérifiées",
        "page": "Page",
        "page_unavailable": "page indisponible",
        "unknown_source": "Source inconnue",
        "retrieval_evaluation": "Évaluation de la recherche",
        "accuracy": "Exactitude",
        "precision": "Précision",
        "recall": "Rappel",
        "f1_score": "Score F1",
        "intent_classification_time": "Temps de classification de l’intention (ms)",
        "faiss_index_lookup_time": "Temps de recherche dans l’index FAISS (ms)",
        "cross_encoder_reranking_time": "Temps de reclassement du cross-encodeur (ms)",
        "answer_generation_time": "Temps de génération de la réponse (ms)",
        "original_rag_pipeline": "Pipeline RAG d’origine",
        "timing_breakdown": "Détail des temps (ms)",
        "cached_response_lookup": "Recherche de réponse en cache",
        "total_query_round_trip": "Temps total de traitement de la requête",
        "uploaded_index_not_ready": "L’index du document importé n’est pas prêt. Importez à nouveau le document avant de lancer une requête.",
        "answer_request_failed": "Impossible de récupérer une réponse étayée.",
        "translation_unavailable": "Le modèle de traduction multilingue est indisponible ; aucune réponse n’a été affichée.",
        "provider_refused": "Le modèle de langage a refusé de répondre à cette question de recherche.",
        "invalid_response": "Le modèle de langage a renvoyé une réponse vide ou mal formée. Veuillez réessayer.",
        "related_context_notice": "Contexte connexe uniquement : les passages récupérés ne répondent pas directement à la question.",
    },
}


def detect_language(text: str) -> str:
    """Infer one supported language from script and common function words."""
    if re.search(r"[\u0B80-\u0BFF]", text):
        return "Tamil"
    if re.search(r"[\u0900-\u097F]", text):
        return "Hindi"
    lowered = f" {text.lower()} "
    scores = {
        language: sum(
            len(re.findall(rf"\b{re.escape(marker)}\b", lowered))
            for marker in configuration.markers
        )
        for language, configuration in LANGUAGE_CONFIG.items()
        if language in {"Spanish", "French"}
    }
    if re.search(r"[áíóúñ¿¡]", lowered):
        scores["Spanish"] += 2
    if re.search(r"[àâçèêëîïôûùüÿœ]", lowered):
        scores["French"] += 2
    if max(scores.values(), default=0) > 0:
        return max(scores, key=scores.get)
    return "English"


def text_matches_language(text: str, response_language: str) -> bool:
    """Check that substantive prose uses the requested script or language markers."""
    configuration = LANGUAGE_CONFIG[response_language]
    letters = re.findall(r"[^\W\d_]", text, flags=re.UNICODE)
    if not letters:
        return True
    if response_language in {"Tamil", "Hindi"}:
        script = r"[\u0B80-\u0BFF]" if response_language == "Tamil" else r"[\u0900-\u097F]"
        script_letters = len(re.findall(script, text))
        return script_letters / len(letters) >= 0.35
    if response_language in {"Spanish", "French"}:
        lowered = f" {text.casefold()} "
        marker_count = sum(
            bool(re.search(rf"\b{re.escape(marker)}\b", lowered))
            for marker in configuration.markers
        )
        accent = (
            r"[áíóúñ¿¡]"
            if response_language == "Spanish"
            else r"[àâçèêëîïôûùüÿœ]"
        )
        french_acute_e = response_language == "French" and marker_count > 0 and "é" in text.casefold()
        return marker_count > 0 or bool(re.search(accent, text, re.I)) or french_acute_e
    if detect_language(text) in {"Tamil", "Hindi", "Spanish", "French"}:
        return False
    return True


def resolve_response_language(query: str, selection: str | None = None) -> str:
    """Resolve a user-selected language or fall back to query-language detection."""
    if selection:
        normalized = LANGUAGE_ALIASES.get(selection.strip().casefold())
        if normalized:
            return normalized
        if selection.strip().casefold() not in {"auto", "detect", "detect from query"}:
            raise ValueError(f"Unsupported response language: {selection}")
    return detect_language(query)


def localized_response_text(response_language: str, key: str) -> str:
    """Return a localized label used in response metadata and notices."""
    language = LANGUAGE_ALIASES.get(response_language.strip().casefold())
    if language is None:
        raise ValueError(f"Unsupported response language: {response_language}")
    copy = _RESPONSE_UI_COPY[language]
    return copy[key]


class MultilingualResponseError(RuntimeError):
    """A safe, localized response error suitable for display without provider details."""

    def __init__(self, response_language: str, message_key: str) -> None:
        self.response_language = response_language
        self.message_key = message_key
        super().__init__(localized_response_text(response_language, message_key))