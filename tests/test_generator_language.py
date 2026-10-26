import pytest

from src.generator import AnswerGenerator, _format_answer, _generation_instructions
from src import llm as llm_module
from src.llm import SimpleLLM, localized_fallback_message
from src.multilingual import (
    MultilingualResponseError,
    RESPONSE_HEADINGS,
    SUPPORTED_LANGUAGES,
)


def test_generator_uses_canonical_target_language_for_missing_evidence():
    generator = AnswerGenerator()
    expected = f"**{localized_fallback_message('French')}**"

    assert generator.generate("Explain the method", [], response_language="fr") == expected
    assert generator.generate_fast("Explain the method", [], response_language="fr") == expected
    assert generator.generate_stream(
        "Explain the method",
        [],
        stream_consumer=lambda parts: "".join(parts),
        response_language="fr",
    ) == expected


def test_generation_uses_one_canonical_language_for_prompt_and_fallback(monkeypatch):
    generator = AnswerGenerator()
    captured = {}

    def return_missing_evidence(query, chunks, **kwargs):
        captured.update(kwargs)
        return "Information not found in the corpus."

    monkeypatch.setattr(generator.llm, "generate_answer", return_missing_evidence)
    answer = generator.generate(
        "Explain the method",
        [{"text": "Retrieved research context."}],
        response_language="fr",
    )

    assert captured["response_language"] == "French"
    assert "Requested target language: French." in captured["instructions"]
    assert answer == f"**{localized_fallback_message('French')}**"


def test_synthesis_instructions_require_uniform_target_language():
    instructions = _generation_instructions("Methods", False, "French")

    assert "Synthesize only the retrieved research evidence." in instructions
    assert "Requested target language: French." in instructions


def test_generation_prompt_formats_evidence_as_labeled_passages(monkeypatch):
    monkeypatch.delenv("HF_TOKEN", raising=False)
    monkeypatch.delenv("GROQ_API_KEY", raising=False)
    monkeypatch.setattr(llm_module, "MAX_TOTAL_PROMPT_CHARS", 10000)
    captured = {}

    def capture_prompt(prompt, **kwargs):
        captured["prompt"] = prompt
        return [{"generated_text": "### 1. மேலோட்டம்\n- முறை பல ஆதாரங்களை ஒருங்கிணைக்கிறது."}]

    llm = SimpleLLM(model_name="test")
    llm._generator = capture_prompt
    llm.generate_answer(
        "Explain the method",
        [
            {
                "text": "The method combines evidence from multiple passages.",
                "source": "paper.pdf",
                "page_number": 2,
                "section_type": "Methods",
            },
            {"text": "The evaluation compares the resulting summaries."},
        ],
        response_language="Tamil",
    )

    prompt = captured["prompt"]
    assert "[Evidence 1 | paper.pdf | 2 | Methods]" in prompt
    assert "[Evidence 2]" in prompt
    assert "The method combines evidence from multiple passages." in prompt
    assert "The evaluation compares the resulting summaries." in prompt


@pytest.mark.parametrize("language", SUPPORTED_LANGUAGES)
def test_grounded_content_and_source_citations_are_preserved_per_language(language):
    claim = {
        "English": "The method combines evidence from multiple studies.",
        "Tamil": "இந்த முறை பல ஆய்வுகளின் ஆதாரங்களை ஒருங்கிணைக்கிறது.",
        "Hindi": "यह विधि कई अध्ययनों के साक्ष्यों को जोड़ती है।",
        "Spanish": "El método combina pruebas de varios estudios.",
        "French": "La méthode combine les preuves de plusieurs études.",
    }[language]
    chunks = [{
        "text": "The method combines evidence from multiple studies.",
        "source": "2401.12345.pdf",
        "page_number": 2,
        "section_type": "Methods",
    }]
    answer = _format_answer(
        f"### 1. {RESPONSE_HEADINGS[language][0]}\n- {claim} [Evidence 1]",
        chunks,
        "Methods",
        target_language=language,
        source_citations="- [2401.12345 | Page 2](https://arxiv.org/abs/2401.12345)",
    )

    assert claim in answer
    assert f"### 1. {RESPONSE_HEADINGS[language][0]}" in answer
    assert "[2401.12345 | Page 2](https://arxiv.org/abs/2401.12345)" in answer
    assert "I'm sorry, but I can't comply with that request." not in answer
    assert len([line for line in answer.splitlines() if line.startswith("### ")]) == 5


def test_answer_cache_isolated_by_response_language(monkeypatch):
    from src import generator as generator_module

    generator_module._answer_cache.clear()
    calls = []

    def generate_uncached(self, query, chunks, target_language, fallback):
        del self, query, chunks, fallback
        calls.append(target_language)
        return f"answer in {target_language}"

    monkeypatch.setattr(AnswerGenerator, "_generate_uncached", generate_uncached)
    generator = AnswerGenerator()
    query = "What method does the research describe?"
    chunks = [{"text": "The paper evaluates a retrieval method.", "source": "paper.pdf"}]

    english = generator.generate(query, chunks, response_language="English")
    tamil = generator.generate(query, chunks, response_language="Tamil")
    cached_english = generator.generate(query, chunks, response_language="English")

    assert english == cached_english == "answer in English"
    assert tamil == "answer in Tamil"
    assert calls == ["English", "Tamil"]


@pytest.mark.parametrize(
    "answer",
    ["", "I'm sorry, but I can't comply with that request."],
)
def test_empty_and_refusal_responses_are_not_formatted_as_answers(answer):
    with pytest.raises(MultilingualResponseError):
        llm_module._enforce_response_structure(answer, "Tamil")


def test_translation_failure_is_localized_and_a_later_attempt_can_recover(monkeypatch):
    attempts = []

    def translate(text, source_language, target_language, model_name):
        del text, source_language, target_language, model_name
        attempts.append("called")
        if len(attempts) == 1:
            raise RuntimeError("translation model unavailable")
        return "இந்த முறை ஆதாரங்களை ஒருங்கிணைக்கிறது."

    monkeypatch.setattr(llm_module, "_translate_text_cached", translate)

    with pytest.raises(MultilingualResponseError) as error:
        llm_module._translate_point_if_needed(
            "The method combines evidence.",
            "Tamil",
        )
    assert error.value.message_key == "translation_unavailable"
    assert llm_module._translate_point_if_needed(
        "The method combines evidence.",
        "Tamil",
    ) == "இந்த முறை ஆதாரங்களை ஒருங்கிணைக்கிறது."
    assert len(attempts) == 2
