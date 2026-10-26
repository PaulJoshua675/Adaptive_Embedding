import json
import pickle
import pytest
from pathlib import Path
from types import SimpleNamespace
from unittest.mock import MagicMock

import fitz

from src.chunk_documents import chunk_documents
from src.classifier import QueryClassifier
from src import llm as llm_module
from src import reranker
from src.multilingual import resolve_response_language
from src.llm import (
    SimpleLLM,
    _RESPONSE_HEADINGS,
    _enforce_response_structure,
    _grounded_section_fallback,
    localized_fallback_message,
    generate_response,
)
from src.generator import _format_answer
from src.generator import AnswerGenerator
from src.extract_pdf import clean_extracted_text
from src.dataset_classifier import _classify_text
from src.preprocessing import extract_pdf_pages
from src import vector_db
from app import _domain_paper_counts
from config import DEFAULT_TOP_K


def test_classifier_fallback_routes_all_intents_without_model_downloads():
    classifier = QueryClassifier()
    classifier._load_zero_shot = lambda: False

    technical = classifier.predict("derive the gradient equation")
    conceptual = classifier.predict("explain why this architecture works")
    entity = classifier.predict("BERT")

    assert (technical["query_type"], technical["embedding_key"]) == ("technical_mathematical", "bge")
    assert (conceptual["query_type"], conceptual["embedding_key"]) == ("conceptual_general", "e5")
    assert (entity["query_type"], entity["embedding_key"]) == ("short_keyword_specific_entity", "minilm")
    assert all(0.0 <= float(route["confidence"]) <= 1.0 for route in (technical, conceptual, entity))


def test_classifier_uses_zero_shot_and_falls_back_when_inference_fails():
    classifier = QueryClassifier(model_name="test-zero-shot")
    classifier._zero_shot = lambda query, candidate_labels, multi_label: {
        "labels": ["technical_mathematical"],
        "scores": [0.91],
    }

    zero_shot_route = classifier.predict("derive the gradient")
    assert zero_shot_route["embedding_key"] == "bge"
    assert zero_shot_route["classifier_model"] == "test-zero-shot"

    classifier._zero_shot = lambda *args, **kwargs: (_ for _ in ()).throw(RuntimeError("model unavailable"))
    classifier._load_fallback = lambda: None
    classifier._predict_fallback = lambda query: ("conceptual_general", 0.75)
    fallback_route = classifier.predict("explain the method")
    assert fallback_route["embedding_key"] == "e5"
    assert fallback_route["classifier_model"] == "tfidf-logistic-regression-fallback"
    assert fallback_route["confidence"] == 0.75


def test_response_language_uses_selection_or_detects_query_script():
    assert resolve_response_language("How does it work?", "Tamil") == "Tamil"
    assert resolve_response_language("¿Cómo funciona?") == "Spanish"
    assert resolve_response_language("Pourquoi ce modèle?") == "French"
    assert resolve_response_language("இந்த முறை எப்படி?") == "Tamil"
    assert resolve_response_language("यह कैसे काम करता है?") == "Hindi"


def test_missing_evidence_fallback_is_localized_for_supported_languages():
    assert "passages récupérés" in localized_fallback_message("French")
    assert "pasajes recuperados" in localized_fallback_message("Spanish")
    assert localized_fallback_message("English") == "Information not found in the corpus."


def test_unsupported_section_fallback_uses_the_requested_language():
    fallback = _grounded_section_fallback(
        2,
        [{"source": "2401.12345.pdf", "section_type": "RESULTS"}],
        "French",
    )

    assert "Les passages récupérés" in fallback
    assert "résultats expérimentaux" in fallback
    assert "retrieved passages" not in fallback
    assert "RESULTS" not in fallback


def test_missing_corpus_response_is_localized():
    answer = _enforce_response_structure("Information not found in the corpus.", "French")

    assert answer == localized_fallback_message("French")


def test_answer_generation_prompt_targets_each_supported_language(monkeypatch):
    requested_prompts = []
    monkeypatch.delenv("HF_TOKEN", raising=False)
    monkeypatch.delenv("GROQ_API_KEY", raising=False)
    monkeypatch.setattr(llm_module, "MAX_TOTAL_PROMPT_CHARS", 5000)
    llm = SimpleLLM(model_name="test")

    def capture_prompt(prompt, **kwargs):
        requested_prompts.append(prompt)
        return [{"generated_text": "Generated response."}]

    llm._generator = capture_prompt
    for language in ("English", "Tamil", "Hindi", "Spanish", "French"):
        answer = llm.generate_answer("Explain the method", [{"text": "Verified research context."}], response_language=language)
        assert "Generated response." in answer
        assert len([line for line in answer.splitlines() if line.startswith("### ")]) == 5
        assert all(
            len([line for line in section.splitlines() if line.startswith("- ")]) <= 3
            for section in answer.split("### ")[1:]
        )

    assert len(requested_prompts) == 5
    for language, prompt in zip(("English", "Tamil", "Hindi", "Spanish", "French"), requested_prompts):
        assert f"Write all answer prose in {language}." in prompt
        assert f"CRITICAL: Write the entire five-section response strictly in {language}." in prompt
        assert "Keep technical paper titles and source citations in English." in prompt
        assert "Translate both the structural subheadings AND all detailed body content" in prompt
        assert "at least 100 words of substantive synthesis" in prompt
        assert "CRITICAL: You must translate ALL 5 structural subheadings" in prompt
        assert "For sections 1, 2, 3, 4, and 5 individually" in prompt
        assert all(f"### {index}. {heading}" in prompt for index, heading in enumerate(_RESPONSE_HEADINGS[language], start=1))
        assert "{overview_header}" not in prompt


def test_local_generation_bounds_total_prompt_and_tokenizer(monkeypatch):
    captured = {}
    monkeypatch.delenv("HF_TOKEN", raising=False)
    monkeypatch.delenv("GROQ_API_KEY", raising=False)

    class FakePipeline:
        tokenizer = SimpleNamespace(eos_token_id=99)

        def __call__(self, prompt, **kwargs):
            captured["prompt"] = prompt
            captured["kwargs"] = kwargs
            return [{"generated_text": "Generated response."}]

    llm = SimpleLLM(model_name="test")
    llm._generator = FakePipeline()
    oversized_prompt = "x" * (llm_module.MAX_TOTAL_PROMPT_CHARS + 100)
    monkeypatch.setattr(llm_module, "_build_generation_prompt", lambda *args: oversized_prompt)
    llm.generate_answer("Question", [{"text": "Relevant context."}])

    assert len(captured["prompt"]) == llm_module.MAX_TOTAL_PROMPT_CHARS
    assert captured["prompt"].endswith("[Prompt truncated for model limits]")
    assert captured["kwargs"] == {
        "max_new_tokens": 256,
        "truncation": True,
        "do_sample": False,
        "pad_token_id": 99,
        "return_full_text": False,
    }


def test_extracted_text_cleaner_removes_plot_noise_but_keeps_prose():
    text = (
        "The validation reward improves after training.\n"
        "1e-4 1e-3 1e-2 1e-1\n"
        "Chart values 1e-4 1e-3 1e-2 1 0 20 40 60 followed by axis labels.\n"
        "60 F1 Validation reward 4 (GPT) 40 2\n"
        "3.14159\n"
        "The model reaches an F1 score of 0.91 on the validation set."
    )

    cleaned = clean_extracted_text(text)

    assert "validation reward improves" in cleaned
    assert "1e-4" not in cleaned
    assert "Chart values" not in cleaned
    assert "60 F1 Validation reward" not in cleaned
    assert "3.14159" not in cleaned
    assert "F1 score of 0.91" in cleaned


def test_dataset_classifier_maps_local_research_keywords():
    categories = _classify_text(
        "This paper presents an image colorization model using deep learning "
        "and evaluates a neural network on object detection."
    )

    assert categories == ["cs.CV", "cs.LG"]


def test_answer_generation_discards_low_similarity_context():
    llm = SimpleLLM(model_name="test")
    llm._generator = lambda *args, **kwargs: (_ for _ in ()).throw(AssertionError("generator should not run"))

    answer = llm.generate_answer(
        "Explain transformer attention",
        [{"text": "The paper studies an unrelated optimization procedure.", "score": 0.05}],
    )

    assert answer == "Information not found in the corpus."


def test_fallback_prompt_requests_related_context_in_full_target_language(monkeypatch):
    requested_prompts = []
    monkeypatch.delenv("HF_TOKEN", raising=False)
    monkeypatch.delenv("GROQ_API_KEY", raising=False)
    llm = SimpleLLM(model_name="test")
    llm._generator = lambda prompt, **kwargs: (
        requested_prompts.append(prompt) or [{"generated_text": "Generated response."}]
    )

    llm.generate_answer(
        "Explain an absent topic",
        [{"text": "Related papers study multilingual retrieval.", "score": 0.05}],
        response_language="Tamil",
        fallback=True,
    )

    prompt = requested_prompts[0]
    assert "The exact answer to this query is not explicitly detailed in the corpus." in prompt
    assert "related research themes, and adjacent findings" in prompt
    assert "Translate both the structural subheadings AND all detailed body content" in prompt
    assert "Write all answer prose in Tamil." in prompt


def test_groq_generation_receives_language_and_fallback_prompt(monkeypatch):
    requested_prompts = []
    monkeypatch.setenv("GROQ_API_KEY", "test-token")
    monkeypatch.setattr(
        "src.llm.generate_response",
        lambda prompt: requested_prompts.append(prompt) or "Generated response.",
    )
    llm = SimpleLLM(model_name="test")
    llm._generator = lambda *args, **kwargs: (_ for _ in ()).throw(
        AssertionError("Groq generation should be preferred when configured")
    )

    llm.generate_answer(
        "Explain an absent topic",
        [{"text": "Related papers study multilingual retrieval.", "score": 0.05}],
        response_language="French",
        fallback=True,
    )

    assert len(requested_prompts) == 1
    assert "Write all answer prose in French." in requested_prompts[0]
    assert "related research themes, and adjacent findings" in requested_prompts[0]


def test_groq_errors_are_logged_without_huggingface_model_retries(monkeypatch, caplog):
    from openai import OpenAIError

    monkeypatch.setenv("GROQ_API_KEY", "test-token")
    client = MagicMock()
    client.chat.completions.create.side_effect = OpenAIError("401 Unauthorized")
    monkeypatch.setattr("src.llm.OpenAI", MagicMock(return_value=client))

    assert generate_response("test prompt") == ""
    assert "401 Unauthorized" in caplog.text


def test_groq_connection_errors_are_logged_and_return_empty_response(monkeypatch, caplog):
    from openai import OpenAIError

    monkeypatch.setenv("GROQ_API_KEY", "test-token")
    client = MagicMock()
    client.chat.completions.create.side_effect = OpenAIError("No provider")
    monkeypatch.setattr("src.llm.OpenAI", MagicMock(return_value=client))

    assert generate_response("test prompt") == ""
    assert "No provider" in caplog.text


def test_paper_id_queries_keep_relevant_passages_from_only_the_named_paper():
    chunks = [
        {"source": "2203.02155.pdf", "text": "The paper reports PPO-ptx evaluation results.", "score": 0.15},
        {"source": "2307.09288.pdf", "text": "The paper reports PPO-ptx evaluation results.", "score": 0.95},
    ]

    relevant = SimpleLLM._relevant_chunks(
        "What does 2203.02155 report about PPO-ptx?",
        chunks,
    )

    assert len(relevant) == 1
    assert relevant[0]["source"] == "2203.02155.pdf"


def test_answer_generation_uses_ten_cleaned_context_chunks(monkeypatch):
    assert DEFAULT_TOP_K == 10
    monkeypatch.delenv("HF_TOKEN", raising=False)
    monkeypatch.delenv("GROQ_API_KEY", raising=False)
    monkeypatch.setattr(llm_module, "MAX_TOTAL_PROMPT_CHARS", 20000)
    requested_prompts = []
    llm = SimpleLLM(model_name="test")
    llm._generator = lambda prompt, **kwargs: requested_prompts.append(prompt) or [{"generated_text": "Detailed answer."}]
    chunks: list[dict[str, object]] = [
        {"text": f"Passage {index} reports verified evidence. 1e-4 1e-3 1e-2"}
        for index in range(10)
    ]

    llm.generate_answer("Explain the evidence", chunks)

    assert "Passage 0" in requested_prompts[0]
    assert "Passage 9" in requested_prompts[0]
    assert "1e-4" not in requested_prompts[0]


def test_answer_generation_truncates_context_before_model_prompt(monkeypatch):
    requested_prompts = []
    monkeypatch.delenv("HF_TOKEN", raising=False)
    monkeypatch.delenv("GROQ_API_KEY", raising=False)
    monkeypatch.setattr(llm_module, "MAX_TOTAL_PROMPT_CHARS", 20000)
    llm = SimpleLLM(model_name="test")
    llm._generator = lambda prompt, **kwargs: requested_prompts.append(prompt) or [
        {"generated_text": "Detailed answer."}
    ]

    llm.generate_answer("Explain the evidence", [{"text": "context " * 600}])

    assert "[Context truncated to fit model limit...]" in requested_prompts[0]
    assert "context " * 400 not in requested_prompts[0]
    assert llm_module.MAX_CONTEXT_CHARS == 500
    assert len(llm_module._truncate_context("x" * 1500)) == 500


def test_answer_formatter_omits_placeholder_points_and_empty_sections():
    formatted = _format_answer(
        "### Executive Insights\nAttention weights combine token representations.\n\n"
        "### Technical Breakdown\n* **Point 1:** Attention weights combine token representations.",
        [{"text": "Attention weights combine token representations."}],
        "METHODS",
    )

    assert "Point 1" not in formatted
    assert "Additional metric" not in formatted
    assert "Attention weights" in formatted
    assert len([line for line in formatted.splitlines() if line.startswith("### ")]) == 5


def test_answer_formatter_limits_each_section_and_cleans_axis_noise():
    answer = "\n".join(
        [
            "### 1. Overview & Core Context",
            "- The paper presents a retrieval method.",
            "- It improves evidence selection.",
            "- It targets research question answering.",
            "- This fourth point must be trimmed.",
            "### 2. Methodology & Architecture",
            "- The model uses an encoder architecture.",
            "- 0.1 0.2 0.3 0.4",
            "- It ranks retrieved passages.",
        ]
    )

    formatted = _format_answer(answer, [], "METHODS")

    headings = [line for line in formatted.splitlines() if line.startswith("### ")]
    sections = formatted.split("### ")[1:]
    assert len(headings) == 5
    assert headings == [
        "### 1. Overview & Core Context",
        "### 2. Methodology & Architecture",
        "### 3. Experimental Results",
        "### 4. Limitations & Challenges",
        "### 5. Practical Implications",
    ]
    assert all(len([line for line in section.splitlines() if line.startswith("- ")]) <= 3 for section in sections)
    assert "fourth point" not in formatted.lower()
    assert "0.1 0.2 0.3 0.4" not in formatted


def test_answer_formatter_fills_empty_sections_and_keeps_citations_in_footer():
    formatted = _enforce_response_structure(
        "### 1. Overview & Core Context\n- The method combines two encoders.\n"
        "### 2. Methodology & Architecture\n- **Verified Source Citations:** [untrusted](https://example.invalid)\n",
        source_citations="[2401.00001 | Page 2 | Section: METHODS](https://arxiv.org/abs/2401.00001)",
        context_chunks=[
            {
                "source": "2401.00001.pdf",
                "section_type": "METHODS",
                "text": "The method combines two encoders.",
            }
        ],
    )

    sections, citations = formatted.split("**Verified Source Citations**", 1)
    assert len([line for line in sections.splitlines() if line.startswith("### ")]) == 5
    assert all(
        any(line.startswith("- ") for line in section.splitlines())
        for section in sections.split("### ")[1:]
    )
    assert "untrusted" not in sections
    assert "[2401.00001 | Page 2" in citations
    assert "retrieved passages from 2401.00001" in sections


def test_curated_research_questions_include_two_clickable_examples_per_domain():
    from config import DOMAIN_EXAMPLE_QUESTIONS

    assert set(DOMAIN_EXAMPLE_QUESTIONS) == {"cs.CV", "cs.CL", "cs.LG", "cs.AI", "cs.IR"}
    assert sum(len(questions) for questions in DOMAIN_EXAMPLE_QUESTIONS.values()) >= 10
    assert all(len(questions) >= 2 for questions in DOMAIN_EXAMPLE_QUESTIONS.values())


def test_sidebar_domain_counts_use_unique_manifest_papers(tmp_path):
    dataset_dir = tmp_path / "dataset"
    vector_store_dir = tmp_path / "vector_store"
    dataset_dir.mkdir()
    vector_store_dir.mkdir()
    (dataset_dir / "2401.00001.pdf").touch()
    (dataset_dir / "2401.00002.pdf").touch()
    (dataset_dir / "arxiv_metadata.json").write_text(
        '{"2401.00001": {"categories": ["cs.CV", "cs.AI"]}, '
        '"2401.00002": {"categories": ["cs.LG", "cs.IR", "cs.NE"]}}',
        encoding="utf-8",
    )

    counts, unclassified = _domain_paper_counts(dataset_dir, vector_store_dir)

    assert counts["cs.CV"] == 1
    assert counts["cs.AI"] == 1
    assert counts["cs.LG"] == 1
    assert counts["cs.IR"] == 1
    assert unclassified == 0


def test_sidebar_domain_counts_merge_indexed_tags_and_custom_uploads(tmp_path):
    dataset_dir = tmp_path / "dataset"
    vector_store_dir = tmp_path / "vector_store"
    dataset_dir.mkdir()
    vector_store_dir.mkdir()
    (dataset_dir / "2401.00001.pdf").touch()
    (dataset_dir / "2401.00002.pdf").touch()
    custom_upload_dir = tmp_path / "data" / "runtime_uploads"
    custom_upload_dir.mkdir(parents=True)
    (custom_upload_dir / "custom-upload.pdf").touch()
    (vector_store_dir / "metadata.pkl").write_bytes(
        pickle.dumps(
            {
                "chunks": [
                    {"source": str(dataset_dir / "2401.00002.pdf"), "categories": ["cs.LG"]},
                ]
            }
        )
    )
    (dataset_dir / "arxiv_metadata.json").write_text(
        '{"2401.00001": {"categories": ["cs.CV"]}}',
        encoding="utf-8",
    )

    counts, unclassified = _domain_paper_counts(dataset_dir, vector_store_dir)

    assert counts["cs.CV"] == 1
    assert counts["cs.LG"] == 1
    assert unclassified == 1


def test_long_document_gets_summary_chunk_without_losing_page_chunks():
    documents = [
        {
            "id": f"paper_page_{page}_section_0",
            "source": "paper.pdf",
            "text": f"Page {page} describes a research method. The method uses retrieved evidence.",
            "page_number": page,
            "title": "Research paper",
        }
        for page in range(1, 56)
    ]

    chunks = chunk_documents(documents, chunk_size=100, overlap=10)
    summary_chunks = [chunk for chunk in chunks if chunk.get("is_summary") == 1]

    assert len(summary_chunks) == 1
    assert summary_chunks[0]["section_type"] == "DOCUMENT SUMMARY"
    assert summary_chunks[0]["page_number"] == 1
    assert len([chunk for chunk in chunks if chunk.get("is_summary") != 1]) == 55
    assert "Page 1" in summary_chunks[0]["text"]
    assert "Page 55" in summary_chunks[0]["text"]


def test_pdf_figure_and_caption_metadata_reaches_chunks(tmp_path):
    pdf_path = tmp_path / "figure_paper.pdf"
    document = fitz.open()
    page = document.new_page()
    page.insert_text((48, 72), "Figure 1: Results on the TruthfulQA dataset")
    page.insert_text((48, 100), "On TruthfulQA, the model score rises from 52 to 61 percent.")
    page.insert_text((48, 120), "Table 2: GSM8K benchmark scores")
    page.insert_text((48, 140), "On GSM8K, accuracy reaches 80 percent for the proposed model.")
    image = fitz.Pixmap(fitz.csRGB, fitz.IRect(0, 0, 48, 32), False)
    image.clear_with(0x2D6A73)
    page.insert_image(fitz.Rect(60, 130, 300, 290), pixmap=image)
    document.save(pdf_path)
    document.close()

    pages = extract_pdf_pages(pdf_path, tmp_path / "figures")
    chunks = chunk_documents(pages, chunk_size=100, overlap=10)

    assert pages[0]["figures"][0]["caption"] == "Results on the TruthfulQA dataset"
    assert Path(pages[0]["figures"][0]["file_path"]).is_file()
    assert chunks[0]["figures"][0]["file_path"] == pages[0]["figures"][0]["file_path"]
    figure_chunk = next(chunk for chunk in chunks if chunk.get("visual_type") == "figure")
    table_chunk = next(chunk for chunk in chunks if chunk.get("visual_type") == "table")
    assert figure_chunk["caption"] == "Results on the TruthfulQA dataset"
    assert "52 to 61 percent" in figure_chunk["text"]
    assert "TruthfulQA" in figure_chunk["keywords"]
    assert "dataset:truthfulqa" in figure_chunk["semantic_tags"]
    assert "GSM8K" in table_chunk["keywords"]
    assert "80 percent" in table_chunk["text"]


def test_generation_prompt_requests_figure_and_table_synthesis():
    visual_prompt = llm_module._build_generation_prompt(
        "What does the result show?",
        "Figure 4: Results on the TruthfulQA dataset. Scores rise from 52 to 61 percent.",
        None,
        "English",
        False,
    )
    text_prompt = llm_module._build_generation_prompt(
        "What does the result show?",
        "The reported scores rise from 52 to 61 percent.",
        None,
        "English",
        False,
    )

    assert "Explicitly synthesize and narrate its key trends, scores" in visual_prompt
    assert "never infer unseen visual details" in visual_prompt
    assert "Explicitly synthesize and narrate its key trends, scores" not in text_prompt


def test_fast_fallback_synthesizes_low_similarity_context():
    context = "A related study analyzes retrieval calibration across multilingual research collections."

    answer = AnswerGenerator().generate_fast(
        "What is the effect of ocean tides on lunar geology?",
        [{"text": context, "score": 0.05}],
        fallback=True,
    )

    assert answer != "**Information not found in the corpus.**"
    assert "related study" in answer


def test_llm_dead_end_is_replaced_with_related_fallback_context(monkeypatch):
    monkeypatch.delenv("GROQ_API_KEY", raising=False)
    monkeypatch.delenv("HUGGINGFACEHUB_API_TOKEN", raising=False)
    generator = AnswerGenerator()
    generator.llm._generator = lambda *args, **kwargs: [
        {"generated_text": "Information not found in the corpus."}
    ]
    context = "Related studies evaluate multilingual retrieval across scientific collections."

    answer = generator.generate(
        "What is the effect of ocean tides on lunar geology?",
        [{"text": context, "score": 0.05}],
        fallback=True,
    )

    assert answer != "**Information not found in the corpus.**"
    assert "Related studies evaluate multilingual retrieval" in answer


def test_55_page_pdf_extracts_hierarchical_summary_chunks(tmp_path):
    pdf_path = tmp_path / "long_research.pdf"
    document = fitz.open()
    for page_number in range(1, 56):
        page = document.new_page()
        page.insert_text(
            (48, 72),
            f"Page {page_number} describes a research method. The method uses retrieved evidence.",
        )
    document.save(pdf_path)
    document.close()

    pages = extract_pdf_pages(pdf_path, tmp_path / "figures")
    chunks = chunk_documents(pages, chunk_size=100, overlap=10)
    summaries = [chunk for chunk in chunks if chunk.get("is_summary") == 1]

    assert len({page["page_number"] for page in pages}) == 55
    assert len(summaries) == 1
    assert summaries[0]["section_type"] == "DOCUMENT SUMMARY"
    assert "Page 1" in summaries[0]["text"]
    assert "Page 55" in summaries[0]["text"]


def test_reranker_preserves_retrieval_order_without_local_weights(monkeypatch):
    monkeypatch.setattr(reranker, "_load_cross_encoder", lambda *args: (_ for _ in ()).throw(OSError("weights missing")))
    passages = [{"text": f"passage {index}", "id": index} for index in range(4)]

    instance = reranker.PassageReranker(model_name="local/test-reranker")
    result = instance.rerank("query", passages, top_k=2)

    assert instance.model is None
    assert [passage["id"] for passage in result] == [0, 1]


def test_vector_store_appends_without_rebuilding_trained_index(monkeypatch):
    store = vector_db.FaissVectorStore(dimension=4)
    store.add([{"text": "first passage"}], [[1.0, 0.0, 0.0, 0.0]])
    initial_index = store.index
    monkeypatch.setattr(vector_db, "_new_index", lambda *args: (_ for _ in ()).throw(AssertionError("index rebuilt")))

    store.add([{"text": "second passage"}], [[0.0, 1.0, 0.0, 0.0]])

    assert store.index is initial_index
    assert store.index.ntotal == 2
    assert store.search([0.0, 1.0, 0.0, 0.0], top_k=1)[0][1]["text"] == "second passage"