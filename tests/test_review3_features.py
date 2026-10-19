import pickle

import fitz

from src.chunk_documents import chunk_documents
from src.classifier import QueryClassifier
from src import reranker
from src.multilingual import resolve_response_language
from src.llm import SimpleLLM, _enforce_response_structure
from src.generator import _format_answer
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


def test_answer_generation_prompt_targets_each_supported_language():
    requested_prompts = []
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
        assert "at least 100 words of substantive synthesis" in prompt


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


def test_answer_generation_uses_ten_cleaned_context_chunks():
    assert DEFAULT_TOP_K == 10
    requested_prompts = []
    llm = SimpleLLM(model_name="test")
    llm._generator = lambda prompt, **kwargs: requested_prompts.append(prompt) or [{"generated_text": "Detailed answer."}]
    chunks = [
        {"text": f"Passage {index} reports verified evidence. 1e-4 1e-3 1e-2"}
        for index in range(10)
    ]

    llm.generate_answer("Explain the evidence", chunks)

    assert "Passage 0" in requested_prompts[0]
    assert "Passage 9" in requested_prompts[0]
    assert "1e-4" not in requested_prompts[0]


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