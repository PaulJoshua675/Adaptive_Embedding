import pickle
from pathlib import Path

from src.chunk_documents import split_text_into_chunks
from src.embedding_model import EmbeddingModel
from src.extract_pdf import extract_documents_from_path
from src.llm import SimpleLLM
from src.rag_pipeline import build_and_save_index, answer_query


def test_split_text_into_chunks_creates_overlapping_chunks():
    text = "word " * 250
    chunks = split_text_into_chunks(text, chunk_size=40, overlap=10)
    assert len(chunks) > 1
    assert all(len(chunk["text"].split()) <= 40 for chunk in chunks)


def test_extract_documents_from_path_reads_txt_files(tmp_path):
    dataset_dir = tmp_path / "dataset"
    dataset_dir.mkdir()
    sample_file = dataset_dir / "paper.txt"
    sample_file.write_text("RAG systems improve retrieval by using external knowledge.", encoding="utf-8")

    documents = extract_documents_from_path(dataset_dir)

    assert len(documents) == 1
    assert documents[0]["source"].endswith("paper.txt")
    assert "RAG" in documents[0]["text"]


def test_embedding_model_reuses_shared_vocabulary_for_queries():
    model = EmbeddingModel(model_name="simple-bow")
    texts = ["RAG retrieves passages to answer questions", "The model uses retrieved context"]
    model.fit(texts)

    query_embedding = model.embed_text("What does RAG retrieve to answer questions?")
    chunk_embedding = model.embed_texts(texts)[0]

    assert len(query_embedding) == len(chunk_embedding)
    assert sum(query_embedding) >= 0


def test_build_and_save_index_and_answer_query(tmp_path):
    dataset_dir = tmp_path / "dataset"
    dataset_dir.mkdir()
    (dataset_dir / "paper.txt").write_text(
        "Retrieval-Augmented Generation uses retrieved passages to answer questions about research papers.",
        encoding="utf-8",
    )

    index_dir = tmp_path / "vector_store"
    build_and_save_index(dataset_dir, index_dir)

    metadata_path = index_dir / "metadata.pkl"
    assert metadata_path.exists()

    with metadata_path.open("rb") as handle:
        metadata = pickle.load(handle)

    assert len(metadata["chunks"]) > 0

    answer = answer_query("What does RAG use to answer questions?", index_dir)
    assert "RAG" in answer or "retrieved" in answer.lower()


def test_load_index_builds_index_from_local_temp_dataset(tmp_path):
    temp_root = tmp_path / "uploaded_pdf_temp"
    dataset_dir = temp_root / "dataset"
    dataset_dir.mkdir(parents=True)
    (dataset_dir / "paper.txt").write_text(
        "The temporary dataset is indexed and can be queried using the local vector store.",
        encoding="utf-8",
    )

    index_dir = temp_root / "vector_store"
    from src.rag_pipeline import load_index

    metadata = load_index(index_dir)
    assert metadata["chunks"]
    answer = answer_query("What can be queried?", index_dir)
    assert "temporary" in answer.lower() or "indexed" in answer.lower()


def test_generate_answer_uses_context_to_form_a_direct_response():
    llm = SimpleLLM()
    answer = llm.generate_answer(
        "What does RAG use to answer questions?",
        [{"text": "RAG uses retrieved passages to answer questions about research papers."}],
    )

    assert "retrieved passages" in answer.lower()
    assert "based on the retrieved context" not in answer.lower()


def test_build_and_save_index_falls_back_to_extracted_text_for_pdfs(tmp_path, monkeypatch):
    dataset_dir = tmp_path / "dataset"
    dataset_dir.mkdir(parents=True)
    (dataset_dir / "paper.pdf").write_bytes(b"%PDF-1.4\n")
    (tmp_path / "extracted_text" / "paper.txt").parent.mkdir(parents=True, exist_ok=True)
    (tmp_path / "extracted_text" / "paper.txt").write_text(
        "Fallback text from extracted text should be indexed for PDFs.",
        encoding="utf-8",
    )

    monkeypatch.setattr("src.extract_pdf.extract_text_from_pdf", lambda path: "")

    index_dir = tmp_path / "vector_store"
    metadata = build_and_save_index(dataset_dir, index_dir)

    assert metadata["chunks"]
    assert any("Fallback text" in str(chunk["text"]) for chunk in metadata["chunks"])
