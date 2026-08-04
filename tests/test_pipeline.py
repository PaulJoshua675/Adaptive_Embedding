import pickle
from pathlib import Path

from src.chunk_documents import split_text_into_chunks
from src.extract_pdf import extract_documents_from_path
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
