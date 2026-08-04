# Adaptive RAG for Research Paper Question Answering

This project implements the Review 1 baseline for an adaptive retrieval-augmented generation system. The current version:

- extracts text from PDF files when PyMuPDF is available,
- falls back to plain text files in the dataset directory when PDFs are not present,
- chunks documents into overlapping passages,
- creates a simple vector index for retrieval,
- answers questions with a lightweight LLM-style response generator.

The sample corpus now includes 45 research-paper-style documents in the dataset folder so the retrieval pipeline has a richer collection to search over.

## Project structure

- app.py: CLI and optional Streamlit entry point
- config.py: project-level configuration
- src/: implementation modules for extraction, chunking, embeddings, retrieval, and answering
- dataset/: sample research-paper-like documents
- prompts/example_questions.txt: a ready-made list of questions to try
- vector_store/: generated metadata and index artifacts

## Quick start

1. Install dependencies:
   `pip install -r requirements.txt`
2. Download one or more arXiv papers into the dataset directory:
   `python app.py --dataset-dir dataset --arxiv-ids 1706.03762 2401.12345`
3. Build the index:
   `python app.py --dataset-dir dataset`
4. Ask a question:
   `python app.py --query "What does RAG use to answer questions?"`
5. Start the UI:
   `streamlit run app.py -- --streamlit`

## Example questions

You can try the bundled prompts from [prompts/example_questions.txt](prompts/example_questions.txt) or ask questions such as:

- What is the purpose of retrieval-augmented generation in research paper QA?
- How do embedding models improve retrieval for scientific documents?
- Why is chunking important for long academic text?
- How does FAISS support fast nearest-neighbour search?

## Notes

The fallback embedding and retrieval logic is intentionally lightweight so the project can run in a basic environment without requiring heavyweight model downloads.
