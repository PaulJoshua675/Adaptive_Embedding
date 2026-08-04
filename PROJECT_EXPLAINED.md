# Adaptive Embedding Selection for Research Paper QA

## Overview

This project implements a retrieval-augmented generation (RAG) system for answering questions about research papers. The baseline version uses a single embedding model and a vector store to retrieve relevant passages from a collection of research papers. The project is designed so that the corpus can be expanded with real arXiv papers instead of placeholder text files.

## Core objectives

1. Download real research papers from arXiv.
2. Extract text from downloaded PDFs.
3. Chunk long papers into smaller passages.
4. Create embeddings and store them in a vector database.
5. Retrieve the most relevant chunks for a user query.
6. Generate an answer grounded in the retrieved context.

## Project structure

- app.py: CLI entry point for indexing and question answering.
- config.py: global paths and defaults.
- src/arxiv_loader.py: downloads papers from arXiv by ID.
- src/extract_pdf.py: extracts text from PDFs using PyMuPDF or pypdf.
- src/chunk_documents.py: splits long text into overlapping chunks.
- src/embedding_model.py: creates embeddings for chunks and queries.
- src/create_vector_db.py: stores embeddings and performs similarity search.
- src/retriever.py: retrieves top-k relevant chunks.
- src/llm.py: generates a grounded answer from retrieved chunks.
- src/rag_pipeline.py: orchestrates the full process.
- dataset/: folder for downloaded arXiv papers.
- vector_store/: generated metadata and index files.
- prompts/: sample questions and arXiv IDs.
- tests/: regression tests for the pipeline.

## Data flow

1. The user provides one or more arXiv paper IDs.
2. The project downloads the corresponding PDFs into the dataset folder.
3. Text is extracted from each PDF.
4. The text is split into chunks.
5. Embeddings are generated for each chunk.
6. A similarity search retrieves the passages most relevant to the query.
7. The LLM-style answer generator returns a grounded response.

## Current status

The project currently supports downloading real arXiv papers, indexing them, and answering questions through the baseline RAG workflow. The next step is to extend this baseline with adaptive embedding selection for Review 2.

## How to run

Install dependencies:

```bash
pip install -r requirements.txt
```

Download and index papers:

```bash
python app.py --dataset-dir dataset --arxiv-ids 1706.03762 2401.12345 2203.02155
```

Ask a question:

```bash
python app.py --query "What is this paper about?"
```
