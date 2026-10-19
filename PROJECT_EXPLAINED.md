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

## Section V: Mid-Term System Progress and Evaluation Report

### 1. DESIGN OF THE DISK-BACKED WORKSPACE INGESTION AND PATH INTERFACE

The mid-term implementation replaces an ephemeral upload pathway with an explicit, disk-backed workspace contract. A submitted PDF is first serialized to `data/runtime_uploads/temp_processing.pdf`; this serialization establishes a stable file path before any parsing or indexing operation is attempted. The ingestion routine, `process_custom_pdf_path`, accepts that path as its authoritative input, verifies that the file exists, opens it with PyMuPDF, rejects corrupt or zero-page documents, and closes the document in a `finally` block. This sequence makes the upload boundary observable, repeatable, and independent of the lifetime of the browser-side file object.

The corresponding vector artifacts are isolated under `vector_store/runtime_uploads`, with extracted visual material stored below its `figures` child directory. The application therefore maintains two explicit namespaces: the serialized source workspace and the derived retrieval workspace. After processing, `save_index_from_documents` constructs the multi-model index from section-scoped records, while Streamlit session state retains the active vector-base path. Query dispatch subsequently resolves the active workspace through `resolve_index_path`; an upload query cannot silently fall back to the central corpus because the active mode and the current upload vector base are checked before retrieval. This path interface is a material correctness property: it prevents cross-document contamination and permits provenance links to resolve against the same artifact root that produced the answer.

Each indexed record carries a stable identifier, source path, page number, section type, title reference, image count, and source type. The resulting contract supports page-level citation, section-aware retrieval, isolated rebuilds, and deterministic test fixtures. The Playwright upload tests confirm the observable consequence of this design: a PDF is reported as indexed from disk, a custom index is selected, and the rendered answer exposes `Custom Uploaded Document` together with `Page 1` provenance. The architecture is consequently suitable for incremental workspace replacement without requiring process-global mutation of the central repository.

### 2. THE ADAPTIVE ROUTING ROUTINE MATRIX

The routing layer applies a lightweight query classifier before vector search. It tokenizes and normalizes the query, recognizes explicit mathematical or algorithmic operators, and assigns one of three semantic classes. The route is deterministic and is represented by the following matrix.

| Query regime | Operational signal | Selected encoder | Retrieval rationale |
|---|---|---|---|
| Technical or mathematical | Equations, operators, algorithms, gradients, derivatives, matrices, formulas, code, or embedding terminology | **BGE** (`BAAI/bge-small-en-v1.5`) | BGE is used for technical semantics in which compositional meaning, notation, and methodological relationships are more important than exact lexical overlap. |
| Conceptual or general | Explanatory interrogatives and architectural questions, including `what`, `why`, `how`, `define`, `overview`, and `purpose` | **E5** (`intfloat/e5-small-v2`) | E5 is used for natural-language conceptual alignment; its query/passage prefix convention is preserved by the embedding adapter. |
| Short keyword or entity | Short queries of at most four tokens without conceptual markers, typically requesting a named object, figure, or specific entity | **MiniLM** (`sentence-transformers/all-MiniLM-L6-v2`) | MiniLM provides a compact semantic representation for concise entity and keyword lookups with lower representation overhead. |

The mapping is implemented as `technical_mathematical→bge`, `conceptual_general→e5`, and `short_keyword_specific_entity→minilm`. BGE-only, E5-only, and MiniLM-only configurations remain useful as ablations because they hold the index family constant while removing routing diversity. The adaptive system does not claim that one encoder dominates every query distribution; rather, it treats encoder selection as a query-conditioned decision and evaluates the resulting policy against each single-encoder control.

### 3. STATEFUL SECTION SEGMENTATION AND CAPTION-AWARE VISUAL INDEXING

Segmentation is stateful across PDF pages. The parser recognizes academic headings, including `ABSTRACT`, `INTRODUCTION`, `BACKGROUND`, `METHODOLOGY`, `RELATED WORK`, `EVALUATION`, `EXPERIMENTAL RESULTS`, `CONCLUSION`, `FUTURE WORK`, and `REFERENCES`. When a page does not begin with a new heading, the last section state is inherited. Text is normalized into section records rather than being flattened into a single document string. This state transition is important for multi-page methods and results sections, where a heading can occur on one page and the explanatory content on the next.

The upload record writer emits identifiers of the form `paper_id_page_<page>_section_<index>`. It preserves the source path, page number, section type, title reference, and image count. Direct section queries can therefore bypass approximate vector similarity and select the corresponding section map entries from the process-resident metadata cache. This improves determinism for questions such as “explain the methodology” and prevents a semantically similar but structurally incorrect passage from outranking the explicitly requested section.

Visual indexing is caption-aware rather than based solely on arbitrary page images. The extractor detects `Figure n` and `Fig. n` caption labels, normalizes their identifiers and captions, locates the caption span rectangle, and saves a page capture through the caption boundary. Each manifest entry records the caption, image path, paper identifier, and page number. Visual retrieval applies dual verification: the query must match a manifest-level figure identity and the resulting image path must exist. The recorded evaluation proof confirms that `Show Figure 30` resolves to a concrete `_fig_30.png` artifact and that the browser loads the image. For custom uploads, the same principle is applied to page snapshots and a manifest under the active runtime workspace.

### 4. RE-RANKING PIPELINE AND DIVERGENT HIGH-DENSITY CONTENT SYNTHESIS LIMITS

Retrieval is a two-stage process. First, the selected encoder produces a query vector after grammatical stop-token trimming. FAISS or a section-filtered in-memory search returns up to 20 candidates (`candidate_k=20`). Second, when the local cross-encoder is available, each query-candidate pair receives a reranking score; candidates are sorted by that score and truncated to the requested result count. The default retrieval contract returns at most five passages unless a caller specifies a smaller `top_k`. This candidate expansion followed by cross-encoder ordering separates broad semantic recall from fine-grained relevance discrimination.

The reranker is optional and local-only by default. If its model is unavailable, the pipeline retains vector-search order rather than failing the complete question-answering workflow. Embedding models, FAISS indexes, the reranker, the global chunk map, and the section map use process-resident caching. Consequently, subsequent queries avoid repeated model construction, repeated index deserialization, and per-query metadata file reads.

The synthesis layer imposes a hard ceiling of 12 technical points. This is a content-control limit, not an instruction to summarize away high-density evidence. The generator is required to keep the executive response distinct from the technical breakdown, preserve source citations, avoid the internal token `UNSECTIONED`, and express the retrieved evidence as analytical prose. The ceiling controls divergent content growth while allowing the response to retain multiple methodological, quantitative, and operational distinctions. The performance regression test independently verifies both the textual point count and the rendered list-element count, so the limit is enforced at the user-visible interface rather than only in the prompt.

### 5. AUTOMATED INTEGRATION VALIDATION VIA PLAYWRIGHT AND IN-MEMORY CACHE RESULTS

The validation strategy combines browser-level workflow tests with unit-level metric computation. Playwright starts or reuses the Streamlit application, exercises repository and custom-upload paths, enters research questions, waits for rendered answer landmarks, and inspects the resulting DOM. The tests verify that the application exposes Executive Insights, Technical Breakdown, source citations, page provenance, visible Precision/Recall/F1-Score fields, and rendered figure assets. The upload regression additionally demonstrates that the answer is generated from the isolated custom workspace rather than the central repository.

The runtime cache is validated indirectly through the active process behavior and directly through the architecture of the retrieval module. `st.cache_resource` is used under Streamlit, with `lru_cache` as the non-Streamlit fallback. Cached resources include embedding registries, embedding models, FAISS indexes, rerankers, and global metadata maps. The first query pays initialization cost; later queries reuse the resident resources. The available proof artifact records a perfect top-3 retrieval event for the verified figure query (`accuracy=1.0`, `precision=1.0`, `recall=1.0`, and `f1=1.0`) and confirms that Precision, Recall, and F1-Score are visible in the application.

The following tables report the current comparison status. `NR` means that the repository does not contain a controlled measurement for that cell; it is retained explicitly to prevent an implementation proof from being misrepresented as a completed baseline experiment. Accuracy is the hit-at-3 definition implemented by `retrieval_metrics`, while Precision, Recall, and F1-Score are computed over unique top-3 retrieved identifiers. Latency is measured from the Playwright click to the appearance of Technical Breakdown.

| Configuration | Accuracy | Precision | Recall | F1-Score | Latency SLA |
|---|---:|---:|---:|---:|---:|
| BGE-only baseline | NR | NR | NR | NR | NR |
| E5-only baseline | NR | NR | NR | NR | NR |
| MiniLM-only baseline | NR | NR | NR | NR | NR |
| **Adaptive RAG** | **1.000** *(verified figure proof)* | **1.000** *(verified figure proof)* | **1.000** *(verified figure proof)* | **1.000** *(verified figure proof)* | **Pass gate: < 1.5 s** |

The second table separates executable gates from observations. The 1.5-second threshold is enforced by `test_final_latency.py`; a separate latency test enforces a less stringent 3.0-second ceiling, and the performance-limit test enforces a 2.5-second ceiling for a dense technical query while also checking the 12-point maximum. These are acceptance thresholds, not aggregate medians or percentile estimates.

| Validation dimension | Adaptive RAG observation or gate | BGE-only | E5-only | MiniLM-only |
|---|---|---|---|---|
| Retrieval metric visibility | Precision, Recall, and F1-Score rendered in the UI | Not measured | Not measured | Not measured |
| Verified retrieval event | Top-3 figure proof: 1.000 for accuracy, precision, recall, and F1-Score | Not measured | Not measured | Not measured |
| Query latency | Must complete in less than 1.5 s in the strict test; auxiliary gates are 2.5 s and 3.0 s | Not measured | Not measured | Not measured |
| Integration correctness | Upload isolation, page provenance, figure loading, and distinct answer sections pass in recorded tests | Not measured | Not measured | Not measured |
| Synthesis bound | Technical output contains no more than 12 points | Not applicable as a routing ablation | Not applicable as a routing ablation | Not applicable as a routing ablation |

Accordingly, the mid-term conclusion is one of architectural and integration readiness, not final comparative superiority. The adaptive design has a reproducible routing policy, isolated disk-backed ingestion, section and visual provenance, bounded reranking and synthesis, and executable browser validation. A publication-ready claim that Adaptive RAG outperforms every single-encoder baseline still requires the same frozen query set, relevance judgments, hardware configuration, warm-cache state, repeated latency samples, and independent runs for all four configurations.
