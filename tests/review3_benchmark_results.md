# Review 3 Retrieval Comparison

- Generated: `2026-10-18T15:00:03.327031+00:00`
- Corpus: 245 sources, 18,352 chunks
- Encoder backend setting: `feature-hash fallback`
- Judgment: one manually mapped relevant paper per query; exploratory smoke benchmark, not a full relevance set
- Protocol: 3 measured runs after one warm-up; top-3 retrieval; reranking disabled.

| Configuration | Hit@K | Precision | Recall | F1 | Median query latency (ms) |
|---|---:|---:|---:|---:|---:|
| BGE baseline | 0.111 | 0.037 | 0.111 | 0.056 | 1.06 |
| E5 baseline | 0.444 | 0.241 | 0.444 | 0.296 | 1.07 |
| MiniLM baseline | 0.333 | 0.148 | 0.333 | 0.204 | 1.01 |
| Adaptive RAG | 0.444 | 0.241 | 0.444 | 0.296 | 1.11 |

## Query Routes

| Query | Gold source | Adaptive route | Adaptive hit@K |
|---|---|---|---:|
| Explain scaled dot-product attention and multi-head attention. | `1706.03762` | `e5` (0.63) | 1 |
| How does the Transformer use positional encodings? | `1706.03762` | `e5` (0.53) | 0 |
| Attention Is All You Need | `1706.03762` | `minilm` (0.44) | 0 |
| What is the supervised fine-tuning stage in InstructGPT? | `2203.02155` | `e5` (0.53) | 0 |
| Describe the reward model objective used for instruction following. | `2203.02155` | `e5` (0.48) | 0 |
| Training language models to follow instructions with human feedback | `2203.02155` | `e5` (0.56) | 1 |
| How was Llama 2 fine-tuned with human preference data? | `2307.09288` | `e5` (0.57) | 1 |
| Compare the context windows reported for Llama 2 models. | `2307.09288` | `e5` (0.56) | 1 |
| Llama 2 | `2307.09288` | `minilm` (0.48) | 0 |

Latency covers query encoding and FAISS search only; it excludes loading, classification, reranking, and answer generation.

## UI Latency Optimization & Hosting Architecture

| Deployment path | Cold page render | Query path | Hosting |
|---|---|---|---|
| Baseline Streamlit startup | Synchronously built and saved the full corpus before rendering controls | Model/index work and answer generation were not separated in the UI | Local Streamlit process |
| Cached and lazy-loaded deployment | Renders controls before indexing or loading query models | Caches metadata and FAISS handles; loads only the selected index; reports intent, FAISS, reranking, and generation timings | Hugging Face Streamlit Space, Python 3.10, CPU-pinned dependencies, Streamlit on port 8501 |

The expected improvement is limited to first-page rendering and avoiding repeated disk/model setup; query latency remains dependent on cold model initialization, Space hardware, and generation configuration. The 1.5-second target is not represented as a measured Space result.

**Live evaluator URL:** Not deployed yet. After creating and validating a Space, replace this line with `https://<username>-<space-name>.hf.space`.
