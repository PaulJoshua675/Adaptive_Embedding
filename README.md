---
title: Research Intelligence
colorFrom: teal
colorTo: green
sdk: docker
---

# Adaptive RAG for Research Paper Question Answering

This project implements an adaptive retrieval-augmented generation system for research papers. The current version:

- extracts text from PDF files when PyMuPDF is available,
- falls back to plain text files in the dataset directory when PDFs are not present,
- chunks documents into overlapping passages,
- creates dedicated FAISS indexes for BGE, E5, and MiniLM embeddings,
- classifies each query and routes it to the most suitable index,
- classifies query intent with a zero-shot model and a trained TF-IDF/logistic-regression fallback,
- optionally re-ranks up to 15 candidates with `cross-encoder/ms-marco-MiniLM-L-6-v2`,
- creates hierarchical document summaries for PDFs with 50 or more pages,
- supports English, Tamil, Hindi, Spanish, and French answer generation,
- answers questions with Groq's OpenAI-compatible API when `GROQ_API_KEY` is set, with local/extractive fallbacks and source attribution.

The corpus downloader targets arXiv papers across machine learning (`cs.LG`), computer vision (`cs.CV`), computational linguistics (`cs.CL`), AI (`cs.AI`), and information retrieval (`cs.IR`). Downloads are concurrent, validated, retryable, and safe to resume. It stores each paper's actual arXiv categories and title in `dataset/arxiv_metadata.json`; older datasets without that manifest are shown as unclassified until downloaded or refreshed with the downloader.

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
2. Download a 200-paper arXiv corpus into the dataset directory:
   `python download_arxiv_corpus.py --output-dir dataset --max-results 200 --workers 8`
   Or download selected papers with `python app.py --dataset-dir dataset --arxiv-ids 1706.03762 2401.12345`.
3. Build the index:
   `python app.py --dataset-dir dataset`
4. Ask a question:
   `python app.py --query "What does RAG use to answer questions?"`
5. Start the UI:
   `streamlit run app.py -- --streamlit`

Set `GROQ_API_KEY` in the process environment to enable hosted answer generation with Groq's `llama-3.1-8b-instant` model. Without a key, the application uses its configured local or extractive fallback.

## API deployment

Run the API locally with `uvicorn main_api:app --host 0.0.0.0 --port 8000`. The API exposes `GET /health`, `POST /ask`, and `POST /index`. The optional Docker container starts the API on port 8000 and Streamlit on port 7860:

```powershell
docker build -t research-rag-app .
docker run --rm -p 8000:8000 -p 7860:7860 research-rag-app
```

The `/ask` JSON body accepts `question`, optional `response_language` (English, Tamil, Hindi, Spanish, French, or `Detect from query`), and `top_k`. Model weights may be downloaded on first use; set the existing `EMBEDDING_LOCAL_ONLY` and `RERANKER_LOCAL_ONLY` options to control local-only loading.

### Host process scripts

On Windows, after creating `.venv` and installing `requirements.txt`, start the API detached with `powershell -ExecutionPolicy Bypass -File deploy/start_api.ps1`. Check readiness with `powershell -ExecutionPolicy Bypass -File deploy/healthcheck.ps1`; stop the managed process with `powershell -ExecutionPolicy Bypass -File deploy/stop_api.ps1`. The start script waits for `/health`, writes PID and logs under `data/`, and fails if the selected port is already occupied. Pass `-Port 8010` to the start and stop scripts to select a different port, or `-BaseUrl http://127.0.0.1:8010` to the health check.

On Linux, use `python -m venv .venv`, install `requirements.txt`, then run `bash deploy/start_api.sh`. Set `RAG_API_HOST`, `RAG_API_PORT`, `RAG_PYTHON`, or `RAG_API_STARTUP_TIMEOUT` to override defaults. Check readiness with `curl --fail http://127.0.0.1:8000/health` and stop it with `bash deploy/stop_api.sh`. These scripts are suitable for a single-host deployment; place a TLS-terminating reverse proxy and firewall in front of the API for public hosting. Keep one Uvicorn worker by default because each worker loads its own large vector stores and model state. Configure `EMBEDDING_USE_TRANSFORMERS`, `EMBEDDING_LOCAL_ONLY`, and `RERANKER_LOCAL_ONLY` explicitly for the target host.

### Hugging Face Spaces deployment

The Streamlit Space uses Python 3.10 and pinned CPU dependencies from `requirements-hf.txt`. Hugging Face installs the uploaded dependency file as `requirements.txt` and serves Streamlit on port 8501. The app renders before loading the corpus; persisted metadata and the selected FAISS index load on the first query. Intent routing and cross-encoder initialization are deferred until query submission. Set `GROQ_API_KEY` in the Space secrets to enable hosted answer generation; without it, the local or extractive answer fallbacks remain available.

Create a Streamlit Space at https://huggingface.co/new-space, or create it with the CLI. Authenticate locally, then run the deploy script from the repository root:

```powershell
hf auth login
hf repos create YOUR_HF_USERNAME/your-space-name --repo-type space --sdk streamlit
# Run the deploy script from Git Bash:
bash deploy/deploy_hf.sh YOUR_HF_USERNAME/your-space-name
```

The script uploads the dataset (including `arxiv_metadata.json` when present), prompts, central metadata, three FAISS stores, and figure assets explicitly; it omits runtime uploads and test indexes. The local dataset and index directories are git-ignored, so the script uses the Hugging Face CLI to upload these assets directly rather than relying on a git push. The Space URL will be `https://YOUR_HF_USERNAME-your-space-name.hf.space`. No live evaluator URL exists until a Space has been created and smoke-tested.

### Free Render deployment

The root `render.yaml` defines a free Docker Web Service named `adaptive-embedding-rag`. Streamlit listens on the configured `PORT` (`8501` on Render); the container retains the FastAPI process on internal port `8000`. Render checks Streamlit at `/_stcore/health`.

1. Push the repository, including `render.yaml`, `Dockerfile`, and the application source, to GitHub.
2. Sign in to Render, choose **New** > **Blueprint**, connect the repository, and select the branch to deploy. Render reads `render.yaml`; confirm the service plan is **Free** before creating it.
3. Wait for the image build and deployment. Open the generated `https://<service-name>.onrender.com` URL and verify that the Research Intelligence page loads. Free services can take time to wake after inactivity.
4. Before asking a retrieval question, ensure the image contains both the corpus and its matching vector-store files. The local `dataset/` and `vector_store/` directories are git-ignored and will not be included in a normal Git-based Render build. Include suitable assets in the build context or change the deployment process to provision them.

No additional secrets are needed for the configured offline embedding fallback. Render's free filesystem is ephemeral: uploaded PDFs and indexes generated at runtime are lost on restart or redeploy. Free instances also have limited memory and CPU, so large corpus indexing and model-backed inference may exceed available resources. Use a small prebuilt corpus/index for a demo; durable uploads and larger workloads need a host with persistent storage and more memory. The public Render route serves Streamlit; the co-hosted API remains on internal port `8000` and is not separately exposed by this Blueprint.

### Review 3 UI latency and hosting architecture

| Path | First page render | First query | Subsequent queries |
|---|---|---|---|
| Previous Streamlit startup | Built the complete corpus before displaying the page | Repeated metadata reads; model/index setup could be paid together | Process caches helped, but stage latency was not visible |
| Cached/lazy Space | Page and controls render without corpus indexing or model initialization | Loads cached metadata and the routed FAISS index; routing/reranking initialize only after submission | Reuses process-cached metadata, FAISS handles, embedding registry, and loaded models |

The UI reports intent classification, FAISS lookup, cross-encoder reranking, and answer-generation timings, plus total query time. The 1.5-second target is an acceptance goal, not a guarantee: first-query model initialization, CPU tier, corpus size, and answer-generation configuration affect latency. Evaluator URL: **not deployed yet**; add the generated `https://<username>-<space-name>.hf.space` URL here after verifying the Space.

### Review 3 retrieval benchmark

Run `python -m src.benchmark_comparison --runs 5 --top-k 3` to compare BGE-only, E5-only, MiniLM-only, and adaptive routing against the persisted stores. The command writes `tests/review3_benchmark_results.json` and `.md`. It uses paper-level relevant-source judgments for nine fixed queries and reports its limited judgment set and encoder backend; treat the output as an exploratory smoke benchmark, not a statistically powered comparison.

## Example questions

You can try the bundled prompts from [prompts/example_questions.txt](prompts/example_questions.txt) or ask questions such as:

- What is the purpose of retrieval-augmented generation in research paper QA?
- How do embedding models improve retrieval for scientific documents?
- Why is chunking important for long academic text?
- How does FAISS support fast nearest-neighbour search?

## Notes

The adaptive indexes use the configured transformer models when `EMBEDDING_USE_TRANSFORMERS=1`. By default, stable feature-hashed vectors are used so indexing works offline and starts quickly; this is a deterministic fallback, not a placeholder. Set `EMBEDDING_LOCAL_ONLY=1` to restrict transformer loading to models already cached locally. Set `RERANKER_LOCAL_ONLY=0` to enable loading the cross-encoder re-ranker when it is available.
