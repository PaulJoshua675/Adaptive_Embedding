#!/usr/bin/env bash
set -euo pipefail

ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
REPO_ID="${1:-${HF_SPACE_ID:-}}"

if [[ -z "$REPO_ID" ]]; then
  echo "Usage: $0 <huggingface-username/space-name>" >&2
  exit 2
fi
if ! command -v hf >/dev/null 2>&1; then
  echo "Install the Hugging Face CLI first: python -m pip install 'huggingface_hub==0.30.2'" >&2
  exit 2
fi

cd "$ROOT"
hf upload "$REPO_ID" . . \
  --repo-type space \
  --commit-message "Deploy Research Intelligence Streamlit Space" \
  --include "README.md" \
  --include "requirements-hf.txt" \
  --include "app.py" \
  --include "config.py" \
  --include "main_api.py" \
  --include "download_arxiv_corpus.py" \
  --include "prompts/**" \
  --include "dataset/**" \
  --include "src/**" \
  --include "vector_store/metadata.pkl" \
  --include "vector_store/faiss_bge/**" \
  --include "vector_store/faiss_e5/**" \
  --include "vector_store/faiss_minilm/**" \
  --include "vector_store/figures/**"

hf upload "$REPO_ID" "$ROOT/requirements-hf.txt" requirements.txt \
  --repo-type space \
  --commit-message "Use pinned CPU dependencies for Streamlit Space"
