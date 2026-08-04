from __future__ import annotations

import argparse
import sys
from pathlib import Path

from config import DATASET_DIR, VECTOR_STORE_DIR
from src.arxiv_loader import download_arxiv_papers
from src.rag_pipeline import answer_query, build_and_save_index


def main() -> None:
    if hasattr(sys.stdout, "reconfigure"):
        sys.stdout.reconfigure(encoding="utf-8")
    if hasattr(sys.stderr, "reconfigure"):
        sys.stderr.reconfigure(encoding="utf-8")

    parser = argparse.ArgumentParser(description="Run the baseline research-paper RAG pipeline")
    parser.add_argument("--dataset-dir", default=str(DATASET_DIR), help="Directory with research papers")
    parser.add_argument("--vector-store-dir", default=str(VECTOR_STORE_DIR), help="Directory for the index metadata")
    parser.add_argument("--query", default=None, help="Ask a question directly from the CLI")
    parser.add_argument("--streamlit", action="store_true", help="Launch the Streamlit demo interface")
    parser.add_argument("--arxiv-ids", nargs="*", default=None, help="One or more arXiv IDs to download into the dataset directory")
    args = parser.parse_args()

    if args.arxiv_ids:
        ids = []
        for item in args.arxiv_ids:
            ids.extend([value.strip() for value in item.split(",") if value.strip()])
        downloaded = download_arxiv_papers(ids, args.dataset_dir)
        print(f"Downloaded {len(downloaded)} arXiv paper(s) into {args.dataset_dir}")

    if args.query:
        build_and_save_index(args.dataset_dir, args.vector_store_dir)
        print(answer_query(args.query, args.vector_store_dir))
        return

    if args.streamlit:
        try:
            import streamlit as st
        except ImportError as exc:  # pragma: no cover - optional dependency
            raise SystemExit("Streamlit is not installed. Install requirements.txt to use the UI.") from exc

        st.set_page_config(page_title="Research Paper RAG", page_icon="📚")
        st.title("Research Paper QA with RAG")
        st.write("This baseline version uses one embedding model and a simple retrieval pipeline.")

        st.write("Ask questions about the papers already present in the dataset directory.")
        question = st.text_input("Ask a question about the papers")
        if st.button("Answer") and question:
            with st.spinner("Building the index and retrieving context..."):
                build_and_save_index(args.dataset_dir, args.vector_store_dir)
                answer = answer_query(question, args.vector_store_dir)
            st.write(answer)
        return

    print("Index built successfully. Use --query to ask a question or --streamlit for the UI.")


if __name__ == "__main__":
    main()
