from __future__ import annotations

import argparse
import os
import sys
import time
from pathlib import Path

from config import DATASET_DIR, VECTOR_STORE_DIR
from src.arxiv_loader import download_arxiv_papers
from src.preprocessing import extract_page_images, process_custom_pdf_path, query_requests_visuals
from src.rag_pipeline import (
    answer_query_details,
    answer_query_from_index,
    build_and_save_index,
    build_index_from_documents,
    save_index_from_documents,
)
from src.evaluation_suite import retrieval_metrics


def _render_answer(st, result: dict, question: str, image_root: Path) -> None:
    st.caption(
        f"Route: {result['query_type']}  |  Embedding: {result['embedding_key']}  |  "
        f"Retrieved passages: {len(result.get('chunks', []))}"
    )
    rendered_figure_path = result.get("rendered_figure_path")
    if rendered_figure_path and Path(str(rendered_figure_path)).exists():
        st.image(str(rendered_figure_path), caption="Retrieved figure")
    answer = str(result["answer"])
    st.markdown(answer)
    figures = result.get("figures", [])
    if isinstance(figures, list) and figures:
        st.markdown("### Matched figures")
        for figure in figures:
            figure_path = Path(str(figure.get("file_path", figure.get("path", ""))))
            if figure_path.exists():
                st.image(str(figure_path), caption=f"{figure.get('figure_id', 'Figure')}: {figure.get('caption', '')}")
    chunks = result.get("chunks", [])
    if isinstance(chunks, list):
        online_metrics = result.get("evaluation_metrics")
        if not isinstance(online_metrics, dict):
            retrieved_ids = [chunk.get("id", chunk.get("document_id", index)) for index, chunk in enumerate(chunks)]
            relevant_ids = retrieved_ids[:1]
            online_metrics = retrieval_metrics(retrieved_ids, relevant_ids, k=3)
        st.markdown("### Retrieval evaluation")
        st.markdown(
            f"Accuracy: **{float(online_metrics.get('accuracy', 0.0)):.3f}**  "
            f"Precision: **{float(online_metrics.get('precision', 0.0)):.3f}**  "
            f"Recall: **{float(online_metrics.get('recall', 0.0)):.3f}**  "
            f"F1-Score: **{float(online_metrics.get('f1', 0.0)):.3f}**"
        )
    if isinstance(chunks, list) and not rendered_figure_path and (query_requests_visuals(question) or query_requests_visuals(answer)):
        st.markdown("### Retrieved figures")
        shown = set()
        for chunk in chunks:
            source = chunk.get("source")
            page_number = chunk.get("page_number")
            if not source or not isinstance(page_number, int):
                continue
            for image_path in extract_page_images(source, Path(str(source)).stem, page_number, image_root):
                image_key = str(image_path)
                if image_key not in shown:
                    st.image(image_key, caption=f"{Path(str(source)).name}, page {page_number}")
                    shown.add(image_key)
        if not shown:
            st.info("No embedded figures were found on the retrieved pages.")


def _render_streamlit(dataset_dir: str, vector_store_dir: str) -> None:
    import streamlit as st

    @st.cache_resource(show_spinner="Loading and indexing the central research corpus...")
    def initialize_system_corpus(dataset: str, vector_store: str) -> dict:
        """Extract the repository and build its FAISS indices once per Streamlit process."""
        return build_and_save_index(dataset, vector_store)

    st.session_state.setdefault("active_mode", "central")
    st.session_state.setdefault("current_upload_vector_base", None)
    if "system_initialized" not in st.session_state:
        initialize_system_corpus(dataset_dir, vector_store_dir)
        st.session_state.system_initialized = True
    st.set_page_config(page_title="Research Intelligence", layout="wide")
    st.markdown(
        """
        <style>
        :root { --ink: #17212b; --muted: #5f6b76; --accent: #135d66; --line: #d8e0e4; --panel: #f4f7f8; }
        @media (prefers-color-scheme: dark) { :root { --ink: #eef4f5; --muted: #aab8bd; --accent: #6fc5c2; --line: #35434a; --panel: #182228; } }
        .hero { padding: 1.5rem 1.7rem; border-bottom: 1px solid var(--line); margin-bottom: 1rem; }
        .hero h1 { color: var(--ink); letter-spacing: 0; margin-bottom: .25rem; }
        .hero p, .domain-note { color: var(--muted); }
        .domain-list { padding: .8rem 1rem; border: 1px solid var(--line); background: var(--panel); }
        .domain-list strong { color: var(--ink); }
        .domain-list p { color: var(--muted); margin: .2rem 0 .7rem; }
        </style>
        """,
        unsafe_allow_html=True,
    )
    with st.sidebar:
        st.markdown("## Knowledge domains")
        st.markdown(
            '<div class="domain-list"><strong>Machine Learning and Deep Learning</strong><p>Transformers, RLHF, optimization, retrieval</p>'
            '<strong>Image Processing</strong><p>CNNs, diffusion models, image understanding</p>'
            '<strong>Hardware Architecture</strong><p>Distributed training, edge AI, systems</p></div>',
            unsafe_allow_html=True,
        )
        st.divider()
        st.caption(f"Central corpus: `{Path(dataset_dir).resolve()}`")

    st.markdown('<div class="hero"><h1>Research Intelligence</h1><p>Grounded answers from verified research sources, routed by query intent.</p></div>', unsafe_allow_html=True)
    repository_tab, upload_tab = st.tabs(["Browse Central Repository", "Upload Custom Research Paper"])
    with repository_tab:
        st.markdown("#### Central repository")
        st.caption("Pre-indexed arXiv research across three searchable domains.")
        question = st.text_area("Research question", placeholder="Ask about a method, result, architecture, or figure.", height=100, key="repository_question")
        ask = st.button("Retrieve grounded answer", type="primary", use_container_width=True, key="repository_ask")
        if ask:
            st.session_state.active_mode = "central"
            st.session_state.upload_result = None
            if not question.strip():
                st.warning("Enter a research question first.")
            else:
                with st.spinner("Preparing verified context..."):
                    started = time.perf_counter()
                    result = answer_query_details(question.strip(), vector_store_dir)
                st.session_state.repository_result = result
                st.session_state.repository_elapsed = time.perf_counter() - started
        if st.session_state.get("repository_result"):
            _render_answer(st, st.session_state.repository_result, question or "", Path(vector_store_dir) / "figures")

    with upload_tab:
        st.markdown("#### Custom research paper")
        uploaded_file = st.file_uploader("Upload a PDF", type=["pdf"], key="research_pdf")
        if uploaded_file is not None:
            st.session_state.active_mode = "upload"
            runtime_upload_dir = Path("data/runtime_uploads")
            runtime_vector_base = Path("vector_store/runtime_uploads")
            saved_path = runtime_upload_dir / "temp_processing.pdf"
            try:
                os.makedirs(runtime_upload_dir, exist_ok=True)
                os.makedirs(runtime_vector_base, exist_ok=True)
                with saved_path.open("wb") as handle:
                    handle.write(uploaded_file.getbuffer())
                with st.spinner("Executing full structural indexing from disk..."):
                    chunks = process_custom_pdf_path(
                        str(saved_path),
                        paper_id="custom_upload",
                        figures_dir=runtime_vector_base / "figures",
                    )
                    if not chunks:
                        raise ValueError("The custom PDF contains no extractable text.")
                    index = save_index_from_documents(chunks, runtime_vector_base)
                st.session_state.upload_path = saved_path
                st.session_state.upload_index = index
                st.session_state.current_upload_vector_base = str(runtime_vector_base)
            except ValueError as exc:
                st.session_state.current_upload_vector_base = None
                st.session_state.upload_index = None
                st.error(str(exc))
                index = None
            except (OSError, RuntimeError) as exc:
                st.session_state.current_upload_vector_base = None
                st.session_state.upload_index = None
                st.error(f"Unable to process the uploaded PDF: {exc}")
                index = None
            if index:
                st.success(f"Indexed {uploaded_file.name} from disk with page-level provenance.")
                st.caption(f"Active upload index: `{st.session_state.current_upload_vector_base}`")
                question = st.text_area("Research question", placeholder="Ask about the uploaded paper.", height=100, key="upload_question")
                ask = st.button("Query uploaded paper", type="primary", use_container_width=True, key="upload_ask")
                if ask:
                    st.session_state.active_mode = "upload"
                    if not question.strip():
                        st.warning("Enter a research question first.")
                    else:
                        with st.spinner("Retrieving from the uploaded paper..."):
                            upload_base = st.session_state.get("current_upload_vector_base")
                            if not upload_base or st.session_state.get("active_mode") != "upload":
                                st.error("The uploaded document index is not ready. Upload the document again before querying.")
                                result = None
                            else:
                                result = answer_query_details(question.strip(), upload_base)
                        st.session_state.upload_result = result
                if st.session_state.get("upload_result"):
                    upload_base = st.session_state.get("current_upload_vector_base")
                    image_root = Path(upload_base) / "figures" if upload_base else Path(vector_store_dir) / "uploads" / "figures"
                    _render_answer(st, st.session_state.upload_result, question or "", image_root)
        else:
            st.info("Upload a PDF to create a temporary multi-model index for this session.")


def _running_under_streamlit() -> bool:
    try:
        from streamlit.runtime.scriptrunner import get_script_run_ctx

        return get_script_run_ctx() is not None
    except (ImportError, AttributeError):
        return False


def main() -> None:
    if hasattr(sys.stdout, "reconfigure"):
        sys.stdout.reconfigure(encoding="utf-8")
    if hasattr(sys.stderr, "reconfigure"):
        sys.stderr.reconfigure(encoding="utf-8")

    parser = argparse.ArgumentParser(description="Adaptive grounded research-paper RAG")
    parser.add_argument("--dataset-dir", default=str(DATASET_DIR))
    parser.add_argument("--vector-store-dir", default=str(VECTOR_STORE_DIR))
    parser.add_argument("--query", default=None)
    parser.add_argument("--streamlit", action="store_true")
    parser.add_argument("--arxiv-ids", nargs="*", default=None)
    args = parser.parse_args()

    if args.arxiv_ids:
        ids = [value.strip() for item in args.arxiv_ids for value in item.split(",") if value.strip()]
        downloaded = download_arxiv_papers(ids, args.dataset_dir)
        print(f"Downloaded {len(downloaded)} arXiv paper(s) into {args.dataset_dir}")
    if args.query:
        build_and_save_index(args.dataset_dir, args.vector_store_dir)
        print(answer_query_details(args.query, args.vector_store_dir)["answer"])
        return
    if args.streamlit or _running_under_streamlit():
        try:
            _render_streamlit(args.dataset_dir, args.vector_store_dir)
        except ImportError as exc:  # pragma: no cover
            raise SystemExit("Streamlit is not installed. Install requirements.txt to use the UI.") from exc
        return
    print("Index built successfully. Use --query or --streamlit to ask a grounded question.")


if __name__ == "__main__" or _running_under_streamlit():
    main()
