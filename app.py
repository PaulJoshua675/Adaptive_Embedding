from __future__ import annotations

import argparse
import hashlib
import json
import logging
import os
import pickle
import re
import sys
import time
from functools import lru_cache
from pathlib import Path

from config import DATASET_DIR, DOMAIN_EXAMPLE_QUESTIONS, VECTOR_STORE_DIR


ARXIV_DOMAINS = {
    "cs.CV": "Computer Vision & Pattern Recognition",
    "cs.CL": "Computation and Language / NLP",
    "cs.LG": "Machine Learning",
    "cs.AI": "Artificial Intelligence",
    "cs.IR": "Information Retrieval / Neural Computing",
}
KNOWLEDGE_DOMAINS = {
    "Machine Learning and Deep Learning": "Transformers, RLHF, optimization, retrieval",
    "Image Processing": "CNNs, diffusion models, image understanding",
    "Hardware Architecture": "Distributed training, edge AI, systems",
}
ARXIV_DOMAIN_CODES = {
    "cs.CV": ("cs.CV",),
    "cs.CL": ("cs.CL",),
    "cs.LG": ("cs.LG",),
    "cs.AI": ("cs.AI",),
    "cs.IR": ("cs.IR", "cs.NE"),
}
LOGGER = logging.getLogger(__name__)


def _paper_id(value: str | Path) -> str:
    paper_id = Path(str(value)).name
    if paper_id.lower().endswith(".pdf"):
        paper_id = paper_id[:-4]
    return re.sub(r"v\d+$", "", paper_id)


def _category_tags(record: dict) -> set[str]:
    raw_categories = record.get("categories", record.get("category", []))
    if isinstance(raw_categories, str):
        raw_categories = [raw_categories]
    if not isinstance(raw_categories, (list, tuple, set)):
        return set()
    return {category.strip() for category in raw_categories if isinstance(category, str) and category.strip()}


@lru_cache(maxsize=4)
def _indexed_paper_categories(metadata_path: str, modified_ns: int) -> dict[str, set[str]]:
    try:
        with Path(metadata_path).open("rb") as handle:
            metadata = pickle.load(handle)
    except (OSError, pickle.PickleError, EOFError, AttributeError, TypeError, ValueError) as exc:
        LOGGER.warning("Could not read indexed corpus metadata at %s: %s", metadata_path, exc)
        return {}

    categories_by_id: dict[str, set[str]] = {}
    if not isinstance(metadata, dict):
        LOGGER.warning("Indexed corpus metadata at %s is not a mapping", metadata_path)
        return categories_by_id
    chunks = metadata.get("chunks", [])
    if not isinstance(chunks, list):
        LOGGER.warning("Indexed corpus metadata at %s has no valid chunks list", metadata_path)
        return categories_by_id
    for chunk in chunks:
        if not isinstance(chunk, dict):
            continue
        paper_id = _paper_id(chunk.get("source", ""))
        tags = _category_tags(chunk)
        if paper_id and tags:
            categories_by_id.setdefault(paper_id, set()).update(tags)
    return categories_by_id


def _domain_paper_counts(dataset_dir: str | Path, vector_store_dir: str | Path) -> tuple[dict[str, int], int]:
    dataset_path = Path(dataset_dir)
    pdf_paths = {path for path in dataset_path.rglob("*.pdf") if path.is_file()}
    custom_upload_dir = Path(vector_store_dir).parent / "data" / "runtime_uploads"
    if custom_upload_dir.is_dir():
        pdf_paths.update(path for path in custom_upload_dir.rglob("*.pdf") if path.is_file())
    present_ids = {_paper_id(path) for path in pdf_paths}
    manifest_path = dataset_path / "arxiv_metadata.json"
    dataset_ids = {_paper_id(path) for path in dataset_path.rglob("*.pdf") if path.is_file()}
    existing_manifest_ids: set[str] = set()
    needs_reclassification = False
    if manifest_path.is_file():
        try:
            manifest_for_check = json.loads(manifest_path.read_text(encoding="utf-8"))
        except (OSError, UnicodeError, json.JSONDecodeError) as exc:
            LOGGER.warning("Could not read arXiv metadata manifest at %s: %s", manifest_path, exc)
        else:
            if isinstance(manifest_for_check, dict):
                existing_manifest_ids = {_paper_id(key) for key in manifest_for_check}
                needs_reclassification = any(
                    _paper_id(key) in dataset_ids
                    and isinstance(record, dict)
                    and not _category_tags(record)
                    and record.get("classification_method") != "local-title-abstract-keywords-v5"
                    for key, record in manifest_for_check.items()
                )
    if (dataset_ids - existing_manifest_ids) or needs_reclassification:
        try:
            from src.dataset_classifier import classify_dataset

            classify_dataset(dataset_path, manifest_path)
        except (OSError, RuntimeError, ValueError) as exc:
            LOGGER.warning("Could not classify dataset papers for the sidebar: %s", exc)
    category_by_id: dict[str, set[str]] = {}
    if manifest_path.is_file():
        try:
            loaded = json.loads(manifest_path.read_text(encoding="utf-8"))
        except (OSError, UnicodeError, json.JSONDecodeError) as exc:
            LOGGER.warning("Could not read arXiv metadata manifest at %s: %s", manifest_path, exc)
        else:
            if isinstance(loaded, dict):
                for key, record in loaded.items():
                    if isinstance(record, dict):
                        paper_id = _paper_id(key)
                        tags = _category_tags(record)
                        if paper_id and tags:
                            category_by_id.setdefault(paper_id, set()).update(tags)
            else:
                LOGGER.warning("arXiv metadata manifest at %s is not a mapping", manifest_path)

    missing_ids = present_ids - category_by_id.keys()
    metadata_path = Path(vector_store_dir) / "metadata.pkl"
    if metadata_path.is_file() and (missing_ids or (not present_ids and not category_by_id)):
        try:
            modified_ns = metadata_path.stat().st_mtime_ns
        except OSError as exc:
            LOGGER.warning("Could not inspect indexed corpus metadata at %s: %s", metadata_path, exc)
        else:
            indexed_categories = _indexed_paper_categories(str(metadata_path.resolve()), modified_ns)
            for paper_id in missing_ids or indexed_categories.keys():
                if paper_id not in category_by_id and paper_id in indexed_categories:
                    category_by_id[paper_id] = indexed_categories[paper_id]

    paper_ids = present_ids or set(category_by_id)
    counts = {
        domain: sum(
            bool(category_by_id.get(paper_id, set()).intersection(category_codes))
            for paper_id in paper_ids
        )
        for domain, category_codes in ARXIV_DOMAIN_CODES.items()
    }
    classified_ids = {
        paper_id
        for paper_id in paper_ids
        if any(
            category_by_id.get(paper_id, set()).intersection(category_codes)
            for category_codes in ARXIV_DOMAIN_CODES.values()
        )
    }
    return counts, len(paper_ids - classified_ids)


def _render_answer(st, result: dict, question: str, image_root: Path, render_answer: bool = True) -> None:
    from src.evaluation_suite import retrieval_metrics

    del question, image_root
    confidence = float(result.get("route_confidence", 0.0))
    model_name = str(result.get("embedding_model", result.get("embedding_key", "unknown")))
    st.caption(
        f"ML route: {result['query_type']} ({confidence:.1%}) | "
        f"Embedding backend: {result['embedding_key']} ({model_name}) | "
        f"Retrieved passages: {len(result.get('chunks', []))}"
    )
    answer = str(result["answer"])
    if render_answer:
        st.markdown(answer)
    figures = result.get("figures", [])
    if isinstance(figures, list) and figures:
        st.markdown("### Matched figures")
        for figure in figures:
            figure_path = Path(str(figure.get("file_path", figure.get("path", ""))))
            if figure_path.exists():
                paper_id = str(figure.get("paper_id", ""))
                page_number = figure.get("page_number", "page unavailable")
                st.image(
                    str(figure_path),
                    caption=(
                        f"{figure.get('figure_id', 'Figure')} | Page {page_number} | "
                        f"{paper_id}: {figure.get('caption', '')}"
                    ),
                )
                if paper_id and "." in paper_id and paper_id.replace(".", "", 1).isdigit():
                    st.markdown(f"[Open verified arXiv source](https://arxiv.org/abs/{paper_id})")
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
def _render_performance_metrics(st, result: dict, total_elapsed: float | None = None) -> None:
    timings = result.get("performance_timings_ms", {})
    if total_elapsed is not None:
        st.caption(f"Total query round trip: {total_elapsed * 1000:.1f} ms")
    timing_columns = st.columns(4)
    timing_labels = (
        ("Intent Classification Time (ms)", "intent_classification_ms"),
        ("FAISS Index Lookup Time (ms)", "faiss_index_lookup_ms"),
        ("Cross-Encoder Re-ranking Time (ms)", "cross_encoder_reranking_ms"),
        ("LLM / Multi-Lingual Answer Generation Time (ms)", "answer_generation_ms"),
    )
    for column, (label, key) in zip(timing_columns, timing_labels):
        column.metric(label, f"{float(timings.get(key, 0.0)):.1f}")


def _render_streamlit(dataset_dir: str, vector_store_dir: str) -> None:
    import streamlit as st

    st.set_page_config(page_title="Research Intelligence", layout="wide")

    @st.cache_resource(show_spinner=False)
    def index_uploaded_pdf(
        _pdf_content: bytes,
        content_digest: str,
        original_name: str,
        saved_path: str,
        vector_base: str,
    ) -> dict:
        """Index each uploaded file once without retaining vector objects in session state."""
        from src.preprocessing import process_custom_pdf_path
        from src.rag_pipeline import save_index_from_documents

        del content_digest
        pdf_path = Path(saved_path)
        base_path = Path(vector_base)
        base_path.mkdir(parents=True, exist_ok=True)
        pdf_path.parent.mkdir(parents=True, exist_ok=True)
        pdf_path.write_bytes(_pdf_content)
        chunks = process_custom_pdf_path(
            str(pdf_path),
            paper_id="custom_upload",
            figures_dir=base_path / "figures",
        )
        if not chunks:
            raise ValueError("The custom PDF contains no extractable text.")
        save_index_from_documents(chunks, base_path)
        return {"name": original_name, "chunk_count": len(chunks)}

    st.session_state.setdefault("active_mode", "central")
    st.session_state.setdefault("current_upload_vector_base", None)
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
        response_language = st.selectbox(
            "Response language",
            ["English", "Tamil", "Hindi", "Spanish", "French", "Detect from query"],
            key="response_language",
        )
        st.markdown("## Knowledge domains")
        _, unclassified_count = _domain_paper_counts(dataset_dir, vector_store_dir)
        for label, description in KNOWLEDGE_DOMAINS.items():
            st.markdown(f"**{label}**  \n{description}")
        st.caption(f"Unclassified / custom uploads: {unclassified_count} papers")
        st.divider()
        st.caption(f"Central corpus: {Path(dataset_dir).resolve()}")

    st.markdown('<div class="hero"><h1>Research Intelligence</h1><p>Grounded answers from verified research sources, routed by query intent.</p></div>', unsafe_allow_html=True)
    repository_tab, upload_tab = st.tabs(["Browse Central Repository", "Upload Custom Research Paper"])
    with repository_tab:
        st.markdown("#### Central repository")
        st.caption("Select a domain, then click a curated research question from the local corpus to run it.")
        domain = st.selectbox(
            "Research domain",
            list(ARXIV_DOMAINS),
            format_func=lambda code: f"{code} — {ARXIV_DOMAINS[code]}",
            key="repository_domain",
        )
        selected_example = None
        for example_index, example in enumerate(DOMAIN_EXAMPLE_QUESTIONS[domain]):
            if st.button(
                example,
                key=f"repository_example_{domain}_{example_index}",
                use_container_width=True,
            ):
                st.session_state["repository_question"] = example
                selected_example = example
        question = st.text_area("Research question", placeholder="Ask about a method, result, architecture, or figure.", height=100, key="repository_question")
        fast_mode = st.checkbox(
            "Fast extractive mode (skip LLM generation)",
            key="repository_fast_mode",
        )
        ask = st.button("Retrieve grounded answer", type="primary", use_container_width=True, key="repository_ask") or selected_example is not None
        if ask:
            st.session_state.active_mode = "central"
            st.session_state.upload_result = None
            if not question.strip():
                st.warning("Enter a research question first.")
            else:
                streamed_answer = st.empty()
                with st.spinner("Preparing verified context..."):
                    started = time.perf_counter()
                    from src.rag_pipeline import answer_query_details, build_and_save_index

                    metadata_path = Path(vector_store_dir) / "metadata.pkl"
                    if not metadata_path.exists():
                        build_and_save_index(dataset_dir, vector_store_dir)
                    result = answer_query_details(
                        question.strip(),
                        vector_store_dir,
                        response_language=response_language,
                        stream_consumer=streamed_answer.write_stream,
                        fast_mode=fast_mode,
                    )
                streamed_answer.markdown(result["answer"])
                st.session_state.repository_result = result
                st.session_state.repository_elapsed = time.perf_counter() - started
        if st.session_state.get("repository_result"):
            _render_answer(
                st,
                st.session_state.repository_result,
                question or "",
                Path(vector_store_dir) / "figures",
                render_answer=not ask,
            )
            _render_performance_metrics(
                st,
                st.session_state.repository_result,
                st.session_state.repository_elapsed,
            )

    with upload_tab:
        st.markdown("#### Custom research paper")
        uploaded_file = st.file_uploader("Upload a PDF", type=["pdf"], key="research_pdf")
        if uploaded_file is not None:
            st.session_state.active_mode = "upload"
            runtime_upload_dir = Path("data/runtime_uploads")
            runtime_vector_base = Path("vector_store/runtime_uploads")
            pdf_content = uploaded_file.getvalue()
            content_digest = hashlib.sha256(pdf_content).hexdigest()
            saved_path = runtime_upload_dir / f"{content_digest}.pdf"
            runtime_vector_base = Path("vector_store/runtime_uploads") / content_digest
            try:
                with st.spinner("Executing full structural indexing from disk..."):
                    upload_info = index_uploaded_pdf(
                        pdf_content,
                        content_digest,
                        uploaded_file.name,
                        str(saved_path),
                        str(runtime_vector_base),
                    )
                st.session_state.upload_path = saved_path
                st.session_state.upload_index = upload_info
                st.session_state.current_upload_vector_base = str(runtime_vector_base)
            except ValueError as exc:
                st.session_state.current_upload_vector_base = None
                st.session_state.upload_index = None
                st.error(str(exc))
            except (OSError, RuntimeError) as exc:
                st.session_state.current_upload_vector_base = None
                st.session_state.upload_index = None
                st.error(f"Unable to process the uploaded PDF: {exc}")
            if st.session_state.get("upload_index"):
                st.success(f"Indexed {upload_info['name']} from disk with page-level provenance.")
                st.caption(f"Active upload index: `{st.session_state.current_upload_vector_base}`")
                question = st.text_area("Research question", placeholder="Ask about the uploaded paper.", height=100, key="upload_question")
                fast_mode = st.checkbox(
                    "Fast extractive mode (skip LLM generation)",
                    key="upload_fast_mode",
                )
                ask = st.button("Query uploaded paper", type="primary", use_container_width=True, key="upload_ask")
                if ask:
                    st.session_state.active_mode = "upload"
                    started = time.perf_counter()
                    if not question.strip():
                        st.warning("Enter a research question first.")
                    else:
                        from src.rag_pipeline import answer_query_details

                        streamed_answer = st.empty()
                        with st.spinner("Retrieving from the uploaded paper..."):
                            upload_base = st.session_state.get("current_upload_vector_base")
                            if not upload_base or st.session_state.get("active_mode") != "upload":
                                st.error("The uploaded document index is not ready. Upload the document again before querying.")
                                result = None
                            else:
                                result = answer_query_details(
                                    question.strip(),
                                    upload_base,
                                    response_language=response_language,
                                    stream_consumer=streamed_answer.write_stream,
                                    fast_mode=fast_mode,
                                )
                        if result is not None:
                            streamed_answer.markdown(result["answer"])
                        st.session_state.upload_result = result
                        st.session_state.upload_elapsed = time.perf_counter() - started
                if st.session_state.get("upload_result"):
                    upload_base = st.session_state.get("current_upload_vector_base")
                    image_root = Path(upload_base) / "figures" if upload_base else Path(vector_store_dir) / "uploads" / "figures"
                    _render_answer(
                        st,
                        st.session_state.upload_result,
                        question or "",
                        image_root,
                        render_answer=not ask,
                    )
                    _render_performance_metrics(
                        st,
                        st.session_state.upload_result,
                        st.session_state.get("upload_elapsed"),
                    )
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
    parser.add_argument("--response-language", default=None)
    parser.add_argument("--streamlit", action="store_true")
    parser.add_argument("--arxiv-ids", nargs="*", default=None)
    args = parser.parse_args()

    if args.arxiv_ids:
        from src.arxiv_loader import download_arxiv_papers

        ids = [value.strip() for item in args.arxiv_ids for value in item.split(",") if value.strip()]
        downloaded = download_arxiv_papers(ids, args.dataset_dir)
        print(f"Downloaded {len(downloaded)} arXiv paper(s) into {args.dataset_dir}")
    if args.query:
        from src.rag_pipeline import answer_query_details, build_and_save_index

        build_and_save_index(args.dataset_dir, args.vector_store_dir)
        print(answer_query_details(
            args.query,
            args.vector_store_dir,
            response_language=args.response_language,
        )["answer"])
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
