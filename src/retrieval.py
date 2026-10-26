from __future__ import annotations

import os
from pathlib import Path

try:
    import streamlit as st
except ImportError:  # pragma: no cover - optional runtime dependency
    st = None


def _synchronize_huggingface_environment() -> None:
    """Load HF credentials before importing any component that can contact the Hub."""
    project_env = Path(__file__).resolve().parents[1] / ".env"
    if project_env.exists():
        try:
            for raw_line in project_env.read_text(encoding="utf-8").splitlines():
                line = raw_line.strip()
                if not line or line.startswith("#") or "=" not in line:
                    continue
                key, value = line.split("=", 1)
                key = key.strip()
                value = value.strip().strip('"').strip("'")
                if key in {"HF_TOKEN", "HUGGINGFACEHUB_API_TOKEN"} and value and key not in os.environ:
                    os.environ[key] = value
        except OSError:
            pass
    token = os.environ.get("HF_TOKEN") or os.environ.get("HUGGINGFACEHUB_API_TOKEN")
    os.environ["HF_TOKEN"] = token or ""
    if token:
        os.environ["HUGGINGFACEHUB_API_TOKEN"] = token
    else:
        os.environ.setdefault("HF_HUB_OFFLINE", "1")
        os.environ.setdefault("TRANSFORMERS_OFFLINE", "1")


_synchronize_huggingface_environment()

import difflib
import json
import re
import pickle
import time
from urllib.parse import quote
try:
    from dotenv import load_dotenv
except ImportError:  # pragma: no cover - optional runtime dependency
    def load_dotenv(*args, **kwargs) -> bool:
        return False

from functools import lru_cache
from typing import Dict, Iterable, List, Tuple

from config import DEFAULT_TOP_K, MIN_RERANKER_SCORE, MIN_RETRIEVAL_SIMILARITY, RERANKER_MODEL, VECTOR_STORE_DIR
from .embeddings import EmbeddingRegistry
from .reranker import PassageReranker

load_dotenv(Path(__file__).resolve().parents[1] / ".env")
if os.getenv("HF_TOKEN"):
    os.environ["HF_TOKEN"] = os.getenv("HF_TOKEN", "")


from .preprocessing import section_query_name, trim_query_tokens
from .extract_pdf import clean_extracted_text
from .vector_db import MultiVectorStore
from .profiler import profile_stage

try:
    from sentence_transformers import CrossEncoder  # type: ignore
except ImportError:  # pragma: no cover
    CrossEncoder = None

try:
    from sentence_transformers import SentenceTransformer  # type: ignore
except ImportError:  # pragma: no cover
    SentenceTransformer = None

try:
    from transformers import AutoModelForSequenceClassification, AutoTokenizer  # type: ignore
except ImportError:  # pragma: no cover
    AutoModelForSequenceClassification = None
    AutoTokenizer = None

_SECTION_QUERY = re.compile(r"\b(?:the\s+)?(abstract|introduction|background|related work|methods?|methodology|experiments?|results?|discussion|conclusions?|references)\b", re.I)
_GENERIC_QUERY_TERMS = {
    "a", "an", "and", "are", "about", "can", "could", "describe", "did", "do",
    "does", "explain", "for", "from", "how", "in", "is", "me", "of", "on",
    "please", "tell", "that", "the", "this", "to", "what", "which", "with",
    "would",
}
_ARXIV_ID = re.compile(r"(?<!\d)(\d{4}\.\d{4,5})(?:v\d+)?(?!\d)", re.I)


def _paper_id_from_document(document: Dict[str, object]) -> str | None:
    source = str(document.get("source", document.get("document_id", "")))
    match = _ARXIV_ID.search(source)
    return match.group(1) if match else None


def _resource_cache(function):
    if st is None:
        return lru_cache(maxsize=8)(function)
    return st.cache_resource(show_spinner=False)(function)


@_resource_cache
def get_global_chunk_map() -> Dict[int, Dict[str, object]]:
    """Return the process-resident chunk payload map used after vector search."""
    return {}


@_resource_cache
def get_global_section_map() -> Dict[str, List[int]]:
    """Return the process-resident section-to-chunk ID map."""
    return {}


def populate_global_chunk_map(chunks: Iterable[Dict[str, object]]) -> None:
    """Populate RAM metadata for the active index without per-query file reads."""
    chunk_map = get_global_chunk_map()
    section_map = get_global_section_map()
    chunk_map.clear()
    section_map.clear()
    for chunk_id, chunk in enumerate(chunks):
        payload = dict(chunk)
        payload["chunk_id"] = chunk_id
        payload["page"] = int(chunk.get("page_number", 1))
        payload["section"] = str(chunk.get("section_type", "ABSTRACT"))
        chunk_map[chunk_id] = payload
        if isinstance(chunk, dict):
            chunk["chunk_id"] = chunk_id
        section_map.setdefault(str(payload["section"]).upper(), []).append(chunk_id)


def _ram_payload(document: Dict[str, object], fallback_id: int) -> Dict[str, object]:
    chunk_map = get_global_chunk_map()
    chunk_id = document.get("chunk_id", fallback_id)
    try:
        return dict(chunk_map.get(int(chunk_id), document))
    except (TypeError, ValueError):
        return dict(document)


def _direct_section_retrieval(query: str, top_k: int) -> List[Dict[str, object]]:
    section = section_query_name(query)
    if section is None:
        return []
    chunk_map = get_global_chunk_map()
    section_map = get_global_section_map()
    chunk_ids = section_map.get(section, [])[: min(max(int(top_k), 3), 5)]
    return [_ram_payload(chunk_map[chunk_id], chunk_id) for chunk_id in chunk_ids if chunk_id in chunk_map]


@_resource_cache
def get_cached_embedding_model(model_name: str):
    """Keep embedding weights resident in system memory for the process lifetime."""
    if SentenceTransformer is None:
        raise RuntimeError("sentence-transformers is not installed")
    local_only = os.getenv("EMBEDDING_LOCAL_ONLY", "1").lower() not in {"0", "false", "no"}
    model_kwargs = {"local_files_only": True} if local_only else {}
    return SentenceTransformer(model_name, model_kwargs=model_kwargs)


def load_cached_embedding_model(model_name: str):
    """Backward-compatible alias for the Streamlit-cached embedding loader."""
    return get_cached_embedding_model(model_name)


@_resource_cache
def load_cached_reranker(model_name: str = RERANKER_MODEL):
    """Keep a transformer reranker tokenizer and model resident for the process lifetime."""
    if AutoTokenizer is None or AutoModelForSequenceClassification is None:
        raise RuntimeError("transformers is not installed")
    local_only = os.getenv("RERANKER_LOCAL_ONLY", "1").lower() not in {"0", "false", "no"}
    kwargs = {"local_files_only": True} if local_only else {}
    tokenizer = AutoTokenizer.from_pretrained(model_name, **kwargs)
    model = AutoModelForSequenceClassification.from_pretrained(model_name, **kwargs)
    model.eval()
    return tokenizer, model


@_resource_cache
def load_cached_faiss_index(index_path: str, modified_ns: int, size: int):
    """Pin one FAISS binary in process memory instead of rereading it per query."""
    del modified_ns, size
    import faiss  # type: ignore

    faiss.omp_set_num_threads(4)
    return faiss.read_index(index_path)


@_resource_cache
def load_cached_embedding_registry(models: Tuple[Tuple[str, str], ...]) -> EmbeddingRegistry:
    return EmbeddingRegistry(dict(models))


def resolve_index_path(base_index_path: str | Path | None = None) -> Path:
    """Resolve the requested index, honoring the active Streamlit upload workspace."""
    if base_index_path is not None:
        return Path(base_index_path)
    try:
        import streamlit as st

        if st.session_state.get("active_mode") == "upload":
            runtime_path = st.session_state.get("current_upload_vector_base")
            if runtime_path:
                return Path(str(runtime_path))
    except (ImportError, RuntimeError, AttributeError):
        pass
    return Path("vector_store")


def _index_signature(base_path: Path, keys: Tuple[str, ...]) -> Tuple[Tuple[str, int, int], ...]:
    signature: List[Tuple[str, int, int]] = []
    for key in keys:
        for filename in ("index.faiss", "documents.json", "embeddings.json"):
            path = base_path / f"faiss_{key}" / filename
            try:
                stat = path.stat()
                signature.append((str(path), stat.st_mtime_ns, stat.st_size))
            except OSError:
                signature.append((str(path), 0, 0))
    return tuple(signature)


@_resource_cache
def _load_cached_store(base_path: str, keys: Tuple[str, ...], signature: Tuple[Tuple[str, int, int], ...]) -> MultiVectorStore:
    index_signatures = {path: (modified_ns, size) for path, modified_ns, size in signature}

    def load_index(index_path: str):
        modified_ns, size = index_signatures.get(index_path, (0, 0))
        return load_cached_faiss_index(index_path, modified_ns, size)

    return MultiVectorStore.load(base_path, list(keys), index_loader=load_index)


def load_cached_vector_store(base_path: str | Path, keys: List[str]) -> MultiVectorStore:
    root = Path(base_path)
    normalized_keys = tuple(keys)
    return _load_cached_store(str(root), normalized_keys, _index_signature(root, normalized_keys))


def populate_chunk_map_from_store(store: MultiVectorStore, key: str) -> None:
    """Hydrate the RAM payload map from a loaded vector store."""
    selected_store = store.stores.get(key)
    if selected_store is not None:
        populate_global_chunk_map(selected_store.documents)


def section_from_query(query: str) -> str | None:
    match = _SECTION_QUERY.search(query)
    if not match:
        return None
    value = match.group(1).upper()
    return value[:-1] if value.endswith("S") and value not in {"RESULTS", "METHODS"} else value


def _tokens(value: str) -> set[str]:
    return set(re.findall(r"[a-z0-9]+", value.lower()))


def _lexical_overlap(query: str, passage: str) -> float:
    query_tokens = _tokens(_ARXIV_ID.sub(" ", query)) - _GENERIC_QUERY_TERMS
    if not query_tokens:
        return 0.0
    return len(query_tokens & _tokens(passage)) / len(query_tokens)


def _visual_query(query: str) -> bool:
    return any(
        word in query.lower()
        for word in ("figure", "diagram", "chart", "picture", "graph", "fig", "image", "plot", "architecture", "visual", "table")
    )


def _manifest_entries(manifest: object) -> Iterable[Dict[str, object]]:
    if not isinstance(manifest, dict):
        return []
    entries: List[Dict[str, object]] = []
    for paper_id, figures in manifest.items():
        if isinstance(figures, dict) and "caption" in figures:
            entries.append({"figure_id": str(paper_id), **figures})
            continue
        if isinstance(figures, dict):
            for figure_id, entry in figures.items():
                if isinstance(entry, dict):
                    entries.append({"paper_id": str(paper_id), "figure_id": str(figure_id), **entry})
    return entries


def _query_text(query: str) -> str:
    query = re.sub(r"\b(?:figure|fig\.?)\s*\d+\b", " ", query, flags=re.I)
    return re.sub(r"\b(?:show|find|display|the|a|an|figure|fig\.?|diagram|chart|picture|plot)\b", " ", query, flags=re.I)


def _manifest_path(manifest_path: str | Path) -> Path:
    path = Path(manifest_path)
    if path.exists() or manifest_path != Path("vector_store/figures/figure_manifest.json"):
        return path
    configured = os.getenv("RAG_FIGURES_DIR")
    if configured:
        return Path(configured) / "figure_manifest.json"
    if os.getenv("RAG_STATIC_DIR"):
        return Path(os.environ["RAG_STATIC_DIR"]) / "figure_manifest.json"
    return path


def _figure_identifier(query: str) -> str | None:
    match = re.search(r"\b(?:figure|fig\.?)\s*(\d+)\b", query, re.I)
    return f"Figure {match.group(1)}" if match else None


def _figure_number(value: object) -> str | None:
    value = str(value)
    match = re.search(r"\b(?:figure|fig\.?)\s*(\d+)\b", value, re.I)
    if match:
        return match.group(1)
    return value.strip() if value.strip().isdigit() else None


def _normalise_figure_identifier(value: str) -> str:
    match = re.search(r"(?:figure|fig\.?)\s*(\d+)", value, re.I)
    return f"figure {match.group(1)}" if match else value.lower()


def _surrounding_context(entry: Dict[str, object], manifest_path: Path) -> str:
    """Load page-local text used to verify a manifest candidate."""
    paper_id = str(entry.get("paper_id", "")).strip()
    page_number = entry.get("page_number")
    candidates: List[Path] = []
    extracted_dir = manifest_path.parent.parent.parent / "extracted_text"
    if paper_id and isinstance(page_number, int):
        candidates.extend(sorted(extracted_dir.glob(f"{paper_id}_page_{page_number}*.txt")))
    source = entry.get("source")
    if source:
        source_path = Path(str(source))
        if source_path.exists():
            try:
                import fitz  # type: ignore

                with fitz.open(source_path) as document:
                    if isinstance(page_number, int) and 1 <= page_number <= len(document):
                        return document[page_number - 1].get_text("text") or ""
            except Exception:
                pass
    parts: List[str] = []
    for candidate in candidates:
        try:
            parts.append(candidate.read_text(encoding="utf-8"))
        except OSError:
            continue
    return "\n".join(parts)


def _semantic_alignment(query: str, caption: str, context: str) -> float:
    """Deterministic lightweight semantic proxy for the manifest search channel."""
    query_tokens = _tokens(_query_text(query))
    description_tokens = _tokens(f"{caption} {context}")
    if not query_tokens:
        return 1.0
    token_score = len(query_tokens & description_tokens) / len(query_tokens)
    sequence_score = difflib.SequenceMatcher(None, _query_text(query).lower().strip(), caption.lower()).ratio()
    return max(token_score, sequence_score)


def _verify_figure_alignment(query: str, entry: Dict[str, object], context: str, threshold: float) -> Tuple[bool, float]:
    caption = str(entry.get("caption", ""))
    target_identifier = _figure_identifier(query)
    context_labels = re.findall(r"\b(?:figure|fig\.?)\s*\d+\b", context, re.I)
    if target_identifier and context_labels and not any(
        _normalise_figure_identifier(label) == _normalise_figure_identifier(target_identifier) for label in context_labels
    ):
        return False, 0.0
    score = _semantic_alignment(query, caption, context)
    if target_identifier and not context:
        return True, max(score, 1.0)
    return score >= threshold, score


def _figure_referenced_by_passages(entry: Dict[str, object], passages: Iterable[Dict[str, object]]) -> bool:
    figure_label = str(entry.get("figure_id", "")) or str(entry.get("caption", ""))
    figure_id = _normalise_figure_identifier(figure_label)
    figure_match = re.fullmatch(r"figure\s+(\d+)", figure_id)
    if not figure_match:
        return False
    reference = re.compile(rf"\b(?:figure|fig\.?)\s*{figure_match.group(1)}\b", re.I)
    paper_id = str(entry.get("paper_id", "")).strip().lower()
    source = str(entry.get("source", "")).strip()
    for passage in passages:
        text = str(passage.get("text", ""))
        if not reference.search(text):
            continue
        passage_source = str(passage.get("source", "")).strip()
        if paper_id and (
            not passage_source
            or (
                Path(passage_source).stem.lower() != paper_id
                and (not source or Path(passage_source).resolve() != Path(source).resolve())
            )
        ):
            continue
        if not paper_id and source and (
            not passage_source or Path(passage_source).resolve() != Path(source).resolve()
        ):
            continue
        return True
    return False


def find_matching_figures(
    query: str,
    manifest_path: str | Path = "vector_store/figures/figure_manifest.json",
    threshold: float = 0.35,
    passages: Iterable[Dict[str, object]] | None = None,
) -> List[Dict[str, object]]:
    if not _visual_query(query):
        return []
    path = _manifest_path(manifest_path)
    if not path.exists():
        return []
    manifest = json.loads(path.read_text(encoding="utf-8"))
    query_text = _query_text(query).strip()
    query_tokens = _tokens(query_text)
    matches: List[Dict[str, object]] = []
    requested_number = _figure_number(query)
    for entry in _manifest_entries(manifest):
        figure_number = _figure_number(entry.get("number", entry.get("figure_number", entry.get("figure_id", ""))))
        identifier_match = requested_number is not None and figure_number == requested_number
        if requested_number is not None and not identifier_match:
            continue
        if passages is not None and not identifier_match and not _figure_referenced_by_passages(entry, passages):
            continue
        caption = str(entry.get("caption", ""))
        figure_id = str(entry.get("figure_id", ""))
        caption_tokens = _tokens(caption)
        overlap = len(query_tokens & caption_tokens) / max(len(query_tokens), 1)
        similarity = difflib.SequenceMatcher(None, query_text.lower(), caption.lower()).ratio()
        context = _surrounding_context(entry, path)
        score = 1.0 if identifier_match else max(overlap, similarity)
        verified, alignment = _verify_figure_alignment(query, entry, context, threshold)
        if identifier_match:
            descriptive_tokens = query_tokens - _GENERIC_QUERY_TERMS
            figure_tokens = _tokens(f"{caption} {context}")
            content_alignment = (
                len(descriptive_tokens & figure_tokens) / len(descriptive_tokens)
                if descriptive_tokens
                else 1.0
            )
            alignment = max(alignment, content_alignment)
            verified = content_alignment >= threshold
        if not verified:
            print("Verifying figure alignment... Figure discarded due to low context confidence.")
            continue
        if score >= threshold:
            file_path = str(entry.get("file_path", entry.get("path", "")))
            matches.append({"score": score, "alignment_score": alignment, "path": file_path, "file_path": file_path, **entry})
    unique: Dict[str, Dict[str, object]] = {}
    for match in matches:
        unique[str(match.get("file_path", match.get("path", "")))] = match
    return sorted(
        unique.values(),
        key=lambda item: (float(item["score"]), float(item.get("alignment_score", 0.0))),
        reverse=True,
    )


def _resolved_figure_payload(figure: Dict[str, object], manifest_path: str | Path) -> Dict[str, object]:
    file_path = Path(str(figure.get("file_path", figure.get("path", ""))))
    if not file_path.is_absolute():
        candidates = (
            file_path,
            Path(manifest_path).parent / file_path,
            VECTOR_STORE_DIR / file_path,
        )
        file_path = next((candidate for candidate in candidates if candidate.exists()), candidates[1])
    resolved_path = file_path.resolve()
    number = figure.get("number") or figure.get("figure_number") or _figure_number(figure.get("figure_id", ""))
    resolved = {
        **figure,
        "number": number or "",
        "file_path": str(resolved_path),
        "path": str(resolved_path),
        "resolved_path": str(resolved_path),
    }
    try:
        image_relative_path = resolved_path.relative_to((VECTOR_STORE_DIR / "figures").resolve())
    except ValueError:
        return resolved
    resolved["image_url"] = f"/images/{quote(image_relative_path.as_posix(), safe='/')}"
    return resolved


def inject_visual_payload(
    payload: Dict[str, object],
    query: str,
    manifest_path: str | Path = "vector_store/figures/figure_manifest.json",
) -> Dict[str, object]:
    """Attach only figures explicitly referenced by retrieved passages."""
    passages = payload.get("chunks", [])
    requested_number = _figure_number(query)
    matches = find_matching_figures(
        query,
        manifest_path,
        passages=passages if requested_number is None and isinstance(passages, list) else None,
    )
    linked_figures: List[Dict[str, object]] = []
    if isinstance(passages, list):
        for passage in passages:
            if not isinstance(passage, dict):
                continue
            figures = passage.get("figures", [])
            if isinstance(figures, list):
                linked_figures.extend(
                    figure
                    for figure in figures
                    if isinstance(figure, dict)
                    and _figure_referenced_by_passages(figure, [passage])
                    and (
                        _visual_query(query)
                        or _semantic_alignment(query, str(figure.get("caption", "")), "") >= 0.35
                    )
                )
    unique_figures: Dict[str, Dict[str, object]] = {}
    if requested_number is None:
        candidate_figures = [*linked_figures, *matches]
    else:
        candidate_figures = matches[:1] or [
            figure
            for figure in linked_figures
            if _figure_number(
                figure.get(
                    "number",
                    figure.get("figure_number", figure.get("figure_id", figure.get("caption", ""))),
                )
            )
            == requested_number
        ]
    for figure in candidate_figures:
        file_path = str(figure.get("file_path", figure.get("path", "")))
        if file_path:
            resolved_figure = _resolved_figure_payload(figure, manifest_path)
            unique_figures[str(resolved_figure["resolved_path"])] = resolved_figure
    result = dict(payload)
    result["figures"] = list(unique_figures.values())
    if result["figures"]:
        result["matched_figure"] = result["figures"][0]
    else:
        result.pop("matched_figure", None)
    result.pop("rendered_figure_path", None)
    if result["figures"]:
        result["rendered_figure_path"] = result["figures"][0]["file_path"]
    return result


def retrieve_context(
    query: str,
    selected_model: str,
    base_index_path: str | Path | None = None,
    top_k: int = 5,
    candidate_k: int = 20,
    section_type: str | None = None,
) -> List[Dict[str, object]]:
    """Retrieve context from the explicitly selected persisted index.

    A missing or incomplete runtime index is treated as an empty result so a
    Streamlit rerun cannot accidentally fall through to the central corpus.
    """
    base_path = resolve_index_path(base_index_path)
    metadata_path = base_path / "metadata.pkl"
    try:
        if not metadata_path.exists():
            print(f"Retrieval index is not ready: {metadata_path}")
            return []
        with metadata_path.open("rb") as handle:
            metadata = pickle.load(handle)
        models = dict(metadata.get("embedding_models") or {})
        keys = list(metadata.get("embedding_keys") or models.keys())
        if selected_model not in keys or selected_model not in models:
            print(f"Embedding index is unavailable for model {selected_model!r} at {base_path}")
            return []
        stores = load_cached_vector_store(base_path, [selected_model])
        populate_chunk_map_from_store(stores, selected_model)
        direct_matches = _direct_section_retrieval(query, top_k)
        if direct_matches:
            return direct_matches
        registry = load_cached_embedding_registry(((selected_model, str(models[selected_model])),))
        return AdaptiveRetriever(stores, registry).retrieve(
            query,
            embedding_key=selected_model,
            top_k=top_k,
            candidate_k=candidate_k,
            section_type=section_type,
        )
    except (OSError, ValueError, KeyError, pickle.PickleError) as exc:
        print(f"Unable to retrieve from index {base_path}: {exc}")
        return []


class AdaptiveRetriever:
    def __init__(self, stores: MultiVectorStore, registry: EmbeddingRegistry, reranker_model: str = RERANKER_MODEL):
        self.stores = stores
        self.registry = registry
        self.reranker_model = reranker_model
        self.reranker = None
        self.last_timings_ms: Dict[str, float] = {
            "faiss_index_lookup_ms": 0.0,
            "cross_encoder_reranking_ms": 0.0,
        }

    def retrieve(
        self,
        query: str,
        embedding_key: str,
        top_k: int = DEFAULT_TOP_K,
        candidate_k: int = 20,
        section_type: str | None = None,
        query_vector: List[float] | None = None,
        include_low_relevance: bool = False,
    ) -> List[Dict[str, object]]:
        self.last_timings_ms = {
            "faiss_index_lookup_ms": 0.0,
            "cross_encoder_reranking_ms": 0.0,
        }
        result_limit = min(max(int(top_k), 3), 5)
        candidate_limit = min(max(int(candidate_k), 3), 5)
        encoder = self.registry[embedding_key]
        store = self.stores.stores[embedding_key]
        with profile_stage("Query Vectorization"):
            if query_vector is None:
                query_vector = encoder.encode_query(trim_query_tokens(query))
        reference_query = bool(re.search(r"\b(?:references?|bibliography|citations?)\b", query, re.I))
        requested_paper = _ARXIV_ID.search(query)
        target_paper_id = requested_paper.group(1) if requested_paper else None
        minimum_similarity = -1.0 if include_low_relevance else (
            max(0.12, MIN_RETRIEVAL_SIMILARITY * 0.5)
            if target_paper_id is not None
            else MIN_RETRIEVAL_SIMILARITY
        )
        search_limit = candidate_limit
        if section_type:
            wanted = section_type.upper()
            candidates = [
                (sum(a * b for a, b in zip(query_vector, embedding)), document)
                for embedding, document in zip(store.embeddings, store.documents)
                if str(document.get("section_type", "")).upper() == wanted
                and (target_paper_id is None or _paper_id_from_document(document) == target_paper_id)
            ]
            candidates.sort(key=lambda item: item[0], reverse=True)
            candidates = candidates[:search_limit]
        elif target_paper_id is not None:
            candidates = [
                (sum(a * b for a, b in zip(query_vector, embedding)), document)
                for embedding, document in zip(store.embeddings, store.documents)
                if _paper_id_from_document(document) == target_paper_id
            ]
            candidates.sort(key=lambda item: item[0], reverse=True)
            candidates = candidates[:search_limit]
        else:
            lookup_started = time.perf_counter()
            with profile_stage("FAISS Index Search"):
                candidates = store.search(query_vector, search_limit)
            self.last_timings_ms["faiss_index_lookup_ms"] = (time.perf_counter() - lookup_started) * 1000
        ranked = []
        for raw_score, document in candidates:
            if not reference_query and str(document.get("section_type", "")).upper() in {
                "REFERENCE",
                "REFERENCES",
                "BIBLIOGRAPHY",
            }:
                continue
            text = clean_extracted_text(str(document.get("text", "")))
            if not text:
                continue
            similarity = float(raw_score)
            if similarity >= minimum_similarity:
                ranked.append({"score": similarity, **document, "text": text})
        ranked = [_ram_payload(item, index) | {"score": item.get("score", 0.0)} for index, item in enumerate(ranked)]
        use_cross_encoder = not include_low_relevance and os.getenv("RAG_ENABLE_CROSS_ENCODER_RERANKER", "0").lower() in {
            "1",
            "true",
            "yes",
        }
        if use_cross_encoder and ranked:
            rerank_started = time.perf_counter()
            if self.reranker is None:
                self.reranker = PassageReranker(self.reranker_model)
            with profile_stage("Cross-Encoder Reranking"):
                reranked = self.reranker.rerank(
                    query,
                    ranked,
                    top_k=result_limit,
                    candidate_k=candidate_limit,
                )
            self.last_timings_ms["cross_encoder_reranking_ms"] = (time.perf_counter() - rerank_started) * 1000
        else:
            reranked = ranked[:result_limit]
        return [
            item
            for item in reranked
            if float(item.get("score", 0.0)) >= minimum_similarity
            and (
                "rerank_score" not in item
                or float(item["rerank_score"]) >= MIN_RERANKER_SCORE
            )
            and (
                "rerank_score" in item
                or _lexical_overlap(query, str(item.get("text", "")))
                >= (0.15 if target_paper_id is not None else MIN_RETRIEVAL_SIMILARITY)
                or include_low_relevance
            )
        ]


@_resource_cache
def _load_cached_cross_encoder(model_name: str):
    if CrossEncoder is None:
        raise RuntimeError("sentence-transformers is not installed")
    local_only = os.getenv("RERANKER_LOCAL_ONLY", "1").lower() not in {"0", "false", "no"}
    return CrossEncoder(model_name, local_files_only=local_only)
