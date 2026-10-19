"""Infer arXiv subject categories from locally available paper text."""

from __future__ import annotations

import argparse
import json
import logging
import re
from pathlib import Path
from typing import Dict, List

from config import DATASET_DIR

LOGGER = logging.getLogger(__name__)

_CATEGORY_TERMS = {
    "cs.CV": (
        "computer vision", "image classification", "object detection",
        "image segmentation", "visual recognition", "video understanding",
        "image generation", "vision transformer", "image colorization",
        "image colourizer", "3d reconstruction", "visual question answering",
        "saliency detection", "fashion recognition", "object tracking",
        "video generator", "vision-language", "visual-language",
        "scene reconstruction", "occupancy network", "pathology foundation",
        "medical imaging", "lesion evidence", "video captioning", "image editing",
        "audiovisual", "super-resolution", "spatial reconstruction",
        "autonomous driving", "motion forecasting", "camera-based", "occupancy networks",
        "image semantic segmentation", "low-light image", "image segmentation",
        "skeleton-based", "segmentation mistakes", "multi-modal fusion",
        "modality fusion for image", "low-light image enhancement",
    ),
    "cs.CL": (
        "natural language processing", "language model", "large language model",
        "machine translation", "text generation", "speech recognition",
        "question answering", "tokenization", "dialogue system",
        "language understanding", "text classification", "text retrieval",
        "speech model", "automatic speech", "large language models",
        "multilingual", "instruction-following", "nmt", "llm", "vlm",
        "transformer", "sequence generation", "token", "n-gram", "captioning",
        "mandarin", "writing", "speech attribute", "audio-video",
        "language models", "byte-level language", "community question-answering",
        "question-answering", "machine translation", "cqa",
    ),
    "cs.LG": (
        "machine learning", "deep learning", "neural network", "representation learning",
        "self-supervised learning", "supervised learning", "unsupervised learning",
        "reinforcement learning", "optimization algorithm", "continual learning",
        "transfer learning", "evolution strategy", "evolutionary algorithm",
        "evolutionary computation", "neuroevolution", "gradient descent",
        "model training", "training a", "generalization", "clustering algorithm",
        "classification model", "regression model", "knn", "cma-es",
        "optimizer", "optimization", "learning", "learner", "co-evolution",
        "co-evolved", "representation", "backpropagation", "contrastive",
        "model merging", "knowledge distillation", "policy evaluation",
        "adaptive step size", "test-time optimization", "sampling", "tensor",
        "prediction model", "neural", "genetic algorithm", "active learning",
        "foundation model", "data compression", "causal integrated gradients",
        "reservoir computing", "time series foundation", "time series forecasting",
        "multivariate trajectories", "generative forecasting", "probabilistic weather",
        "forecasting model", "motion forecasting", "weather model",
        "weather prediction", "tropical cyclone forecasting",
    ),
    "cs.AI": (
        "artificial intelligence", "intelligent agent", "automated planning",
        "commonsense reasoning", "symbolic reasoning", "knowledge representation",
        "multi-agent system", "decision making",
        "agentic", "autonomous agent", "program search", "self-improving",
        "self-reproduction", "co-evolved communication", "evolutionary architecture",
        "agent", "multi-agent", "multi agent", "decision", "planning",
        "reasoning", "tool use", "agent memory", "agent safety", "robotics",
        "robotic", "gui interaction", "policy", "information gain",
        "environment-agnostic", "communication", "interactive system",
        "autonomous driving", "traffic agents", "weather forecasting",
        "ai forecast", "ai model", "ai-based",
    ),
    "cs.IR": (
        "information retrieval", "retrieval augmented generation", "document retrieval",
        "search engine", "search ranking", "information access",
        "recommender system", "recommendation system",
        "speech retrieval", "web search", "query processing", "event retrieval",
        "popularity bias", "information seeking", "reader data",
        "retrieval", "search", "ranking", "reference wording", "web scale",
        "recommendation", "query-efficient", "paper retrieval", "graph search",
        "event detection", "open-web", "sequential recommendation",
    ),
    "cs.NE": (
        "computational neuroscience", "neural coding", "spiking neural network",
        "brain imaging", "neuronal activity",
        "spiking neuron", "neural evolution", "neuroevolutionary",
        "brain connectivity", "fMRI connectivity", "neural population",
        "neuroevolution", "evolutionary", "spiking", "neuron", "brain",
        "cma-es", "genetic algorithm", "coupled oscillator", "reservoir computing",
    ),
}
_CATEGORY_CODE = re.compile(r"\bcs\.(?:CV|CL|LG|AI|IR|NE)\b", re.I)
_ABSTRACT_START = re.compile(r"\babstract\b\s*[:.]?", re.I)
_ABSTRACT_END = re.compile(
    r"\b(?:1\s*\.?\s*)?(?:introduction|keywords|index terms)\b", re.I
)
_CLASSIFICATION_METHOD = "local-title-abstract-keywords-v5"


def _extract_paper_text(pdf_path: Path) -> str:
    try:
        import fitz
    except ImportError:
        LOGGER.warning("PyMuPDF is unavailable; cannot classify %s from its PDF text", pdf_path)
        return ""

    try:
        with fitz.open(pdf_path) as document:
            return "\n".join(page.get_text("text") or "" for page in list(document)[:5])[:40000]
    except (OSError, RuntimeError, ValueError) as exc:
        LOGGER.warning("Could not read paper text from %s: %s", pdf_path, exc)
        return ""


def _paper_title(text: str, fallback: str) -> str:
    for line in text.splitlines()[:24]:
        candidate = " ".join(line.split()).strip(" .")
        if (
            15 <= len(candidate) <= 240
            and not _ABSTRACT_START.match(candidate)
            and not candidate.lower().startswith(("arxiv:", "copyright", "proceedings"))
        ):
            return candidate
    return fallback


def _classify_text(text: str) -> List[str]:
    explicit = {
        f"cs.{match.group(0).split('.', 1)[1].upper()}"
        for match in _CATEGORY_CODE.finditer(text)
    }
    lowered = " ".join(text.casefold().split())
    categories = set(explicit)
    for category, phrases in _CATEGORY_TERMS.items():
        if any(re.search(r"(?<!\w)" + re.escape(phrase) + r"(?!\w)", lowered) for phrase in phrases):
            categories.add(category)
    return sorted(categories)


def _load_metadata(path: Path) -> Dict[str, Dict[str, object]]:
    if not path.exists():
        return {}
    try:
        metadata = json.loads(path.read_text(encoding="utf-8"))
    except (OSError, UnicodeError, json.JSONDecodeError) as exc:
        raise ValueError(f"Could not read arXiv metadata manifest {path}: {exc}") from exc
    if not isinstance(metadata, dict):
        raise ValueError(f"arXiv metadata manifest {path} must contain a JSON object")
    return {
        str(paper_id): dict(record)
        for paper_id, record in metadata.items()
        if isinstance(record, dict)
    }


def classify_dataset(
    dataset_dir: str | Path = DATASET_DIR,
    metadata_path: str | Path | None = None,
) -> Dict[str, Dict[str, object]]:
    """Classify papers missing a manifest entry and persist the resulting metadata."""
    root = Path(dataset_dir)
    manifest_path = Path(metadata_path) if metadata_path else root / "arxiv_metadata.json"
    metadata = _load_metadata(manifest_path)
    changed = False

    for pdf_path in sorted(root.rglob("*.pdf")):
        if not pdf_path.is_file():
            continue
        old_record = metadata.get(pdf_path.stem, {})
        if old_record.get("categories") or old_record.get("classification_method") == _CLASSIFICATION_METHOD:
            continue
        text = _extract_paper_text(pdf_path)
        abstract_match = _ABSTRACT_START.search(text)
        if abstract_match:
            abstract = text[abstract_match.end():]
            end_match = _ABSTRACT_END.search(abstract)
            if end_match:
                abstract = abstract[:end_match.start()]
        else:
            abstract = text[:12000]
        title = str(old_record.get("title") or _paper_title(text, pdf_path.stem))
        categories = _classify_text(f"{title}\n{abstract}")
        metadata[pdf_path.stem] = {
            "title": title,
            "categories": categories,
            "classification_method": _CLASSIFICATION_METHOD,
        }
        changed = True

    if changed:
        manifest_path.parent.mkdir(parents=True, exist_ok=True)
        manifest_path.write_text(
            json.dumps(metadata, ensure_ascii=False, indent=2) + "\n",
            encoding="utf-8",
        )
    return metadata


def main() -> None:
    parser = argparse.ArgumentParser(description="Infer arXiv categories from local paper PDFs.")
    parser.add_argument("--dataset-dir", type=Path, default=DATASET_DIR)
    parser.add_argument("--metadata-path", type=Path, default=None)
    args = parser.parse_args()
    metadata = classify_dataset(args.dataset_dir, args.metadata_path)
    classified = sum(bool(record.get("categories")) for record in metadata.values())
    print(f"Classified {classified} of {len(metadata)} papers in {args.dataset_dir}")


if __name__ == "__main__":
    main()
