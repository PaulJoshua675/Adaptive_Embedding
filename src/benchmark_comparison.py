"""Run a small document-judged comparison over the persisted embedding stores."""

from __future__ import annotations

import argparse
import json
import os
import pickle
import statistics
import time
from collections import defaultdict
from datetime import datetime, timezone
from pathlib import Path, PureWindowsPath
from typing import Any

from config import EMBEDDING_MODELS, VECTOR_STORE_DIR
from src.classifier import classify_query
from src.embeddings import EmbeddingRegistry
from src.evaluation_suite import retrieval_metrics
from src.vector_db import FaissVectorStore


BENCHMARK_CASES = (
    {
        "query": "Explain scaled dot-product attention and multi-head attention.",
        "relevant_source": "1706.03762",
        "intent": "technical",
    },
    {
        "query": "How does the Transformer use positional encodings?",
        "relevant_source": "1706.03762",
        "intent": "conceptual",
    },
    {
        "query": "Attention Is All You Need",
        "relevant_source": "1706.03762",
        "intent": "entity",
    },
    {
        "query": "What is the supervised fine-tuning stage in InstructGPT?",
        "relevant_source": "2203.02155",
        "intent": "conceptual",
    },
    {
        "query": "Describe the reward model objective used for instruction following.",
        "relevant_source": "2203.02155",
        "intent": "technical",
    },
    {
        "query": "Training language models to follow instructions with human feedback",
        "relevant_source": "2203.02155",
        "intent": "entity",
    },
    {
        "query": "How was Llama 2 fine-tuned with human preference data?",
        "relevant_source": "2307.09288",
        "intent": "conceptual",
    },
    {
        "query": "Compare the context windows reported for Llama 2 models.",
        "relevant_source": "2307.09288",
        "intent": "technical",
    },
    {
        "query": "Llama 2",
        "relevant_source": "2307.09288",
        "intent": "entity",
    },
)
CONFIGURATIONS = ("BGE baseline", "E5 baseline", "MiniLM baseline", "Adaptive RAG")


def _source_id(document: dict[str, Any]) -> str:
    source = str(document.get("source", ""))
    return Path(PureWindowsPath(source).name).stem


def _mean(values: list[float]) -> float:
    return statistics.fmean(values) if values else 0.0


def run_benchmark(index_dir: Path, runs: int = 5, top_k: int = 3) -> dict[str, Any]:
    if runs < 1 or top_k < 1:
        raise ValueError("runs and top_k must both be at least 1")

    with (index_dir / "metadata.pkl").open("rb") as handle:
        metadata = pickle.load(handle)
    keys = list(metadata.get("embedding_keys", []))
    models = dict(metadata.get("embedding_models", {}))
    if not set(EMBEDDING_MODELS).issubset(keys) or not set(EMBEDDING_MODELS).issubset(models):
        raise ValueError(f"Expected BGE, E5, and MiniLM stores under {index_dir}")

    stores = {key: FaissVectorStore.load(index_dir / f"faiss_{key}") for key in EMBEDDING_MODELS}
    registry = EmbeddingRegistry({key: models[key] for key in EMBEDDING_MODELS})
    observations: dict[str, list[dict[str, Any]]] = defaultdict(list)
    baseline_keys = {"BGE baseline": "bge", "E5 baseline": "e5", "MiniLM baseline": "minilm"}

    for case in BENCHMARK_CASES:
        adaptive_route = classify_query(case["query"])
        adaptive_key = str(adaptive_route["embedding_key"])
        for configuration in CONFIGURATIONS:
            encoder_key = baseline_keys.get(configuration, adaptive_key)
            encoder = registry[encoder_key]
            latencies: list[float] = []
            documents: list[dict[str, Any]] = []
            for run_index in range(runs + 1):
                started = time.perf_counter()
                query_vector = encoder.encode_query(case["query"])
                retrieved = stores[encoder_key].search(query_vector, top_k)
                elapsed_ms = (time.perf_counter() - started) * 1000
                documents = [document for _, document in retrieved]
                if run_index > 0:
                    latencies.append(elapsed_ms)

            retrieved_sources = list(dict.fromkeys(_source_id(document) for document in documents))
            metrics = retrieval_metrics(retrieved_sources, [case["relevant_source"]], k=top_k)
            observations[configuration].append(
                {
                    "query": case["query"],
                    "intent": case["intent"],
                    "relevant_source": case["relevant_source"],
                    "embedding_key": encoder_key,
                    "adaptive_route": adaptive_key,
                    "adaptive_route_confidence": float(adaptive_route.get("confidence", 0.0)),
                    "retrieved_sources": retrieved_sources,
                    "latency_median_ms": statistics.median(latencies) if latencies else 0.0,
                    "metrics": metrics,
                }
            )

    summary: dict[str, Any] = {}
    for configuration in CONFIGURATIONS:
        rows = observations[configuration]
        metric_names = (f"hit@{top_k}", "precision", "recall", "f1")
        summary[configuration] = {
            **{name: _mean([float(row["metrics"][name]) for row in rows]) for name in metric_names},
            "median_latency_ms": statistics.median(row["latency_median_ms"] for row in rows),
            "query_count": len(rows),
        }

    return {
        "generated_at": datetime.now(timezone.utc).isoformat(),
        "index_dir": str(index_dir.resolve()),
        "chunk_count": len(metadata.get("chunks", [])),
        "source_count": len({str(chunk.get("source", "")) for chunk in metadata.get("chunks", [])}),
        "embedding_models": {key: models[key] for key in EMBEDDING_MODELS},
        "embedding_backend_setting": "transformers" if os.getenv("EMBEDDING_USE_TRANSFORMERS", "0").lower() in {"1", "true", "yes"} else "feature-hash fallback",
        "reranking": "disabled to isolate encoder/routing retrieval quality",
        "judgment": "one manually mapped relevant paper per query; exploratory smoke benchmark, not a full relevance set",
        "runs_per_query": runs,
        "top_k": top_k,
        "cases": list(BENCHMARK_CASES),
        "summary": summary,
        "results": dict(observations),
    }


def render_markdown(result: dict[str, Any]) -> str:
    top_k = result["top_k"]
    lines = [
        "# Review 3 Retrieval Comparison",
        "",
        f"- Generated: `{result['generated_at']}`",
        f"- Corpus: {result['source_count']:,} sources, {result['chunk_count']:,} chunks",
        f"- Encoder backend setting: `{result['embedding_backend_setting']}`",
        f"- Judgment: {result['judgment']}",
        f"- Protocol: {result['runs_per_query']} measured runs after one warm-up; top-{top_k} retrieval; reranking disabled.",
        "",
        "| Configuration | Hit@K | Precision | Recall | F1 | Median query latency (ms) |",
        "|---|---:|---:|---:|---:|---:|",
    ]
    for name, metrics in result["summary"].items():
        lines.append(
            f"| {name} | {metrics[f'hit@{top_k}']:.3f} | {metrics['precision']:.3f} | "
            f"{metrics['recall']:.3f} | {metrics['f1']:.3f} | {metrics['median_latency_ms']:.2f} |"
        )
    lines.extend(["", "## Query Routes", "", "| Query | Gold source | Adaptive route | Adaptive hit@K |", "|---|---|---|---:|"])
    for row in result["results"]["Adaptive RAG"]:
        lines.append(
            f"| {row['query']} | `{row['relevant_source']}` | `{row['adaptive_route']}` "
            f"({row['adaptive_route_confidence']:.2f}) | {row['metrics'][f'hit@{top_k}']:.0f} |"
        )
    lines.extend(["", "Latency covers query encoding and FAISS search only; it excludes loading, classification, reranking, and answer generation.", ""])
    return "\n".join(lines)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--index-dir", type=Path, default=VECTOR_STORE_DIR)
    parser.add_argument("--runs", type=int, default=5)
    parser.add_argument("--top-k", type=int, default=3)
    parser.add_argument("--json-out", type=Path, default=Path("tests/review3_benchmark_results.json"))
    parser.add_argument("--markdown-out", type=Path, default=Path("tests/review3_benchmark_results.md"))
    args = parser.parse_args()

    result = run_benchmark(args.index_dir, runs=args.runs, top_k=args.top_k)
    args.json_out.parent.mkdir(parents=True, exist_ok=True)
    args.markdown_out.parent.mkdir(parents=True, exist_ok=True)
    args.json_out.write_text(json.dumps(result, indent=2), encoding="utf-8")
    args.markdown_out.write_text(render_markdown(result), encoding="utf-8")
    print(render_markdown(result))
    print(f"JSON results: {args.json_out}")
    print(f"Markdown report: {args.markdown_out}")


if __name__ == "__main__":
    main()