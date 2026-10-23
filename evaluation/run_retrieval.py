"""Run a reproducible, single-document retrieval comparison without an LLM API."""

import argparse
import hashlib
import json
import platform
from datetime import datetime, timezone
from importlib.metadata import version
from pathlib import Path

import pymupdf
from huggingface_hub import scan_cache_dir

from evaluation.metrics import scores
from retrieval.bm25_search import BM25Retriever
from retrieval.chunking import DEFAULT_CHUNK_SIZE, DEFAULT_OVERLAP
from retrieval.dense_search import (
    MODEL_NAME as EMBEDDING_MODEL,
    DenseRetriever,
    encode_chunks,
    load_embedding_model,
)
from retrieval.hybrid_search import (
    DEFAULT_BM25_WEIGHT,
    DEFAULT_DENSE_WEIGHT,
    HybridRetriever,
)
from retrieval.index_manager import IndexManager
from retrieval.reranker import MODEL_NAME as RERANKER_MODEL, Reranker


ROOT = Path(__file__).resolve().parents[1]
QUESTIONS = Path(__file__).with_name("nist_questions.json")
SOURCE = "NIST_AI_RMF_1.0.pdf"
K = 5
CANDIDATES = 20
RRF_K = 60


def cached_revision(repo_id):
    """Record the cached main revision used by default model loading."""
    for repo in scan_cache_dir().repos:
        if repo.repo_id == repo_id:
            for revision in repo.revisions:
                if "main" in revision.refs:
                    return revision.commit_hash
    raise RuntimeError(f"Cached main revision not found for {repo_id}")


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--output", type=Path, default=ROOT / "evaluation" / "results.json")
    args = parser.parse_args()

    manager = IndexManager()
    manager.sync_documents()
    document_id = manager.resolve_document_id(SOURCE)
    chunks = manager.load_all_chunks([document_id])
    dataset = json.loads(QUESTIONS.read_text(encoding="utf-8"))
    if dataset.get("version") != 2:
        raise ValueError("Expected evidence-group annotation version 2")
    questions = dataset["questions"]
    if len(questions) != 20 or len({q["id"] for q in questions}) != len(questions):
        raise ValueError("Expected 20 unique, manually annotated questions")

    labels = {}
    annotation_details = {}
    with pymupdf.open(manager.documents_dir / SOURCE) as pdf:
        for question in questions:
            evidence = question["evidence_groups"]
            if not evidence:
                raise ValueError(f"No evidence groups: {question['id']}")
            groups = []
            details = []
            for group in evidence:
                page_number = group["page"]
                quote = group["quote"]
                if not quote or not 1 <= page_number <= len(pdf):
                    raise ValueError(f"Invalid evidence group: {question['id']}")
                if quote not in pdf[page_number - 1].get_text("text", sort=True):
                    raise ValueError(f"Gold quote missing in original PDF: {question['id']}")
                matches = {
                    chunk["chunk_id"] for chunk in chunks
                    if chunk["page"] == page_number and quote in chunk["text"]
                }
                if not matches:
                    raise ValueError(f"Gold quote missing in index: {question['id']}")
                groups.append(matches)
                details.append({"pdf_page": page_number, "quote": quote,
                                "matching_chunk_ids": sorted(matches)})
            labels[question["id"]] = groups
            annotation_details[question["id"]] = details

    embedding = load_embedding_model()
    manager.ensure_dense_vectors(lambda items: encode_chunks(embedding, items), [document_id])
    snapshot = manager.load_snapshot([document_id])
    bm25 = BM25Retriever(snapshot.chunks)
    dense = DenseRetriever(snapshot.chunks, snapshot.vectors, embedding)
    hybrid = HybridRetriever(bm25, dense)
    reranker = Reranker()
    model_revisions = {
        "embedding": cached_revision(EMBEDDING_MODEL),
        "reranker": cached_revision(RERANKER_MODEL),
    }

    rows = []
    for index, question in enumerate(questions, 1):
        query = question["query"]
        candidates = hybrid.search(
            query, top_k=CANDIDATES, candidate_k=CANDIDATES,
            rrf_k=RRF_K, bm25_weight=DEFAULT_BM25_WEIGHT,
            dense_weight=DEFAULT_DENSE_WEIGHT,
        )
        variants = {
            "bm25": bm25.search(query, top_k=K),
            "dense": dense.search(query, top_k=K),
            "hybrid": candidates[:K],
            "hybrid_reranker": reranker.rerank(query, candidates, top_k=K),
        }
        row = {
            "id": question["id"],
            "query": query,
            "evidence_groups": annotation_details[question["id"]],
            "variants": {
                name: {
                    **scores(results, labels[question["id"]], k=K),
                    "top_5_chunk_ids": [item["chunk_id"] for item in results[:K]],
                }
                for name, results in variants.items()
            },
        }
        rows.append(row)
        print(f"{index:02}/{len(questions)} {question['id']}", flush=True)

    summary = {}
    for name in ("bm25", "dense", "hybrid", "hybrid_reranker"):
        hits = sum(row["variants"][name]["hit_at_5"] for row in rows)
        summary[name] = {
            "hit_count_at_5": hits,
            "hit_rate_at_5": round(hits / len(rows), 4),
            **{
                metric: round(
                    sum(row["variants"][name][metric] for row in rows) / len(rows), 4
                )
                for metric in (
                    "evidence_recall_at_5", "mrr_at_5", "evidence_ndcg_at_5"
                )
            },
        }
    result = {
        "evaluation_format_version": 2,
        "run_at_utc": datetime.now(timezone.utc).isoformat(),
        "dataset": QUESTIONS.relative_to(ROOT).as_posix(),
        "dataset_version": dataset["version"],
        "dataset_sha256": hashlib.sha256(QUESTIONS.read_bytes()).hexdigest(),
        "source": SOURCE,
        "source_sha256": hashlib.sha256(
            (manager.documents_dir / SOURCE).read_bytes()
        ).hexdigest(),
        "python": platform.python_version(),
        "packages": {
            name: version(name)
            for name in (
                "sentence-transformers", "transformers", "huggingface-hub",
                "torch", "numpy", "rank-bm25", "jieba", "PyMuPDF",
            )
        },
        "questions": len(questions),
        "indexed_chunks": len(chunks),
        "config": {
            "k": K,
            "candidate_k_per_retriever": CANDIDATES,
            "reranker_input_candidates": CANDIDATES,
            "chunk_size_characters": DEFAULT_CHUNK_SIZE,
            "chunk_overlap_characters": DEFAULT_OVERLAP,
            "rrf_k": RRF_K,
            "bm25_weight": DEFAULT_BM25_WEIGHT,
            "dense_weight": DEFAULT_DENSE_WEIGHT,
            "embedding_model": EMBEDDING_MODEL,
            "embedding_revision": model_revisions["embedding"],
            "embedding_normalized": True,
            "reranker_model": RERANKER_MODEL,
            "reranker_revision": model_revisions["reranker"],
            "reranker_max_length": 512,
            "metric_unit": "annotated evidence group",
        },
        "summary": summary,
        "queries": rows,
    }
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(result, ensure_ascii=False, indent=2), encoding="utf-8")
    print(json.dumps(summary, indent=2), flush=True)
    print(f"Saved {args.output}", flush=True)


if __name__ == "__main__":
    main()
