"""Run a reproducible, single-document retrieval comparison without an LLM API."""

import argparse
import hashlib
import json
import platform
from datetime import datetime, timezone
from importlib.metadata import version

import pymupdf
from evaluation.metrics import scores
from pathlib import Path

from retrieval.bm25_search import BM25Retriever
from retrieval.dense_search import DenseRetriever, encode_chunks, load_embedding_model
from retrieval.hybrid_search import HybridRetriever
from retrieval.index_manager import IndexManager
from retrieval.reranker import Reranker


ROOT = Path(__file__).resolve().parents[1]
QUESTIONS = Path(__file__).with_name("nist_questions.json")
SOURCE = "NIST_AI_RMF_1.0.pdf"
K = 5
CANDIDATES = 20


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--output", type=Path, default=ROOT / "evaluation" / "results.json")
    args = parser.parse_args()

    manager = IndexManager()
    manager.sync_documents()
    document_id = manager.resolve_document_id(SOURCE)
    chunks = manager.load_all_chunks([document_id])
    questions = json.loads(QUESTIONS.read_text(encoding="utf-8"))
    if len(questions) != 20 or len({q["id"] for q in questions}) != len(questions):
        raise ValueError("Expected 20 unique, manually annotated questions")

    labels = {}
    with pymupdf.open(manager.documents_dir / SOURCE) as pdf:
        for question in questions:
            page_number = question["page"]
            if question["evidence_quote"] not in pdf[page_number - 1].get_text("text", sort=True):
                raise ValueError(f"Gold quote missing in original PDF: {question['id']}")
            matches = {
                chunk["chunk_id"] for chunk in chunks
                if chunk["page"] == page_number
                and question["evidence_quote"] in chunk["text"]
            }
            if not matches:
                raise ValueError(f"Gold quote missing in index: {question['id']}")
            labels[question["id"]] = matches

    embedding = load_embedding_model()
    manager.ensure_dense_vectors(lambda items: encode_chunks(embedding, items), [document_id])
    snapshot = manager.load_snapshot([document_id])
    bm25 = BM25Retriever(snapshot.chunks)
    dense = DenseRetriever(snapshot.chunks, snapshot.vectors, embedding)
    hybrid = HybridRetriever(bm25, dense)
    reranker = Reranker()

    rows = []
    for index, question in enumerate(questions, 1):
        query = question["query"]
        candidates = hybrid.search(query, top_k=CANDIDATES, candidate_k=CANDIDATES)
        variants = {
            "bm25": bm25.search(query, top_k=K),
            "dense": dense.search(query, top_k=K),
            "hybrid": candidates[:K],
            "hybrid_reranker": reranker.rerank(query, candidates, top_k=K),
        }
        row = {
            "id": question["id"],
            "query": query,
            "pdf_page": question["page"],
            "gold_chunk_ids": sorted(labels[question["id"]]),
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

    summary = {
        name: {
            metric: round(sum(row["variants"][name][metric] for row in rows) / len(rows), 4)
            for metric in ("recall_at_5", "mrr_at_5", "ndcg_at_5")
        }
        for name in ("bm25", "dense", "hybrid", "hybrid_reranker")
    }
    result = {
        "run_at_utc": datetime.now(timezone.utc).isoformat(),
        "dataset": QUESTIONS.relative_to(ROOT).as_posix(),
        "source": SOURCE,
        "source_sha256": hashlib.sha256(
            (manager.documents_dir / SOURCE).read_bytes()
        ).hexdigest(),
        "python": platform.python_version(),
        "packages": {
            name: version(name)
            for name in ("sentence-transformers", "torch", "numpy", "rank-bm25", "jieba")
        },
        "questions": len(questions),
        "k": K,
        "candidate_k": CANDIDATES,
        "summary": summary,
        "queries": rows,
    }
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(result, ensure_ascii=False, indent=2), encoding="utf-8")
    print(json.dumps(summary, indent=2), flush=True)
    print(f"Saved {args.output}", flush=True)


if __name__ == "__main__":
    main()
