"""Retrieval metrics for binary relevance labels."""

import math


def scores(results, relevant, k=5):
    if not relevant:
        raise ValueError("At least one relevant chunk is required")
    hits = [int(result["chunk_id"] in relevant) for result in results[:k]]
    recall = sum(hits) / len(relevant)
    reciprocal_rank = next((1 / rank for rank, hit in enumerate(hits, 1) if hit), 0)
    dcg = sum(hit / math.log2(rank + 1) for rank, hit in enumerate(hits, 1))
    ideal = sum(1 / math.log2(rank + 1) for rank in range(1, min(k, len(relevant)) + 1))
    return {
        f"recall_at_{k}": recall,
        f"mrr_at_{k}": reciprocal_rank,
        f"ndcg_at_{k}": dcg / ideal,
    }
