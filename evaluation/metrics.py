"""Metrics over annotated evidence groups, not overlapping chunk copies."""

import math


def scores(results, evidence_groups, k=5):
    """Count each required evidence passage once, regardless of chunk overlap.

    Each group contains the chunk IDs that reproduce one annotated passage.
    A result can cover several groups; later overlapping copies add no credit.
    """
    if k <= 0:
        raise ValueError("k must be positive")
    groups = [set(group) for group in evidence_groups]
    if not groups or any(not group for group in groups):
        raise ValueError("At least one nonempty evidence group is required")

    first_ranks = {}
    for rank, result in enumerate(results[:k], start=1):
        chunk_id = result["chunk_id"]
        for index, group in enumerate(groups):
            if index not in first_ranks and chunk_id in group:
                first_ranks[index] = rank

    first_hit = min(first_ranks.values(), default=None)
    # The ideal score is one first-rank hit per evidence group. This custom
    # group-level discounted coverage is bounded by 1 even when one chunk
    # contains several distinct passages.
    discounted_coverage = sum(
        1 / math.log2(rank + 1) for rank in first_ranks.values()
    ) / len(groups)
    return {
        f"hit_at_{k}": int(first_hit is not None),
        f"evidence_recall_at_{k}": len(first_ranks) / len(groups),
        f"mrr_at_{k}": 1 / first_hit if first_hit is not None else 0.0,
        f"evidence_ndcg_at_{k}": discounted_coverage,
    }
