"""Simple retrieval evaluation helpers for CloudInsight."""

from __future__ import annotations


def precision_at_k(relevant_hits: list[bool], k: int) -> float:
    """Compute precision@k for a ranked relevance list."""

    if k <= 0:
        raise ValueError("k must be greater than zero.")

    truncated_hits = relevant_hits[:k]
    if not truncated_hits:
        return 0.0

    return sum(truncated_hits) / len(truncated_hits)


def recall_at_k(relevant_hits: list[bool], total_relevant: int, k: int) -> float:
    """Compute recall@k for a ranked relevance list."""

    if total_relevant <= 0:
        return 0.0

    return sum(relevant_hits[:k]) / total_relevant
