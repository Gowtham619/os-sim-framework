"""Small metric helpers: latency summaries, recall, rank-based hit rates."""
from __future__ import annotations

from typing import Iterable, Sequence

import numpy as np


def percentile(values: Sequence[float], p: float) -> float | None:
    if not len(values):
        return None
    return float(np.percentile(np.asarray(values, dtype=float), p))


def summarize_latency(ms: Sequence[float]) -> dict:
    if not len(ms):
        return {"count": 0}
    a = np.asarray(ms, dtype=float)
    return {
        "count": int(a.size),
        "mean": round(float(a.mean()), 2),
        "p50": round(float(np.percentile(a, 50)), 2),
        "p90": round(float(np.percentile(a, 90)), 2),
        "p95": round(float(np.percentile(a, 95)), 2),
        "p99": round(float(np.percentile(a, 99)), 2),
        "max": round(float(a.max()), 2),
    }


def recall_at_k(truth: Sequence[str], got: Sequence[str], k: int | None = None) -> float:
    truth = list(truth)[:k] if k else list(truth)
    if not truth:
        return 1.0
    got_k = list(got)[:k] if k else list(got)
    return len(set(truth) & set(got_k)) / len(truth)


def rank_of(target: str, ids: Sequence[str]) -> int | None:
    for i, x in enumerate(ids):
        if x == target:
            return i + 1
    return None


TIE_EPS = 1e-4


def rank_with_ties(target: str, hits: Sequence) -> int | None:
    """Rank of `target` among hits (objects with .id and .similarity), counting only hits that are strictly
    more similar. Cases with identical content have identical vectors and the engine may return them in any
    order, so a tie must not count as a miss."""
    own = next((h for h in hits if h.id == target), None)
    if own is None:
        return None
    return 1 + sum(1 for h in hits if h.similarity > own.similarity + TIE_EPS)


def hit_rate(ranks: Iterable[int | None], k: int) -> float | None:
    ranks = list(ranks)
    if not ranks:
        return None
    return sum(1 for r in ranks if r is not None and r <= k) / len(ranks)


def mrr(ranks: Iterable[int | None]) -> float | None:
    ranks = list(ranks)
    if not ranks:
        return None
    return sum((1.0 / r) if r else 0.0 for r in ranks) / len(ranks)


def r4(x: float | None) -> float | None:
    if x is None:
        return None
    if isinstance(x, int) and not isinstance(x, bool):
        return x
    return round(float(x), 4)
