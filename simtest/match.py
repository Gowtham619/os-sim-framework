"""Dataset-to-dataset matching: every record of dataset A is searched against the indexed corpus (dataset B)."""
from __future__ import annotations

import csv
import os
import time
from concurrent.futures import ThreadPoolExecutor
from typing import Callable

from .metrics import hit_rate, mrr, r4, rank_of


def run_match(
    wb,
    records: list[dict],
    k: int,
    min_sim: float,
    out_csv: str,
    batch: int = 50,
    workers: int = 4,
    log: Callable[[str], None] = print,
) -> dict:
    t0 = time.perf_counter()
    vecs = wb.embedder.embed_many([r["text"] for r in records], progress=True)
    t_embed = time.perf_counter() - t0

    batches = [list(range(i, min(i + batch, len(records)))) for i in range(0, len(records), batch)]

    def do(idxs: list[int]):
        return idxs, wb.store.msearch([vecs[i] for i in idxs], k)

    t1 = time.perf_counter()
    results = [None] * len(records)
    with ThreadPoolExecutor(max_workers=workers) as ex:
        for idxs, res in ex.map(do, batches):
            for i, hits in zip(idxs, res):
                results[i] = hits
    t_search = time.perf_counter() - t1

    os.makedirs(os.path.dirname(os.path.abspath(out_csv)), exist_ok=True)
    rows = 0
    with open(out_csv, "w", newline="", encoding="utf-8") as fh:
        w = csv.writer(fh)
        w.writerow(["query_id", "rank", "match_id", "score", "similarity", "above_min_similarity"])
        for rec, hits in zip(records, results):
            for rank, h in enumerate(hits, 1):
                w.writerow([rec["id"], rank, h.id, round(h.score, 6), round(h.similarity, 6), h.similarity >= min_sim])
                rows += 1

    metrics: dict = {
        "queries": len(records),
        "k": k,
        "rows_written": rows,
        "embed_seconds": round(t_embed, 2),
        "search_seconds": round(t_search, 2),
        "search_qps": round(len(records) / t_search, 1) if t_search > 0 else None,
        "top1_above_min_similarity": sum(1 for h in results if h and h[0].similarity >= min_sim),
        "csv": out_csv,
    }
    labelled = [(r, h) for r, h in zip(records, results) if "expected_match" in r]
    if labelled:
        pos = [rank_of(r["expected_match"], [x.id for x in h]) for r, h in labelled if r["expected_match"]]
        neg = [h for r, h in labelled if not r["expected_match"]]
        metrics.update(
            {
                "labelled_positive": len(pos),
                "hit_at_1": r4(hit_rate(pos, 1)),
                "hit_at_k": r4(hit_rate(pos, k)),
                "mrr": r4(mrr(pos)),
                "labelled_negative": len(neg),
            }
        )
        if min_sim > 0 and neg:
            metrics["false_positive_rate"] = r4(sum(1 for h in neg if h and h[0].similarity >= min_sim) / len(neg))
    return metrics
