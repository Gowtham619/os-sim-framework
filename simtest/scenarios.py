"""Test scenarios for similarity search over existing, new, updated and deleted cases.

Every scenario compares what the target (managed domain / Serverless collection / local) returns against what the
framework knows must be true, using an in-memory exact mirror of the corpus as ground truth.
"""
from __future__ import annotations

import time
import traceback
from concurrent.futures import ThreadPoolExecutor
from typing import Callable, Optional

import numpy as np

from .data import perturb_text
from .embedder import BaseEmbedder
from .metrics import hit_rate, mrr, percentile, r4, rank_of, rank_with_ties, recall_at_k, summarize_latency
from .store import Doc, LocalStore, VectorStore

_EVAL_ONLY_KEYS = {"id", "text", "expected_neighbor", "expected_match"}


def _mean(xs):
    xs = list(xs)
    return float(sum(xs) / len(xs)) if xs else None


class Workbench:
    """Holds the target store, the embedder, and an exact in-memory mirror of what *should* be in the index."""

    def __init__(self, cfg: dict, store: VectorStore, embedder: BaseEmbedder):
        import random

        self.cfg = cfg
        self.store = store
        self.embedder = embedder
        run = cfg["run"]
        self.k = int(run["k"])
        self.n_queries = int(run["n_queries"])
        self.drop = float(run["perturb_drop"])
        self.query_workers = int(run["query_workers"])
        self.vis_timeout = float(run["visibility_timeout"])
        self.vis_sample = int(run["visibility_sample"])
        self.update_sample = run.get("update_sample")
        self.delete_sample = int(run["delete_sample"])
        self.thresholds = cfg["thresholds"]
        self.rng = random.Random(int(run["seed"]))
        self.mirror = LocalStore("mirror")
        self.records: dict[str, dict] = {}

    # -- corpus ----------------------------------------------------------------------------------
    def load_corpus(self, records: list[dict], progress: bool = False) -> None:
        vecs = self.embedder.embed_many([r["text"] for r in records], progress=progress)
        for rec, vec in zip(records, vecs):
            meta = {k: v for k, v in rec.items() if k not in _EVAL_ONLY_KEYS}
            self.records[rec["id"]] = {"text": rec["text"], "version": 1, "meta": meta}
            self.mirror.upsert(Doc(rec["id"], rec["text"], vec, 1, meta))

    def docs(self) -> list[Doc]:
        out = []
        for _id in self.mirror.ids():
            rec = self.records[_id]
            out.append(Doc(_id, rec["text"], self.mirror.get_vector(_id), self.mirror.get_version(_id), rec["meta"]))
        return out

    def ingest(self, recreate: bool = True, progress: bool = True) -> dict:
        self.store.create_index(recreate=recreate)
        stats = self.store.bulk_index(self.docs(), progress=progress)
        self.store.refresh()
        stats["docs_per_sec"] = round(stats["indexed"] / stats["seconds"], 1) if stats.get("seconds") else None
        return stats

    def ingest_streaming(
        self,
        records: list[dict],
        recreate: bool = True,
        start: int = 0,
        chunk_size: int = 2000,
        on_chunk: Optional[Callable[[int, int, dict], None]] = None,
    ) -> dict:
        """Embed and index `records` chunk by chunk, so memory stays flat and a failure can resume at a chunk boundary.

        Does not fill the in-memory mirror (only the scenarios need it). Stops at the first chunk with indexing errors
        without calling `on_chunk` for it, so a checkpoint never moves past data that was not indexed.
        """
        self.store.create_index(recreate=recreate)
        total = {"indexed": 0, "errors": 0, "throttled": 0, "retries": 0, "seconds": 0.0, "stopped_at": None}
        t0 = time.perf_counter()
        for i in range(start, len(records), chunk_size):
            chunk = records[i : i + chunk_size]
            vecs = self.embedder.embed_many([r["text"] for r in chunk])
            docs = [
                Doc(r["id"], r["text"], v, 1, {k: val for k, val in r.items() if k not in _EVAL_ONLY_KEYS})
                for r, v in zip(chunk, vecs)
            ]
            res = self.store.bulk_index(docs)
            for key in ("indexed", "errors", "throttled", "retries"):
                total[key] += res[key]
            if res["errors"]:
                total["stopped_at"] = i
                break
            if on_chunk:
                on_chunk(i + len(chunk), len(records), total)
        total["seconds"] = time.perf_counter() - t0
        self.store.refresh()
        total["docs_per_sec"] = round(total["indexed"] / total["seconds"], 1) if total["seconds"] else None
        return total

    # -- query helpers ---------------------------------------------------------------------------
    def sample_ids(self, n: int) -> list[str]:
        ids = sorted(self.mirror.ids())
        return self.rng.sample(ids, min(n, len(ids)))

    def perturbed(self, text: str) -> str:
        return perturb_text(text, self.rng, self.drop)

    def run_queries(self, vectors, k: Optional[int] = None, workers: Optional[int] = None):
        k = k or self.k
        results: list = [None] * len(vectors)
        lats = [0.0] * len(vectors)

        def one(i: int) -> None:
            t0 = time.perf_counter()
            results[i] = self.store.search(vectors[i], k)
            lats[i] = (time.perf_counter() - t0) * 1000

        with ThreadPoolExecutor(max_workers=workers or self.query_workers) as ex:
            list(ex.map(one, range(len(vectors))))
        return results, lats


# -------------------------------------------------------------------------------------------------
def evaluate(metrics: dict, thresholds: dict, spec: list[tuple[str, str, str]]) -> list[dict]:
    checks = []
    for metric, op, tkey in spec:
        thr, val = thresholds.get(tkey), metrics.get(metric)
        if thr is None or val is None:
            continue
        ok = val >= thr if op == ">=" else val <= thr if op == "<=" else val == thr
        checks.append({"metric": metric, "op": op, "threshold": thr, "value": r4(val), "passed": bool(ok)})
    return checks


def _finish(name: str, t0: float, wb: Workbench, metrics: dict, spec: list[tuple[str, str, str]]) -> dict:
    checks = evaluate(metrics, wb.thresholds, spec)
    return {
        "name": name,
        "seconds": round(time.perf_counter() - t0, 2),
        "metrics": metrics,
        "checks": checks,
        "passed": all(c["passed"] for c in checks),
    }


def _tt_stats(times: list[Optional[float]]) -> dict:
    ok = [t for t in times if t is not None]
    return {
        "visibility_timeouts": sum(1 for t in times if t is None),
        "time_to_searchable_p50_s": r4(percentile(ok, 50)),
        "time_to_searchable_p95_s": r4(percentile(ok, 95)),
        "time_to_searchable_max_s": r4(max(ok)) if ok else None,
    }


# -------------------------------------------------------------------------------------------------
def scenario_recall(wb: Workbench, label: str = "recall") -> dict:
    """ANN recall@k vs exact search on the mirror, using lightly re-worded versions of existing cases."""
    t0 = time.perf_counter()
    ids = wb.sample_ids(wb.n_queries)
    qv = wb.embedder.embed_many([wb.perturbed(wb.records[i]["text"]) for i in ids])
    truths = [[h.id for h in wb.mirror.search(v, wb.k)] for v in qv]
    results, lats = wb.run_queries(qv, wb.k)
    recalls = [recall_at_k(t, [h.id for h in r], wb.k) for t, r in zip(truths, results)]
    metrics = {
        "queries": len(ids),
        "k": wb.k,
        "recall_at_k": r4(_mean(recalls)),
        "recall_min": r4(min(recalls)) if recalls else None,
        "latency_ms": summarize_latency(lats),
    }
    return _finish(label, t0, wb, metrics, [("recall_at_k", ">=", "recall_at_k")])


def scenario_existing(wb: Workbench) -> dict:
    """Existing cases: querying with the exact text must return the case itself; re-worded text should still find it."""
    t0 = time.perf_counter()
    ids = wb.sample_ids(wb.n_queries)
    exact_v = wb.embedder.embed_many([wb.records[i]["text"] for i in ids])
    pert_v = wb.embedder.embed_many([wb.perturbed(wb.records[i]["text"]) for i in ids])
    res_e, lat_e = wb.run_queries(exact_v)
    res_p, lat_p = wb.run_queries(pert_v)
    ranks_e = [rank_with_ties(i, r) for i, r in zip(ids, res_e)]
    ranks_p = [rank_of(i, [h.id for h in r]) for i, r in zip(ids, res_p)]
    metrics = {
        "queries": len(ids),
        "k": wb.k,
        "exact_hit_at_1": r4(hit_rate(ranks_e, 1)),
        "exact_mrr": r4(mrr(ranks_e)),
        "perturbed_hit_at_1": r4(hit_rate(ranks_p, 1)),
        "perturbed_hit_at_k": r4(hit_rate(ranks_p, wb.k)),
        "perturbed_mrr": r4(mrr(ranks_p)),
        "exact_latency_ms": summarize_latency(lat_e),
        "perturbed_latency_ms": summarize_latency(lat_p),
    }
    spec = [
        ("exact_hit_at_1", ">=", "existing_exact_hit_at_1"),
        ("perturbed_hit_at_k", ">=", "existing_perturbed_hit_at_k"),
    ]
    return _finish("existing", t0, wb, metrics, spec)


def scenario_new(wb: Workbench, new_records: list[dict]) -> dict:
    """New cases: (1) before indexing, do they find their expected similar existing case? (2) after indexing,
    how long until they are searchable, and does each one retrieve itself?"""
    t0 = time.perf_counter()
    vecs = wb.embedder.embed_many([r["text"] for r in new_records])
    docs = [
        Doc(r["id"], r["text"], v, 1, {k: val for k, val in r.items() if k not in _EVAL_ONLY_KEYS})
        for r, v in zip(new_records, vecs)
    ]

    # 1) incoming case -> similar existing cases (the production flow: search first, then store)
    pre_res, pre_lat = wb.run_queries(vecs)
    exp_ranks, novel_top1 = [], []
    for r, hits in zip(new_records, pre_res):
        ids = [h.id for h in hits]
        if r.get("expected_neighbor"):
            exp_ranks.append(rank_of(r["expected_neighbor"], ids))
        elif hits:
            novel_top1.append(hits[0].similarity)

    # 2) index them: a sample one-by-one to time "searchable", the rest in bulk
    sample = docs[: wb.vis_sample]
    rest = docs[wb.vis_sample :]

    def one(d: Doc) -> Optional[float]:
        wb.store.upsert(d)
        return wb.store.wait_visible(d.id, d.vector, timeout=wb.vis_timeout, expect_version=d.version)

    times: list[Optional[float]] = []
    if sample:
        with ThreadPoolExecutor(max_workers=min(8, len(sample))) as ex:
            times = list(ex.map(one, sample))
    bulk_stats = {}
    if rest:
        bulk_stats = wb.store.bulk_index(rest)
        wb.store.refresh()
        for d in wb.rng.sample(rest, min(5, len(rest))):  # let the bulk become visible before verifying
            wb.store.wait_visible(d.id, d.vector, timeout=wb.vis_timeout, expect_version=d.version)

    # 3) every new case must now retrieve itself
    post_res, post_lat = wb.run_queries(vecs)
    post_ranks = [rank_with_ties(d.id, hits) for d, hits in zip(docs, post_res)]

    for d in docs:
        wb.mirror.upsert(d)
        wb.records[d.id] = {"text": d.text, "version": 1, "meta": d.meta}

    metrics = {
        "new_cases": len(docs),
        "k": wb.k,
        "neighbor_queries": len(exp_ranks),
        "neighbor_hit_at_1": r4(hit_rate(exp_ranks, 1)),
        "neighbor_hit_at_k": r4(hit_rate(exp_ranks, wb.k)),
        "neighbor_mrr": r4(mrr(exp_ranks)),
        "novel_top1_similarity_p50": r4(percentile(novel_top1, 50)),
        "post_index_hit_at_1": r4(hit_rate(post_ranks, 1)),
        "pre_index_latency_ms": summarize_latency(pre_lat),
        "post_index_latency_ms": summarize_latency(post_lat),
        "timed_individually": len(sample),
        "bulk": {k: (round(v, 2) if isinstance(v, float) else v) for k, v in bulk_stats.items()},
        **_tt_stats(times),
    }
    spec = [
        ("neighbor_hit_at_k", ">=", "new_neighbor_hit_at_k"),
        ("post_index_hit_at_1", ">=", "new_post_index_hit_at_1"),
        ("visibility_timeouts", "<=", "visibility_timeouts"),
        ("time_to_searchable_p95_s", "<=", "time_to_searchable_p95_s"),
    ]
    return _finish("new", t0, wb, metrics, spec)


def scenario_updated(wb: Workbench, updates: list[dict]) -> dict:
    """Updated cases: the new content must be searchable, the old embedding must be gone, and no stale/duplicate
    copies may remain."""
    t0 = time.perf_counter()
    recs = [u for u in updates if u["id"] in wb.records]
    skipped = len(updates) - len(recs)
    if wb.update_sample:
        recs = recs[: int(wb.update_sample)]
    new_vecs = wb.embedder.embed_many([u["text"] for u in recs])
    old_vecs = {u["id"]: wb.mirror.get_vector(u["id"]) for u in recs}
    docs = [
        Doc(u["id"], u["text"], v, wb.mirror.get_version(u["id"]) + 1, wb.records[u["id"]]["meta"])
        for u, v in zip(recs, new_vecs)
    ]

    def one(d: Doc) -> Optional[float]:
        wb.store.upsert(d)
        return wb.store.wait_visible(d.id, d.vector, timeout=wb.vis_timeout, expect_version=d.version)

    times: list[Optional[float]] = []
    if docs:
        with ThreadPoolExecutor(max_workers=min(8, len(docs))) as ex:
            times = list(ex.map(one, docs))

    res_new, lat_new = wb.run_queries([d.vector for d in docs]) if docs else ([], [])
    ranks, self_sims, dup_docs, stale = [], [], 0, 0
    for d, hits in zip(docs, res_new):
        ranks.append(rank_with_ties(d.id, hits))
        same = [h for h in hits if h.id == d.id]
        if same:
            self_sims.append(same[0].similarity)
        if len(same) > 1:
            dup_docs += 1
        stale += sum(1 for h in same if h.version != d.version)

    # the old embedding must no longer be served for this id (only meaningful if content really changed)
    changed = [d for d in docs if float(np.dot(old_vecs[d.id], d.vector)) < 0.95]
    res_old, _ = wb.run_queries([old_vecs[d.id] for d in changed]) if changed else ([], [])
    old_served = sum(1 for d, hits in zip(changed, res_old) if any(h.id == d.id and h.similarity >= 0.995 for h in hits))

    for d in docs:
        wb.mirror.upsert(d)
        wb.records[d.id] = {"text": d.text, "version": d.version, "meta": d.meta}

    metrics = {
        "updated_cases": len(docs),
        "skipped_not_in_index": skipped,
        "hit_at_1": r4(hit_rate(ranks, 1)),
        "self_similarity_p50": r4(percentile(self_sims, 50)),
        "self_similarity_min": r4(min(self_sims)) if self_sims else None,
        "duplicates": dup_docs,
        "stale_versions": stale,
        "old_vector_checked": len(changed),
        "old_vector_served": old_served,
        "latency_ms": summarize_latency(lat_new),
        **_tt_stats(times),
    }
    spec = [
        ("hit_at_1", ">=", "updated_hit_at_1"),
        ("duplicates", "<=", "updated_duplicates"),
        ("stale_versions", "<=", "updated_stale_versions"),
        ("old_vector_served", "<=", "updated_old_vector_served"),
        ("visibility_timeouts", "<=", "visibility_timeouts"),
        ("time_to_searchable_p95_s", "<=", "time_to_searchable_p95_s"),
    ]
    return _finish("updated", t0, wb, metrics, spec)


def scenario_deleted(wb: Workbench) -> dict:
    """Deleted cases must stop appearing in results."""
    t0 = time.perf_counter()
    ids = wb.sample_ids(wb.delete_sample)
    vecs = {i: wb.mirror.get_vector(i) for i in ids}

    def one(i: str) -> Optional[float]:
        wb.store.delete(i)
        return wb.store.wait_gone(i, vecs[i], timeout=wb.vis_timeout)

    times: list[Optional[float]] = []
    if ids:
        with ThreadPoolExecutor(max_workers=min(8, len(ids))) as ex:
            times = list(ex.map(one, ids))
    for i in ids:
        wb.mirror.delete(i)
        wb.records.pop(i, None)
    ok = [t for t in times if t is not None]
    metrics = {
        "deleted_cases": len(ids),
        "deleted_leftover": sum(1 for t in times if t is None),
        "time_to_gone_p50_s": r4(percentile(ok, 50)),
        "time_to_gone_p95_s": r4(percentile(ok, 95)),
    }
    return _finish("deleted", t0, wb, metrics, [("deleted_leftover", "<=", "deleted_leftover")])


# -------------------------------------------------------------------------------------------------
MUTATING = ("new", "updated", "deleted")
CANONICAL = ("recall", "existing", "new", "updated", "deleted")


def plan(names: list[str]) -> list[str]:
    unknown = [n for n in names if n not in CANONICAL]
    if unknown:
        raise ValueError(f"unknown scenarios {unknown}; choose from {list(CANONICAL)}")
    seq: list[str] = []
    if "recall" in names:
        seq.append("recall_baseline")
    seq += [n for n in CANONICAL[1:] if n in names]
    if "recall" in names and any(n in names for n in MUTATING):
        seq.append("recall_after_changes")
    return seq


def run_scenarios(
    wb: Workbench,
    names: list[str],
    new_records: list[dict],
    update_records: list[dict],
    log: Callable[[str], None] = print,
) -> list[dict]:
    results = []
    for step in plan(names):
        log(f"-> {step} ...")
        try:
            if step.startswith("recall"):
                res = scenario_recall(wb, label=step)
            elif step == "existing":
                res = scenario_existing(wb)
            elif step == "new":
                res = scenario_new(wb, new_records)
            elif step == "updated":
                res = scenario_updated(wb, update_records)
            else:
                res = scenario_deleted(wb)
        except Exception as exc:  # keep going so one broken scenario doesn't hide the others
            res = {
                "name": step,
                "seconds": 0,
                "metrics": {},
                "checks": [],
                "passed": False,
                "error": f"{type(exc).__name__}: {exc}",
                "traceback": traceback.format_exc(limit=6),
            }
        results.append(res)
    return results
