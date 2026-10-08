"""Concurrent load test with a stepped ramp (clients per stage), optional concurrent writers, and a saturation verdict.

Query vectors are embedded BEFORE the test starts (and cached), so Bedrock is never in the measured path: the numbers
you get are OpenSearch / Serverless latency and throughput only.
"""
from __future__ import annotations

import random
import threading
import time
from concurrent.futures import ThreadPoolExecutor
from typing import Callable, Optional, Sequence

import numpy as np

from .metrics import r4, summarize_latency
from .store import Doc, VectorStore


def make_query_vectors(wb, n: int) -> np.ndarray:
    ids = sorted(wb.mirror.ids())
    chosen = wb.rng.sample(ids, n) if n <= len(ids) else wb.rng.choices(ids, k=n)
    texts = [wb.perturbed(wb.records[i]["text"]) for i in chosen]
    return wb.embedder.embed_many(texts)


def _is_throttle(exc: Exception) -> bool:
    if getattr(exc, "status_code", None) == 429:
        return True
    s = str(exc)[:300]
    return "429" in s or "TooManyRequests" in s or "Throttl" in s


def _query_worker(store: VectorStore, vectors, k: int, batch: int, measure_from: float, end: float, seed: int) -> dict:
    rng = random.Random(seed)
    n = len(vectors)
    out = {"lat": [], "took": [], "errors": 0, "throttled": 0, "queries": 0, "requests": 0}
    while time.perf_counter() < end:
        t0 = time.perf_counter()
        try:
            if batch > 1:
                store.msearch([vectors[rng.randrange(n)] for _ in range(batch)], k)
                took, nq = None, batch
            else:
                _, took = store.search_with_took(vectors[rng.randrange(n)], k)
                nq = 1
            ms = (time.perf_counter() - t0) * 1000
            if t0 >= measure_from:
                out["lat"].append(ms)
                out["queries"] += nq
                out["requests"] += 1
                if took is not None:
                    out["took"].append(took)
        except Exception as exc:
            if t0 >= measure_from:
                out["errors"] += 1
                if _is_throttle(exc):
                    out["throttled"] += 1
            time.sleep(0.01)
    return out


def _writer_worker(
    store: VectorStore, docs: Sequence[Doc], versions: dict, vlock: threading.Lock, measure_from: float, end: float, seed: int
) -> dict:
    rng = random.Random(seed)
    out = {"lat": [], "errors": 0, "writes": 0}
    while time.perf_counter() < end:
        d = rng.choice(docs)
        with vlock:
            versions[d.id] = versions.get(d.id, d.version) + 1
            ver = versions[d.id]
        t0 = time.perf_counter()
        try:
            store.upsert(Doc(d.id, d.text, d.vector, ver, d.meta))
            if t0 >= measure_from:
                out["lat"].append((time.perf_counter() - t0) * 1000)
                out["writes"] += 1
        except Exception:
            if t0 >= measure_from:
                out["errors"] += 1
            time.sleep(0.01)
    return out


def run_stage(
    store: VectorStore,
    vectors,
    clients: int,
    duration: float,
    warmup: float = 5.0,
    k: int = 10,
    batch: int = 1,
    writer_docs: Optional[Sequence[Doc]] = None,
    writers: int = 0,
    seed: int = 1,
) -> dict:
    start = time.perf_counter()
    measure_from = start + warmup
    end = measure_from + duration
    versions: dict = {}
    vlock = threading.Lock()
    with ThreadPoolExecutor(max_workers=clients + writers) as ex:
        qf = [ex.submit(_query_worker, store, vectors, k, batch, measure_from, end, seed * 1000 + i) for i in range(clients)]
        wf = (
            [ex.submit(_writer_worker, store, writer_docs, versions, vlock, measure_from, end, seed * 7000 + i) for i in range(writers)]
            if writers and writer_docs
            else []
        )
        q = [f.result() for f in qf]
        w = [f.result() for f in wf]

    lat = [x for r in q for x in r["lat"]]
    took = [x for r in q for x in r["took"]]
    errors = sum(r["errors"] for r in q)
    throttled = sum(r["throttled"] for r in q)
    requests = sum(r["requests"] for r in q)
    queries = sum(r["queries"] for r in q)
    attempted = requests + errors
    stage = {
        "clients": clients,
        "batch": batch,
        "duration_s": duration,
        "requests": requests,
        "queries": queries,
        "qps": round(queries / duration, 1) if duration else None,
        "errors": errors,
        "throttled_429": throttled,
        "error_rate": r4(errors / attempted) if attempted else None,
        "latency_ms": summarize_latency(lat),
        "server_took_ms": summarize_latency(took) if took else None,
    }
    if wf:
        wl = [x for r in w for x in r["lat"]]
        stage["writers"] = {
            "count": writers,
            "writes": sum(r["writes"] for r in w),
            "writes_per_sec": round(sum(r["writes"] for r in w) / duration, 1) if duration else None,
            "errors": sum(r["errors"] for r in w),
            "latency_ms": summarize_latency(wl),
        }
    return stage


def run_load(
    store: VectorStore,
    vectors,
    stages: Sequence[int],
    duration: float,
    warmup: float,
    k: int,
    batch: int,
    thresholds: dict,
    writer_docs: Optional[Sequence[Doc]] = None,
    writers: int = 0,
    pause: float = 3.0,
    log: Callable[[str], None] = print,
) -> dict:
    max_p99 = thresholds.get("load_max_p99_ms")
    max_err = thresholds.get("load_max_error_rate")
    out_stages, first_bad, last_ok = [], None, None
    for idx, c in enumerate(stages):
        log(f"-> stage {idx + 1}/{len(stages)}: {c} clients x {duration}s (+{warmup}s warm-up)")
        st = run_stage(store, vectors, c, duration, warmup, k, batch, writer_docs, writers, seed=idx + 1)
        p99 = st["latency_ms"].get("p99")
        ok = True
        if max_err is not None and (st["error_rate"] or 0) > max_err:
            ok = False
        if max_p99 is not None and (p99 is None or p99 > max_p99):
            ok = False
        st["ok"] = ok
        out_stages.append(st)
        if ok and first_bad is None:
            last_ok = c
        if not ok and first_bad is None:
            first_bad = c
        log(
            f"   qps={st['qps']}  p50={st['latency_ms'].get('p50')}ms  p99={p99}ms  "
            f"errors={st['errors']} (429s={st['throttled_429']})  {'ok' if ok else 'DEGRADED'}"
        )
        if pause and idx < len(stages) - 1:
            time.sleep(pause)
    return {
        "stages": out_stages,
        "verdict": {
            "max_p99_ms": max_p99,
            "max_error_rate": max_err,
            "last_ok_clients": last_ok,
            "first_degraded_clients": first_bad,
            "peak_qps": max((s["qps"] or 0) for s in out_stages) if out_stages else None,
        },
    }
