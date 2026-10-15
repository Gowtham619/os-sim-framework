"""Parameter sweeps: run the same scenarios (and optionally a load test) over a grid of settings.

    python -m simtest sweep -t managed --vary m=16,32 --vary ef_search=50,200 --scenarios recall,existing

Every grid point is saved as its own labelled result, then one comparison report is built from all of them.
A setting that only affects queries (`query_ef_search`, `run.*`) reuses the index; anything that changes the index or
the vectors (m, ef_construction, space_type, shards, embedder.dimensions, ...) recreates it. Scenarios that modify the
index (new, updated, deleted) force a rebuild at every point.
"""
from __future__ import annotations

import copy
import itertools
import json
import os
from typing import Any, Callable

import yaml

from .config import get_target
from .data import read_jsonl
from .embedder import make_embedder
from .loadtest import make_query_vectors, run_load
from .meta import build_meta, save_result
from .multireport import write_report
from .scenarios import CANONICAL, MUTATING, Workbench, run_scenarios
from .store import make_store

# Settings that change only how queries are made, so the existing index stays valid.
QUERY_TIME_KEYS = {"query_ef_search", "msearch_batch", "timeout", "pool_maxsize", "backoff_base", "embedder.workers"}


def parse_vary(specs: list[str]) -> list[tuple[str, list[Any]]]:
    """['m=16,32', 'embedder.dimensions=256,1024'] -> [('m', [16, 32]), ('embedder.dimensions', [256, 1024])]"""
    out = []
    for spec in specs:
        if "=" not in spec:
            raise ValueError(f"--vary expects key=v1,v2 (got '{spec}')")
        key, raw = spec.split("=", 1)
        values = [yaml.safe_load(v.strip()) for v in raw.split(",") if v.strip() != ""]
        if not key.strip() or not values:
            raise ValueError(f"--vary needs a key and at least one value (got '{spec}')")
        out.append((key.strip(), values))
    if len({k for k, _ in out}) != len(out):
        raise ValueError("the same setting appears twice in --vary")
    return out


def apply_overrides(base_cfg: dict, target: str, combo: dict) -> dict:
    cfg = copy.deepcopy(base_cfg)
    for key, val in combo.items():
        if key.startswith("embedder."):
            cfg["embedder"][key.split(".", 1)[1]] = val
        elif key.startswith("run."):
            cfg["run"][key.split(".", 1)[1]] = val
        else:
            cfg.setdefault("targets", {}).setdefault(target, {})[key] = val
    return cfg


def _query_time_only(key: str) -> bool:
    return key in QUERY_TIME_KEYS or key.startswith("run.")


def plan(vary: list[tuple[str, list[Any]]], names: list[str], prefix: str = "") -> list[dict]:
    keys = [k for k, _ in vary]
    mutating = any(n in names for n in MUTATING)
    points: list[dict] = []
    for vals in itertools.product(*[v for _, v in vary]):
        combo = dict(zip(keys, vals))
        label = (prefix + " " if prefix else "") + ",".join(f"{k}={v}" for k, v in combo.items())
        prev = points[-1]["combo"] if points else None
        changed = prev is None or any(combo[k] != prev[k] and not _query_time_only(k) for k in keys)
        points.append({"label": label, "combo": combo, "reingest": bool(changed or mutating)})
    return points


def run_sweep(
    base_cfg: dict,
    target: str,
    vary: list[tuple[str, list[Any]]],
    names: list[str],
    out_dir: str,
    limit: int | None = None,
    label_prefix: str = "",
    load: dict | None = None,
    redact: bool = False,
    log: Callable[[str], None] = print,
) -> dict:
    unknown = [n for n in names if n not in CANONICAL]
    if unknown:
        raise ValueError(f"unknown scenarios {unknown}; choose from {list(CANONICAL)}")
    points = plan(vary, names, label_prefix)
    get_target(base_cfg, target)  # fail early on an unknown target
    saved: list[str] = []
    for i, pt in enumerate(points, 1):
        log(f"\n=== [{i}/{len(points)}] {pt['label']}  ({'rebuild index' if pt['reingest'] else 'reuse index'}) ===")
        cfg = apply_overrides(base_cfg, target, pt["combo"])
        tcfg = get_target(cfg, target)
        if tcfg.get("read_only"):
            raise ValueError(f"target '{target}' is read_only; a sweep rebuilds indexes. Use a copy of the index.")
        embedder = make_embedder(cfg)
        store = make_store(tcfg, embedder.dim)
        wb = Workbench(cfg, store, embedder)
        wb.load_corpus(read_jsonl(cfg["data"]["corpus"], limit=limit), progress=False)
        reingest = pt["reingest"]
        if not reingest:
            try:
                reingest = store.count() != wb.mirror.count()
            except Exception:
                reingest = True
        if reingest:
            stats = wb.ingest(recreate=True, progress=False)
            log(f"  indexed {stats['indexed']} docs in {round(stats['seconds'], 1)}s, errors={stats['errors']}")
        new_recs = read_jsonl(cfg["data"]["new_cases"]) if "new" in names else []
        upd_recs = read_jsonl(cfg["data"]["updates"]) if "updated" in names else []
        start_size = len(wb.records)
        results = run_scenarios(wb, names, new_recs, upd_recs, log=log)
        extra = {"sweep": {"combo": pt["combo"], "index": i, "of": len(points), "reingested": bool(reingest)}}
        meta = build_meta(cfg, tcfg, wb, target, pt["label"], corpus_size=start_size, **extra)
        passed = all(r["passed"] for r in results)
        saved.append(save_result(out_dir, target, "run", {"meta": meta, "scenarios": results, "passed": passed}))
        log(f"  scenarios: {'PASS' if passed else 'FAIL'}")
        if load:
            vectors = make_query_vectors(wb, load["queries"])
            res = run_load(
                wb.store, vectors, load["stages"], load["duration"], load["warmup"], wb.k, load.get("batch", 1),
                cfg["thresholds"], writer_docs=None, writers=0, pause=load["pause"], log=log,
            )
            lmeta = build_meta(cfg, tcfg, wb, target, pt["label"], corpus_size=start_size, **extra)
            saved.append(save_result(out_dir, target, "load", {"meta": lmeta, "load": res}))
            v = res["verdict"]
            log(f"  load: peak {v['peak_qps']} qps, last healthy stage {v['last_ok_clients']} clients")
    with open(os.path.join(out_dir, "sweep.json"), "w", encoding="utf-8") as fh:
        json.dump({"target": target, "points": points, "files": saved, "scenarios": names}, fh, indent=2, default=str)
    written, _ = write_report(saved, os.path.join(out_dir, "sweep_report"), "both", redact=redact, title=f"Sweep on {target}")
    return {"points": points, "files": saved, "reports": written}
