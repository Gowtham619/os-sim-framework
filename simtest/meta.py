"""Run metadata saved in every result file: label, a masked snapshot of the effective config, ingest and cost figures."""
from __future__ import annotations

import json
import os
from datetime import datetime
from typing import Any, Optional

_SECRET_HINTS = ("pass", "secret", "token", "key", "cert", "header", "credential")


def _mask(obj: Any) -> Any:
    if isinstance(obj, dict):
        return {k: ("***" if v and any(h in k.lower() for h in _SECRET_HINTS) else _mask(v)) for k, v in obj.items()}
    if isinstance(obj, list):
        return [_mask(v) for v in obj]
    return obj


def config_snapshot(cfg: dict, tcfg: dict) -> dict:
    """The settings that decide how a run behaves, with anything secret-looking masked."""
    target = {k: v for k, v in tcfg.items() if k != "name"}
    return _mask({"embedder": cfg["embedder"], "target": target, "run": cfg["run"], "thresholds": cfg["thresholds"]})


def build_meta(cfg: dict, tcfg: dict, wb, target: str, label: Optional[str] = None, **extra) -> dict:
    ingest = getattr(wb, "last_ingest", None)
    meta = {
        "timestamp": datetime.now().isoformat(timespec="seconds"),
        "target": target,
        "label": label,
        "target_info": wb.store.describe(),
        "index": wb.store.describe().get("index"),
        "embedder": wb.embedder.model_id,
        "dim": wb.embedder.dim,
        "corpus_size": len(wb.records),
        "k": wb.k,
        "embedder_stats": wb.embedder.summary(),
        "ingest": (
            {key: ingest.get(key) for key in ("indexed", "errors", "throttled", "retries", "seconds", "docs_per_sec")}
            if ingest
            else None
        ),
        "config": config_snapshot(cfg, tcfg),
    }
    meta.update(extra)
    return meta


def timestamp() -> str:
    return datetime.now().strftime("%Y%m%d-%H%M%S")


def save_result(out_dir: str, target: str, kind: str, doc: dict) -> str:
    os.makedirs(out_dir, exist_ok=True)
    ts = timestamp()
    path = os.path.join(out_dir, f"{ts}_{target}_{kind}.json")
    n = 1
    while os.path.exists(path):  # never overwrite an earlier result from the same second
        n += 1
        path = os.path.join(out_dir, f"{ts}-{n}_{target}_{kind}.json")
    with open(path, "w", encoding="utf-8") as fh:
        json.dump(doc, fh, indent=2, default=str)
    return path
