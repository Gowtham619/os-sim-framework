"""Configuration loading: YAML file + defaults + ${ENV_VAR} expansion."""
from __future__ import annotations

import copy
import os
from typing import Any

import yaml

DEFAULTS: dict[str, Any] = {
    "embedder": {
        "type": "titan",  # titan | hash (hash = offline, deterministic, no AWS needed)
        "model_id": "amazon.titan-embed-text-v2:0",
        "region": "us-east-1",
        "profile": None,
        "dimensions": 1024,  # Titan v2: 256 | 512 | 1024. Titan v1 is fixed at 1536.
        "normalize": True,
        "workers": 8,
        "max_chars": 40000,
        "cache_path": ".cache/embeddings.sqlite",
    },
    "target_defaults": {
        "kind": "opensearch",  # opensearch | local
        "service": "es",  # es = managed domain, aoss = OpenSearch Serverless
        "auth": "sigv4",  # sigv4 | basic (basic reads OS_USER / OS_PASS from the environment)
        "profile": None,
        "index": "sim-cases",
        "shards": 3,  # ignored on Serverless
        "replicas": 1,  # ignored on Serverless
        "refresh_interval": "1s",  # ignored on Serverless
        "id_mode": "custom",  # custom = use case id as _id; auto = engine-assigned _id (Serverless vector collections)
        "engine": "faiss",
        "space_type": "innerproduct",
        "m": 16,
        "ef_construction": 128,
        "ef_search": 100,  # set to null to omit the index-level setting
        "query_ef_search": None,  # per-query override (OpenSearch >= 2.16)
        "field_extra": {},  # extra keys merged into the knn_vector mapping (e.g. {"data_type": "byte"})
        "settings_override": {},  # merged into index settings
        "bulk_size": 500,
        "bulk_workers": 8,
        "msearch_batch": 50,
        "timeout": 60,
        "pool_maxsize": 64,
        "post_create_wait": None,  # seconds to wait after index creation (default: 10 on aoss, 0 otherwise)
        "backoff_base": 0.5,
    },
    "targets": {},
    "data": {
        "corpus": "data/corpus.jsonl",
        "new_cases": "data/new_cases.jsonl",
        "updates": "data/updates.jsonl",
        "queries": "data/dataset_a.jsonl",
    },
    "run": {
        "k": 10,
        "n_queries": 200,
        "perturb_drop": 0.15,
        "seed": 1,
        "query_workers": 8,
        "visibility_timeout": 60,
        "visibility_sample": 20,
        "update_sample": None,
        "delete_sample": 20,
    },
    "thresholds": {
        "recall_at_k": 0.90,
        "existing_exact_hit_at_1": 0.99,
        "existing_perturbed_hit_at_k": 0.95,
        "new_post_index_hit_at_1": 0.99,
        "new_neighbor_hit_at_k": 0.80,
        "updated_hit_at_1": 0.99,
        "updated_duplicates": 0,
        "updated_stale_versions": 0,
        "updated_old_vector_served": 0,
        "deleted_leftover": 0,
        "visibility_timeouts": 0,
        "time_to_searchable_p95_s": 30,
        "load_max_p99_ms": 500,
        "load_max_error_rate": 0.01,
    },
}


def deep_merge(base: dict, override: dict | None) -> dict:
    out = copy.deepcopy(base)
    for key, value in (override or {}).items():
        if isinstance(value, dict) and isinstance(out.get(key), dict):
            out[key] = deep_merge(out[key], value)
        else:
            out[key] = copy.deepcopy(value)
    return out


def _expand(obj: Any) -> Any:
    if isinstance(obj, str):
        return os.path.expandvars(obj)
    if isinstance(obj, dict):
        return {k: _expand(v) for k, v in obj.items()}
    if isinstance(obj, list):
        return [_expand(v) for v in obj]
    return obj


def load_config(path: str | None) -> dict:
    raw: dict = {}
    if path:
        with open(path, "r", encoding="utf-8") as fh:
            raw = yaml.safe_load(fh) or {}
    return deep_merge(DEFAULTS, _expand(raw))


def get_target(cfg: dict, name: str) -> dict:
    targets = cfg.get("targets") or {}
    if name not in targets:
        known = ", ".join(sorted(targets)) or "(none defined)"
        raise KeyError(f"target '{name}' not found in config. Known targets: {known}")
    merged = deep_merge(cfg["target_defaults"], targets[name])
    merged["name"] = name
    return merged
