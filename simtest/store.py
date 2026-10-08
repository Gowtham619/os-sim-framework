"""Vector stores.

* LocalStore       - in-memory exact (brute force cosine). Used as ground truth and as a no-AWS dry-run target.
* OpenSearchStore  - Amazon OpenSearch Service domain (service=es) or OpenSearch Serverless collection (service=aoss).

Managed vs Serverless differences handled here:
  - Serverless has no shard/replica/refresh settings and no _refresh API, so new data becomes searchable after an
    unspecified delay. `wait_visible` polls the search path to measure that delay ("time to searchable").
  - Vector collections on Serverless may not accept custom document ids. With id_mode=auto the case id lives in the
    `doc_id` field, and updates/deletes look the engine _id up first.
"""
from __future__ import annotations

import os
import random
import time
from concurrent.futures import ThreadPoolExecutor
from dataclasses import dataclass, field
from typing import Optional, Sequence

import numpy as np

RETRYABLE_STATUS = {429, 502, 503, 504}
RETRYABLE_EXC_NAMES = {"ConnectionTimeout", "ConnectionError", "ReadTimeout", "ReadTimeoutError"}


@dataclass
class Doc:
    id: str
    text: str
    vector: np.ndarray
    version: int = 1
    meta: dict = field(default_factory=dict)


@dataclass
class Hit:
    id: str
    score: float
    similarity: float
    version: Optional[int] = None


def _unit(v) -> np.ndarray:
    v = np.asarray(v, dtype=np.float32)
    n = float(np.linalg.norm(v))
    return v / n if n > 0 else v


def score_to_similarity(space_type: str, score: float, engine: str = "faiss") -> float:
    """Best-effort conversion of an OpenSearch k-NN score back to a cosine-like similarity.

    Ranking and all pass/fail checks use ids, never this value; it exists to make reports readable.
    innerproduct: score = ip + 1 (ip >= 0) or 1 / (1 - ip) (ip < 0)
    cosinesimil : score = 1 + cos (faiss / nmslib), (1 + cos) / 2 (lucene)
    other spaces: score is returned unchanged.
    """
    if space_type == "innerproduct":
        if score <= 0:
            return -1.0
        return score - 1.0 if score >= 1.0 else 1.0 - 1.0 / score
    if space_type == "cosinesimil":
        return 2.0 * score - 1.0 if engine == "lucene" else score - 1.0
    return float(score)


class StoreError(RuntimeError):
    pass


class VectorStore:
    kind = "base"
    name = "base"

    # -- to implement ----------------------------------------------------------------------------
    def create_index(self, recreate: bool = False) -> None: ...
    def bulk_index(self, docs: Sequence[Doc], progress: bool = False) -> dict: ...
    def upsert(self, doc: Doc) -> None: ...
    def delete(self, doc_id: str) -> None: ...
    def search(self, vector, k: int = 10) -> list[Hit]: ...
    def count(self) -> int: ...
    def refresh(self) -> None: ...

    # -- shared helpers --------------------------------------------------------------------------
    def search_with_took(self, vector, k: int = 10) -> tuple[list[Hit], Optional[float]]:
        return self.search(vector, k), None

    def msearch(self, vectors: Sequence, k: int = 10) -> list[list[Hit]]:
        return [self.search(v, k) for v in vectors]

    def wait_visible(
        self, doc_id: str, vector, timeout: float = 60.0, poll: float = 0.25, expect_version: Optional[int] = None, k: int = 5
    ) -> Optional[float]:
        """Seconds until a search for `vector` returns `doc_id` (at `expect_version`), or None on timeout."""
        t0 = time.perf_counter()
        while True:
            hits = self.search(vector, k)
            if any(h.id == doc_id and (expect_version is None or h.version == expect_version) for h in hits):
                return time.perf_counter() - t0
            if time.perf_counter() - t0 > timeout:
                return None
            time.sleep(poll)

    def wait_gone(self, doc_id: str, vector, timeout: float = 60.0, poll: float = 0.25, k: int = 10) -> Optional[float]:
        t0 = time.perf_counter()
        while True:
            hits = self.search(vector, k)
            if not any(h.id == doc_id for h in hits):
                return time.perf_counter() - t0
            if time.perf_counter() - t0 > timeout:
                return None
            time.sleep(poll)

    def describe(self) -> dict:
        return {"kind": self.kind, "name": self.name}


# =================================================================================================
class LocalStore(VectorStore):
    kind = "local"

    def __init__(self, name: str = "local", **_ignored):
        self.name = name
        self._ids: list[str] = []
        self._vecs: list[np.ndarray] = []
        self._ver: list[int] = []
        self._pos: dict[str, int] = {}
        self._mat: Optional[np.ndarray] = None

    def create_index(self, recreate: bool = False) -> None:
        if recreate:
            self._ids, self._vecs, self._ver, self._pos, self._mat = [], [], [], {}, None

    def upsert(self, doc: Doc) -> None:
        v = _unit(doc.vector)
        if doc.id in self._pos:
            i = self._pos[doc.id]
            self._vecs[i], self._ver[i] = v, doc.version
        else:
            self._pos[doc.id] = len(self._ids)
            self._ids.append(doc.id)
            self._vecs.append(v)
            self._ver.append(doc.version)
        self._mat = None

    def bulk_index(self, docs: Sequence[Doc], progress: bool = False) -> dict:
        t0 = time.perf_counter()
        for d in docs:
            self.upsert(d)
        return {"indexed": len(docs), "errors": 0, "throttled": 0, "retries": 0, "seconds": time.perf_counter() - t0}

    def delete(self, doc_id: str) -> None:
        i = self._pos.pop(doc_id, None)
        if i is None:
            return
        last = len(self._ids) - 1
        if i != last:
            self._ids[i], self._vecs[i], self._ver[i] = self._ids[last], self._vecs[last], self._ver[last]
            self._pos[self._ids[i]] = i
        self._ids.pop()
        self._vecs.pop()
        self._ver.pop()
        self._mat = None

    def get_vector(self, doc_id: str) -> Optional[np.ndarray]:
        i = self._pos.get(doc_id)
        return None if i is None else self._vecs[i]

    def get_version(self, doc_id: str) -> Optional[int]:
        i = self._pos.get(doc_id)
        return None if i is None else self._ver[i]

    def ids(self) -> list[str]:
        return list(self._ids)

    def _matrix(self) -> np.ndarray:
        if self._mat is None:
            self._mat = np.stack(self._vecs) if self._vecs else np.zeros((0, 1), dtype=np.float32)
        return self._mat

    def search(self, vector, k: int = 10) -> list[Hit]:
        if not self._ids:
            return []
        sims = self._matrix() @ _unit(vector)
        k = min(k, len(self._ids))
        top = np.argpartition(-sims, k - 1)[:k]
        top = top[np.argsort(-sims[top])]
        return [Hit(self._ids[i], float(sims[i]), float(sims[i]), self._ver[i]) for i in top]

    def count(self) -> int:
        return len(self._ids)

    def refresh(self) -> None:
        return None


# =================================================================================================
def build_client(cfg: dict):
    from opensearchpy import AWSV4SignerAuth, OpenSearch, RequestsHttpConnection

    endpoint = cfg["endpoint"]
    host = endpoint.replace("https://", "").replace("http://", "").rstrip("/")
    if cfg.get("auth", "sigv4") == "sigv4":
        import boto3

        session = boto3.Session(profile_name=cfg.get("profile")) if cfg.get("profile") else boto3.Session()
        auth = AWSV4SignerAuth(session.get_credentials(), cfg["region"], cfg.get("service", "es"))
    else:
        user, pwd = os.environ.get("OS_USER"), os.environ.get("OS_PASS")
        if not user or not pwd:
            raise StoreError("auth=basic requires OS_USER and OS_PASS environment variables")
        auth = (user, pwd)
    return OpenSearch(
        hosts=[{"host": host, "port": 443}],
        http_auth=auth,
        use_ssl=True,
        verify_certs=True,
        connection_class=RequestsHttpConnection,
        pool_maxsize=int(cfg.get("pool_maxsize", 64)),
        timeout=int(cfg.get("timeout", 60)),
        max_retries=3,
        retry_on_timeout=True,
    )


class OpenSearchStore(VectorStore):
    kind = "opensearch"

    def __init__(self, cfg: dict, dim: int, client=None):
        self.cfg = cfg
        self.name = cfg.get("name", "opensearch")
        self.dim = dim
        self.index = cfg["index"]
        self.service = cfg.get("service", "es")
        self.serverless = self.service == "aoss"
        self.id_mode = cfg.get("id_mode", "custom")
        self.space_type = cfg.get("space_type", "innerproduct")
        self.engine = cfg.get("engine", "faiss")
        self.backoff_base = float(cfg.get("backoff_base", 0.5))
        self.client = client if client is not None else build_client(cfg)

    # -- helpers ---------------------------------------------------------------------------------
    def describe(self) -> dict:
        host = str(self.cfg.get("endpoint", "")).replace("https://", "")
        return {
            "kind": self.kind,
            "name": self.name,
            "endpoint": host,
            "service": self.service,
            "index": self.index,
            "id_mode": self.id_mode,
            "engine": self.engine,
            "space_type": self.space_type,
        }

    @staticmethod
    def _status(exc: Exception):
        return getattr(exc, "status_code", None)

    def _retryable(self, exc: Exception) -> bool:
        return self._status(exc) in RETRYABLE_STATUS or type(exc).__name__ in RETRYABLE_EXC_NAMES

    def _backoff(self, attempt: int) -> None:
        if self.backoff_base > 0:
            time.sleep(min(10.0, self.backoff_base * (2**attempt)) * (0.5 + random.random() / 2))

    def _doc_body(self, doc: Doc) -> dict:
        body = {"doc_id": doc.id, "text": doc.text, "version": doc.version, "embedding": [float(x) for x in doc.vector]}
        body.update(doc.meta or {})
        return body

    # -- index management ------------------------------------------------------------------------
    def index_body(self) -> dict:
        c = self.cfg
        settings: dict = {"index": {"knn": True}}
        if c.get("ef_search"):
            settings["index"]["knn.algo_param.ef_search"] = int(c["ef_search"])
        if not self.serverless:
            settings["index"]["number_of_shards"] = int(c.get("shards", 3))
            settings["index"]["number_of_replicas"] = int(c.get("replicas", 1))
            settings["index"]["refresh_interval"] = c.get("refresh_interval", "1s")
        for k, v in (c.get("settings_override") or {}).items():
            if isinstance(v, dict) and isinstance(settings.get(k), dict):
                settings[k].update(v)
            else:
                settings[k] = v
        emb = {
            "type": "knn_vector",
            "dimension": self.dim,
            "method": {
                "name": "hnsw",
                "engine": self.engine,
                "space_type": self.space_type,
                "parameters": {"m": int(c.get("m", 16)), "ef_construction": int(c.get("ef_construction", 128))},
            },
        }
        emb.update(c.get("field_extra") or {})
        return {
            "settings": settings,
            "mappings": {
                "properties": {
                    "doc_id": {"type": "keyword"},
                    "text": {"type": "text"},
                    "version": {"type": "integer"},
                    "embedding": emb,
                }
            },
        }

    def create_index(self, recreate: bool = False) -> None:
        exists = self.client.indices.exists(index=self.index)
        if exists and recreate:
            self.client.indices.delete(index=self.index)
            exists = False
        if not exists:
            self.client.indices.create(index=self.index, body=self.index_body())
            wait = self.cfg.get("post_create_wait")
            wait = (10 if self.serverless else 0) if wait is None else wait
            if wait:
                time.sleep(float(wait))

    def refresh(self) -> None:
        if self.serverless:
            return  # _refresh is not available on Serverless
        self.client.indices.refresh(index=self.index)

    def count(self) -> int:
        try:
            return int(self.client.count(index=self.index)["count"])
        except Exception:
            resp = self.client.search(index=self.index, body={"size": 0, "track_total_hits": True})
            return int(resp["hits"]["total"]["value"])

    # -- writes ----------------------------------------------------------------------------------
    def _bulk_chunk(self, chunk: Sequence[Doc]) -> dict:
        stats = {"indexed": 0, "errors": 0, "throttled": 0, "retries": 0}
        pending = list(chunk)
        for attempt in range(7):
            lines: list[dict] = []
            for d in pending:
                action = {"_index": self.index}
                if self.id_mode == "custom":
                    action["_id"] = d.id
                lines.append({"index": action})
                lines.append(self._doc_body(d))
            try:
                resp = self.client.bulk(body=lines)
            except Exception as exc:
                if self._retryable(exc) and attempt < 6:
                    stats["retries"] += 1
                    if self._status(exc) == 429:
                        stats["throttled"] += 1
                    self._backoff(attempt)
                    continue
                raise
            items = resp.get("items", [])
            failed: list[Doc] = []
            for d, item in zip(pending, items):
                result = next(iter(item.values()))
                status = int(result.get("status", 200))
                if status < 300:
                    stats["indexed"] += 1
                elif status in RETRYABLE_STATUS:
                    failed.append(d)
                    if status == 429:
                        stats["throttled"] += 1
                else:
                    stats["errors"] += 1
            if not failed:
                return stats
            pending = failed
            stats["retries"] += 1
            self._backoff(attempt)
        stats["errors"] += len(pending)
        return stats

    def bulk_index(self, docs: Sequence[Doc], progress: bool = False) -> dict:
        size = int(self.cfg.get("bulk_size", 500))
        workers = int(self.cfg.get("bulk_workers", 8))
        chunks = [docs[i : i + size] for i in range(0, len(docs), size)]
        t0 = time.perf_counter()
        total = {"indexed": 0, "errors": 0, "throttled": 0, "retries": 0}
        if chunks:
            with ThreadPoolExecutor(max_workers=workers) as ex:
                for n, res in enumerate(ex.map(self._bulk_chunk, chunks), 1):
                    for key in total:
                        total[key] += res[key]
                    if progress and (n % max(1, len(chunks) // 10) == 0 or n == len(chunks)):
                        print(f"  bulk {n}/{len(chunks)} chunks, {total['indexed']} docs", flush=True)
        total["seconds"] = time.perf_counter() - t0
        return total

    def _lookup_ids(self, doc_id: str) -> list[str]:
        resp = self.client.search(
            index=self.index, body={"size": 10, "_source": False, "query": {"term": {"doc_id": doc_id}}}
        )
        return [h["_id"] for h in resp["hits"]["hits"]]

    def _delete_by_engine_id(self, _id: str) -> None:
        try:
            self.client.delete(index=self.index, id=_id)
        except Exception as exc:
            if self._status(exc) != 404:
                raise

    def upsert(self, doc: Doc) -> None:
        if self.id_mode == "custom":
            self.client.index(index=self.index, id=doc.id, body=self._doc_body(doc))
            return
        for _id in self._lookup_ids(doc.id):
            self._delete_by_engine_id(_id)
        self.client.index(index=self.index, body=self._doc_body(doc))

    def delete(self, doc_id: str) -> None:
        if self.id_mode == "custom":
            self._delete_by_engine_id(doc_id)
            return
        for _id in self._lookup_ids(doc_id):
            self._delete_by_engine_id(_id)

    # -- reads -----------------------------------------------------------------------------------
    def _search_body(self, vector, k: int) -> dict:
        knn: dict = {"vector": [float(x) for x in vector], "k": k}
        if self.cfg.get("query_ef_search"):
            knn["method_parameters"] = {"ef_search": int(self.cfg["query_ef_search"])}
        return {"size": k, "_source": ["doc_id", "version"], "query": {"knn": {"embedding": knn}}}

    def _parse_hits(self, resp: dict) -> list[Hit]:
        hits = []
        for h in resp["hits"]["hits"]:
            src = h.get("_source") or {}
            score = float(h["_score"])
            hits.append(
                Hit(
                    id=src.get("doc_id", h["_id"]),
                    score=score,
                    similarity=score_to_similarity(self.space_type, score, self.engine),
                    version=src.get("version"),
                )
            )
        return hits

    def search_with_took(self, vector, k: int = 10) -> tuple[list[Hit], Optional[float]]:
        resp = self.client.search(index=self.index, body=self._search_body(vector, k))
        took = resp.get("took")
        return self._parse_hits(resp), (float(took) if took is not None else None)

    def search(self, vector, k: int = 10) -> list[Hit]:
        return self.search_with_took(vector, k)[0]

    def msearch(self, vectors: Sequence, k: int = 10) -> list[list[Hit]]:
        results: list[Optional[list[Hit]]] = [None] * len(vectors)
        pending = list(range(len(vectors)))
        for attempt in range(7):
            lines: list[dict] = []
            for i in pending:
                lines += [{"index": self.index}, self._search_body(vectors[i], k)]
            try:
                resp = self.client.msearch(body=lines)
            except Exception as exc:
                if self._retryable(exc) and attempt < 6:
                    self._backoff(attempt)
                    continue
                raise
            failed = []
            for i, r in zip(pending, resp["responses"]):
                if "error" in r:
                    if int(r.get("status", 500)) in RETRYABLE_STATUS:
                        failed.append(i)
                        continue
                    raise StoreError(f"msearch error: {r['error']}")
                results[i] = self._parse_hits(r)
            if not failed:
                return results  # type: ignore[return-value]
            pending = failed
            self._backoff(attempt)
        raise StoreError(f"msearch: {len(pending)} queries still failing after retries")


def make_store(target_cfg: dict, dim: int, client=None) -> VectorStore:
    kind = target_cfg.get("kind", "opensearch")
    if kind == "local":
        return LocalStore(name=target_cfg.get("name", "local"))
    if kind == "opensearch":
        if not target_cfg.get("endpoint") and client is None:
            raise StoreError(f"target '{target_cfg.get('name')}' has no endpoint configured")
        return OpenSearchStore(target_cfg, dim, client=client)
    raise StoreError(f"unknown target kind '{kind}'")
