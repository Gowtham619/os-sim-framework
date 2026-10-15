"""Embedding backends.

* TitanEmbedder  - Amazon Bedrock Titan Text Embeddings (v2 by default), with retry/backoff on throttling.
* HashEmbedder   - offline, deterministic hashed bag-of-words. Not semantic, but similar token overlap gives
                   similar vectors, so it is good enough to exercise the framework without AWS access.

Both go through an on-disk SQLite cache keyed by (model, dimensions, normalize, text) so the same text is
never embedded (or billed) twice across runs.
"""
from __future__ import annotations

import hashlib
import json
import math
import os
import random
import re
import sqlite3
import sys
import threading
import time
from collections import Counter
from concurrent.futures import ThreadPoolExecutor
from typing import Callable, Sequence

import numpy as np

from .metrics import summarize_latency


class EmbeddingCache:
    def __init__(self, path: str | None):
        self.path = path
        if path:
            os.makedirs(os.path.dirname(os.path.abspath(path)), exist_ok=True)
        self.conn = sqlite3.connect(path or ":memory:", check_same_thread=False)
        self.lock = threading.Lock()
        with self.lock:
            self.conn.execute("PRAGMA journal_mode=WAL")
            self.conn.execute("PRAGMA synchronous=OFF")
            self.conn.execute("CREATE TABLE IF NOT EXISTS emb (k TEXT PRIMARY KEY, v BLOB)")
            self.conn.commit()
        self._pending = 0

    def get(self, key: str) -> np.ndarray | None:
        with self.lock:
            row = self.conn.execute("SELECT v FROM emb WHERE k=?", (key,)).fetchone()
        return None if row is None else np.frombuffer(row[0], dtype=np.float32).copy()

    def put(self, key: str, vec: np.ndarray) -> None:
        with self.lock:
            self.conn.execute(
                "INSERT OR REPLACE INTO emb (k, v) VALUES (?, ?)", (key, np.asarray(vec, dtype=np.float32).tobytes())
            )
            self._pending += 1
            if self._pending >= 200:
                self.conn.commit()
                self._pending = 0

    def flush(self) -> None:
        with self.lock:
            self.conn.commit()
            self._pending = 0


class BaseEmbedder:
    cache_tag = "base"
    dim = 0

    def __init__(self, cache: EmbeddingCache | None, workers: int = 8):
        self.cache = cache or EmbeddingCache(None)
        self.workers = workers
        self._lock = threading.Lock()
        self.stats = {"calls": 0, "cache_hits": 0, "throttles": 0, "retries": 0, "truncated": 0}
        self.latencies_ms: list[float] = []

    # -- subclass hook -------------------------------------------------------------------------
    def _embed_uncached(self, text: str) -> np.ndarray:  # pragma: no cover - abstract
        raise NotImplementedError

    # -- public API ----------------------------------------------------------------------------
    def _key(self, text: str) -> str:
        return hashlib.sha256(f"{self.cache_tag}|{text}".encode("utf-8")).hexdigest()

    def _bump(self, name: str, n: int = 1) -> None:
        with self._lock:
            self.stats[name] += n

    def embed(self, text: str, use_cache: bool = True) -> np.ndarray:
        key = self._key(text)
        if use_cache:
            cached = self.cache.get(key)
            if cached is not None:
                self._bump("cache_hits")
                return cached
        t0 = time.perf_counter()
        vec = self._embed_uncached(text)
        ms = (time.perf_counter() - t0) * 1000
        with self._lock:
            self.latencies_ms.append(ms)
        self._bump("calls")
        if use_cache:
            self.cache.put(key, vec)
        return vec

    def embed_many(
        self, texts: Sequence[str], workers: int | None = None, use_cache: bool = True, progress: bool = False
    ) -> np.ndarray:
        uniq = list(dict.fromkeys(texts))
        done = [0]
        total = len(uniq)

        def one(t: str) -> np.ndarray:
            v = self.embed(t, use_cache=use_cache)
            if progress:
                with self._lock:
                    done[0] += 1
                    if done[0] % max(1, total // 20) == 0 or done[0] == total:
                        print(f"  embedded {done[0]}/{total}", file=sys.stderr)
            return v

        with ThreadPoolExecutor(max_workers=workers or self.workers) as ex:
            vecs = list(ex.map(one, uniq))
        self.cache.flush()
        lookup = dict(zip(uniq, vecs))
        if not texts:
            return np.zeros((0, self.dim), dtype=np.float32)
        return np.stack([lookup[t] for t in texts]).astype(np.float32)

    def summary(self) -> dict:
        out = dict(self.stats)
        out["latency_ms"] = summarize_latency(self.latencies_ms)
        return out


class HashEmbedder(BaseEmbedder):
    """Deterministic hashed bag of words + bigrams. Offline stand-in for a real model."""

    def __init__(self, dim: int = 256, cache: EmbeddingCache | None = None, workers: int = 4):
        super().__init__(cache, workers)
        self.dim = dim
        self.model_id = f"hash-{dim}"
        self.cache_tag = f"hash|{dim}"

    @staticmethod
    def _tokens(text: str) -> list[str]:
        words = re.findall(r"[a-z0-9]+(?:[-_.][a-z0-9]+)*", text.lower())
        return words + [f"{a} {b}" for a, b in zip(words, words[1:])]

    def _embed_uncached(self, text: str) -> np.ndarray:
        vec = np.zeros(self.dim, dtype=np.float32)
        for tok, count in Counter(self._tokens(text)).items():
            h = hashlib.blake2b(tok.encode("utf-8"), digest_size=8).digest()
            idx = int.from_bytes(h[:4], "little") % self.dim
            sign = 1.0 if h[4] & 1 else -1.0
            vec[idx] += sign * (1.0 + math.log(count))
        norm = float(np.linalg.norm(vec))
        return vec / norm if norm > 0 else vec


class TitanEmbedder(BaseEmbedder):
    RETRYABLE = {
        "ThrottlingException",
        "TooManyRequestsException",
        "ModelTimeoutException",
        "ModelNotReadyException",
        "ServiceUnavailableException",
        "InternalServerException",
    }
    # SAML/SSO credentials are temporary; on these codes rebuild the client so fresh credentials are picked up
    EXPIRED = {"ExpiredTokenException", "ExpiredToken", "UnrecognizedClientException", "RequestExpired"}

    def __init__(
        self,
        model_id: str = "amazon.titan-embed-text-v2:0",
        region: str = "us-east-1",
        dimensions: int = 1024,
        normalize: bool = True,
        workers: int = 8,
        cache: EmbeddingCache | None = None,
        max_chars: int = 40000,
        profile: str | None = None,
        client=None,
        max_retries: int = 8,
        sleep: Callable[[float], None] = time.sleep,
    ):
        super().__init__(cache, workers)
        self.model_id = model_id
        self.is_v2 = "titan-embed-text-v2" in model_id
        if self.is_v2 and dimensions not in (256, 512, 1024):
            raise ValueError("Titan Text Embeddings v2 supports dimensions 256, 512 or 1024")
        self.dim = dimensions if self.is_v2 else 1536
        self.normalize = normalize
        self.max_chars = max_chars
        self.max_retries = max_retries
        self._sleep = sleep
        self.cache_tag = f"titan|{model_id}|{self.dim}|{normalize}"
        self._profile, self._region, self._workers = profile, region, workers
        self._client_lock = threading.Lock()
        self._can_rebuild = client is None
        self.client = client if client is not None else self._build_client()

    def _build_client(self):
        import boto3
        from botocore.config import Config

        session = boto3.Session(profile_name=self._profile) if self._profile else boto3.Session()
        return session.client(
            "bedrock-runtime",
            region_name=self._region,
            config=Config(
                max_pool_connections=max(10, self._workers * 2),
                read_timeout=60,
                connect_timeout=10,
                retries={"max_attempts": 1, "mode": "standard"},
            ),
        )

    def _rebuild_client(self, stale) -> None:
        with self._client_lock:
            if self.client is stale:  # another thread may already have refreshed it
                self.client = self._build_client()

    def _request_body(self, text: str) -> dict:
        body: dict = {"inputText": text}
        if self.is_v2:
            body["dimensions"] = self.dim
            body["normalize"] = self.normalize
        return body

    def _embed_uncached(self, text: str) -> np.ndarray:
        from botocore.exceptions import BotoCoreError, ClientError

        text = (text or "").strip()
        if not text:
            raise ValueError("cannot embed empty text")
        if len(text) > self.max_chars:
            text = text[: self.max_chars]
            self._bump("truncated")
        body = json.dumps(self._request_body(text))
        last_err: Exception | None = None
        for attempt in range(self.max_retries + 1):
            try:
                client = self.client
                resp = client.invoke_model(
                    modelId=self.model_id, body=body, contentType="application/json", accept="application/json"
                )
                payload = json.loads(resp["body"].read())
                vec = np.asarray(payload["embedding"], dtype=np.float32)
                if vec.shape[0] != self.dim:
                    raise ValueError(f"expected {self.dim} dims from {self.model_id}, got {vec.shape[0]}")
                if self.normalize:
                    n = float(np.linalg.norm(vec))
                    if n > 0:
                        vec = vec / n
                return vec
            except ClientError as exc:
                code = exc.response.get("Error", {}).get("Code", "")
                if code in self.EXPIRED and self._can_rebuild and attempt < 2:
                    self._rebuild_client(client)  # re-reads the profile / re-runs credential_process
                    last_err = exc
                    continue
                if code not in self.RETRYABLE:
                    raise
                last_err = exc
                if code in ("ThrottlingException", "TooManyRequestsException"):
                    self._bump("throttles")
            except BotoCoreError as exc:  # connection resets, read timeouts, ...
                last_err = exc
            self._bump("retries")
            self._sleep(min(20.0, 0.5 * (2**attempt)) * (0.5 + random.random() / 2))
        raise RuntimeError(f"Bedrock embedding failed after {self.max_retries} retries: {last_err}")


def make_embedder(cfg: dict) -> BaseEmbedder:
    e = cfg["embedder"]
    cache = EmbeddingCache(e.get("cache_path")) if e.get("cache_path") else EmbeddingCache(None)
    kind = e.get("type", "titan")
    if kind == "hash":
        return HashEmbedder(dim=int(e.get("dimensions", 256)), cache=cache, workers=int(e.get("workers", 4)))
    if kind == "titan":
        return TitanEmbedder(
            model_id=e["model_id"],
            region=e["region"],
            dimensions=int(e["dimensions"]),
            normalize=bool(e.get("normalize", True)),
            workers=int(e.get("workers", 8)),
            cache=cache,
            max_chars=int(e.get("max_chars", 40000)),
            profile=e.get("profile"),
        )
    raise ValueError(f"unknown embedder type '{kind}'")
