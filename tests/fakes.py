"""In-memory fakes for Bedrock and OpenSearch so the framework logic can be tested without AWS."""
from __future__ import annotations

import io
import json

import numpy as np
from botocore.exceptions import ClientError

from simtest.embedder import HashEmbedder


class FakeBedrock:
    """Mimics bedrock-runtime.invoke_model for Titan embeddings."""

    def __init__(self, fail_first: int = 0, code: str = "ThrottlingException"):
        self.calls: list[dict] = []
        self.fail = fail_first
        self.code = code
        self._h = None

    def invoke_model(self, modelId, body, contentType, accept):
        data = json.loads(body)
        self.calls.append({"modelId": modelId, "body": data})
        if self.fail > 0:
            self.fail -= 1
            raise ClientError({"Error": {"Code": self.code, "Message": "simulated"}}, "InvokeModel")
        dim = data.get("dimensions", 1536)
        if self._h is None or self._h.dim != dim:
            self._h = HashEmbedder(dim=dim)
        vec = self._h._embed_uncached(data["inputText"])
        return {"body": io.BytesIO(json.dumps({"embedding": vec.tolist(), "inputTextTokenCount": 5}).encode())}


class NotFound(Exception):
    status_code = 404


class FakeIndices:
    def __init__(self, c: "FakeOS"):
        self.c = c

    def exists(self, index):
        return self.c.index_exists

    def create(self, index, body):
        self.c.created_body = body
        self.c.index_exists = True
        self.c.docs = {}

    def delete(self, index):
        self.c.index_exists = False
        self.c.docs = {}

    def refresh(self, index):
        self.c.refreshes += 1


class FakeOS:
    """Emulates the slice of the OpenSearch API the framework uses (knn + term search, bulk, index, delete, msearch).

    lag:                 a document becomes searchable only after this many search calls (Serverless-like delay)
    throttle_first_bulk: first bulk call returns 429 for its first item (exercises retry logic)
    broken_term_lookup:  term queries return nothing, so id_mode=auto upserts cannot find the old copy to delete
    """

    def __init__(self, lag: int = 0, throttle_first_bulk: bool = False, broken_term_lookup: bool = False):
        self.docs: dict[str, dict] = {}
        self.born: dict[str, int] = {}
        self.index_exists = False
        self.created_body = None
        self.refreshes = 0
        self.search_calls = 0
        self.bulk_calls = 0
        self.indices = FakeIndices(self)
        self.lag = lag
        self.throttle_first_bulk = throttle_first_bulk
        self.broken_term_lookup = broken_term_lookup
        self._auto = 0
        self.last_bulk_body = None
        self.last_search_body = None

    # -- writes
    def _store(self, _id, src):
        self.docs[_id] = src
        self.born[_id] = self.search_calls

    def bulk(self, body):
        self.bulk_calls += 1
        self.last_bulk_body = body
        items = []
        for i in range(0, len(body), 2):
            action, src = body[i]["index"], body[i + 1]
            if self.throttle_first_bulk and self.bulk_calls == 1 and i == 0:
                items.append({"index": {"status": 429, "error": {"type": "es_rejected_execution_exception"}}})
                continue
            _id = action.get("_id")
            if _id is None:
                self._auto += 1
                _id = f"auto{self._auto}"
            self._store(_id, src)
            items.append({"index": {"_id": _id, "status": 201}})
        return {"errors": any(it["index"]["status"] >= 300 for it in items), "items": items}

    def index(self, index, body, id=None):
        if id is None:
            self._auto += 1
            id = f"auto{self._auto}"
        self._store(id, body)
        return {"_id": id, "result": "created"}

    def delete(self, index, id):
        if id not in self.docs:
            raise NotFound()
        del self.docs[id]
        self.born.pop(id, None)
        return {"result": "deleted"}

    def count(self, index):
        return {"count": len(self.docs)}

    # -- reads
    def _visible(self):
        return {i: s for i, s in self.docs.items() if self.search_calls - self.born[i] >= self.lag}

    def search(self, index, body):
        self.search_calls += 1
        self.last_search_body = body
        q = body["query"]
        vis = self._visible()
        if "term" in q:
            if self.broken_term_lookup:
                return {"took": 1, "hits": {"hits": []}}
            (field, val), = q["term"].items()
            hits = [{"_id": i, "_score": 1.0} for i, s in vis.items() if s.get(field) == val]
            return {"took": 1, "hits": {"hits": hits[: body.get("size", 10)]}}
        knn = q["knn"]["embedding"]
        v = np.asarray(knn["vector"], dtype=np.float32)
        scored = []
        for i, s in vis.items():
            ip = float(np.dot(v, np.asarray(s["embedding"], dtype=np.float32)))
            score = ip + 1.0 if ip >= 0 else 1.0 / (1.0 - ip)
            scored.append((score, i))
        scored.sort(reverse=True)
        hits = [
            {"_id": i, "_score": sc, "_source": {"doc_id": vis[i]["doc_id"], "version": vis[i]["version"]}}
            for sc, i in scored[: knn["k"]]
        ]
        return {"took": 2, "hits": {"hits": hits, "total": {"value": len(vis)}}}

    def msearch(self, body):
        responses = []
        for i in range(0, len(body), 2):
            responses.append(self.search(body[i]["index"], body[i + 1]))
        return {"responses": responses}
