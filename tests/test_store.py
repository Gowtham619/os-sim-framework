import numpy as np
import pytest

from fakes import FakeOS
from simtest.metrics import hit_rate, mrr, percentile, rank_of, recall_at_k, summarize_latency
from simtest.store import Doc, LocalStore, OpenSearchStore, score_to_similarity

DIM = 8


def unit(seed):
    v = np.random.default_rng(seed).normal(size=DIM).astype(np.float32)
    return v / np.linalg.norm(v)


def doc(i, version=1, seed=None):
    return Doc(f"d{i}", f"text {i}", unit(i if seed is None else seed), version)


def os_cfg(**kw):
    cfg = {"name": "t", "index": "idx", "service": "es", "id_mode": "custom", "backoff_base": 0, "post_create_wait": 0,
           "space_type": "innerproduct", "engine": "faiss", "shards": 3, "replicas": 1, "bulk_size": 4, "bulk_workers": 2}
    cfg.update(kw)
    return cfg


# ---- metrics ------------------------------------------------------------------------------------
def test_metrics_helpers():
    assert recall_at_k(["a", "b", "c"], ["c", "a", "x"], 3) == pytest.approx(2 / 3)
    assert recall_at_k([], ["a"]) == 1.0
    assert rank_of("b", ["a", "b"]) == 2 and rank_of("z", ["a"]) is None
    ranks = [1, 2, None, 1]
    assert hit_rate(ranks, 1) == 0.5 and hit_rate(ranks, 2) == 0.75
    assert mrr(ranks) == pytest.approx((1 + 0.5 + 0 + 1) / 4)
    assert percentile([], 50) is None
    assert summarize_latency([1, 2, 3, 4, 100])["max"] == 100
    assert summarize_latency([]) == {"count": 0}


def test_rank_with_ties_does_not_penalise_identical_content():
    from simtest.metrics import rank_with_ties
    from simtest.store import Hit

    hits = [Hit("other-copy", 2.0, 1.0), Hit("me", 2.0, 1.0), Hit("neighbor", 1.7, 0.7)]
    assert rank_with_ties("me", hits) == 1  # tied with an identical copy: still a hit at 1
    worse = [Hit("neighbor", 1.7, 0.9), Hit("me", 1.5, 0.5)]
    assert rank_with_ties("me", worse) == 2
    assert rank_with_ties("ghost", hits) is None


def test_score_to_similarity_roundtrip():
    for ip in (-0.6, -0.1, 0.0, 0.4, 0.99):
        score = ip + 1.0 if ip >= 0 else 1.0 / (1.0 - ip)
        assert score_to_similarity("innerproduct", score) == pytest.approx(ip)
    assert score_to_similarity("cosinesimil", 1.8) == pytest.approx(0.8)
    assert score_to_similarity("l2", 0.5) == 0.5


# ---- local store --------------------------------------------------------------------------------
def test_local_store_search_update_delete():
    s = LocalStore()
    s.bulk_index([doc(i) for i in range(20)])
    assert s.count() == 20
    top = s.search(unit(3), 3)
    assert top[0].id == "d3" and top[0].similarity == pytest.approx(1.0, abs=1e-5)
    s.upsert(Doc("d3", "changed", unit(99), 2))
    assert s.count() == 20 and s.get_version("d3") == 2
    assert s.search(unit(99), 1)[0].id == "d3"
    assert s.search(unit(3), 1)[0].id != "d3"
    s.delete("d3")
    assert s.count() == 19 and all(h.id != "d3" for h in s.search(unit(99), 5))
    s.delete("does-not-exist")


# ---- OpenSearch store against the emulator ------------------------------------------------------
def test_index_body_managed_vs_serverless():
    managed = OpenSearchStore(os_cfg(), DIM, client=FakeOS()).index_body()
    sless = OpenSearchStore(os_cfg(service="aoss"), DIM, client=FakeOS()).index_body()
    assert managed["settings"]["index"]["number_of_shards"] == 3
    assert managed["settings"]["index"]["knn"] is True
    assert "number_of_shards" not in sless["settings"]["index"]
    assert "refresh_interval" not in sless["settings"]["index"]
    for body in (managed, sless):
        emb = body["mappings"]["properties"]["embedding"]
        assert emb["dimension"] == DIM and emb["method"]["engine"] == "faiss"
        assert emb["method"]["space_type"] == "innerproduct"
        assert body["mappings"]["properties"]["doc_id"]["type"] == "keyword"


def test_field_extra_and_ef_search_null():
    body = OpenSearchStore(os_cfg(field_extra={"data_type": "byte"}, ef_search=None), DIM, client=FakeOS()).index_body()
    assert body["mappings"]["properties"]["embedding"]["data_type"] == "byte"
    assert "knn.algo_param.ef_search" not in body["settings"]["index"]


def test_create_index_recreate_and_refresh_behaviour():
    fake = FakeOS()
    s = OpenSearchStore(os_cfg(), DIM, client=fake)
    s.create_index()
    s.bulk_index([doc(1)])
    s.create_index(recreate=False)
    assert s.count() == 1
    s.create_index(recreate=True)
    assert s.count() == 0
    s.refresh()
    assert fake.refreshes == 1
    sl_fake = FakeOS()
    sl = OpenSearchStore(os_cfg(service="aoss"), DIM, client=sl_fake)
    sl.refresh()
    assert sl_fake.refreshes == 0  # no _refresh on Serverless


def test_bulk_custom_vs_auto_ids():
    f1 = FakeOS()
    OpenSearchStore(os_cfg(id_mode="custom"), DIM, client=f1).bulk_index([doc(1), doc(2)])
    assert f1.last_bulk_body[0]["index"]["_id"] == "d1"
    assert set(f1.docs) == {"d1", "d2"}
    f2 = FakeOS()
    OpenSearchStore(os_cfg(id_mode="auto"), DIM, client=f2).bulk_index([doc(1), doc(2)])
    assert "_id" not in f2.last_bulk_body[0]["index"]
    assert {s["doc_id"] for s in f2.docs.values()} == {"d1", "d2"}


def test_bulk_retries_429_items():
    fake = FakeOS(throttle_first_bulk=True)
    st = OpenSearchStore(os_cfg(bulk_size=10), DIM, client=fake).bulk_index([doc(i) for i in range(5)])
    assert st["indexed"] == 5 and st["errors"] == 0 and st["throttled"] == 1 and st["retries"] >= 1
    assert len(fake.docs) == 5


def test_search_parse_and_msearch():
    fake = FakeOS()
    s = OpenSearchStore(os_cfg(query_ef_search=64), DIM, client=fake)
    s.bulk_index([doc(i) for i in range(10)])
    hits, took = s.search_with_took(unit(4), 3)
    assert hits[0].id == "d4" and hits[0].version == 1 and hits[0].similarity == pytest.approx(1.0, abs=1e-4)
    assert took == 2.0
    assert fake.last_search_body["query"]["knn"]["embedding"]["method_parameters"] == {"ef_search": 64}
    res = s.msearch([unit(1), unit(2), unit(3)], 2)
    assert [r[0].id for r in res] == ["d1", "d2", "d3"] and all(len(r) == 2 for r in res)


def test_upsert_delete_custom_mode():
    s = OpenSearchStore(os_cfg(), DIM, client=FakeOS())
    s.bulk_index([doc(i) for i in range(5)])
    s.upsert(Doc("d2", "new", unit(50), 2))
    assert s.count() == 5
    assert s.search(unit(50), 1)[0].version == 2
    s.delete("d2")
    s.delete("d2")  # 404 is tolerated
    assert s.count() == 4


def test_upsert_delete_auto_mode_replaces_old_copy():
    fake = FakeOS()
    s = OpenSearchStore(os_cfg(id_mode="auto", service="aoss"), DIM, client=fake)
    s.bulk_index([doc(i) for i in range(5)])
    s.upsert(Doc("d2", "new", unit(50), 2))
    assert s.count() == 5
    assert sum(1 for x in fake.docs.values() if x["doc_id"] == "d2") == 1
    s.delete("d2")
    assert s.count() == 4


def test_wait_visible_measures_lag_and_times_out():
    fake = FakeOS(lag=3)
    s = OpenSearchStore(os_cfg(), DIM, client=fake)
    d = doc(1)
    s.upsert(d)
    t = s.wait_visible(d.id, d.vector, timeout=5, poll=0.001, expect_version=1)
    assert t is not None and fake.search_calls >= 3
    assert s.wait_visible("ghost", unit(77), timeout=0.05, poll=0.001) is None
    assert s.wait_gone("ghost", unit(77), timeout=1, poll=0.001) is not None
    s.delete("d1")
    assert s.wait_gone("d1", d.vector, timeout=1, poll=0.001) is not None
