import csv

import numpy as np
import pytest

from fakes import FakeOS
from simtest.config import load_config
from simtest.data import generate_sample, read_jsonl
from simtest.embedder import make_embedder
from simtest.loadtest import make_query_vectors, run_load
from simtest.match import run_match
from simtest.scenarios import Workbench, plan, run_scenarios
from simtest.store import LocalStore, OpenSearchStore, StoreError

DIM = 128


@pytest.fixture(scope="module")
def sample(tmp_path_factory):
    d = tmp_path_factory.mktemp("data")
    generate_sample(str(d), n_corpus=300, n_new=40, n_updates=20, n_dataset_a=40, seed=3)
    return d


def make_wb(sample, store_factory):
    cfg = load_config(None)
    cfg["embedder"].update(type="hash", dimensions=DIM, cache_path=None)
    cfg["run"].update(n_queries=40, visibility_sample=5, delete_sample=5, visibility_timeout=10)
    cfg["thresholds"]["new_neighbor_hit_at_k"] = 0.5  # hash embedder is lexical, not semantic
    emb = make_embedder(cfg)
    wb = Workbench(cfg, store_factory(cfg), emb)
    wb.load_corpus(read_jsonl(str(sample / "corpus.jsonl")))
    wb.ingest(recreate=True, progress=False)
    return wb


def os_store(fake, **kw):
    def factory(cfg):
        tcfg = {"name": "t", "index": "idx", "backoff_base": 0, "post_create_wait": 0, "bulk_size": 50, "bulk_workers": 2}
        tcfg.update(kw)
        return OpenSearchStore(tcfg, DIM, client=fake)

    return factory


def run_all(wb, sample, names=("recall", "existing", "new", "updated", "deleted")):
    return run_scenarios(
        wb, list(names), read_jsonl(str(sample / "new_cases.jsonl")), read_jsonl(str(sample / "updates.jsonl")), log=lambda m: None
    )


def assert_all_pass(results):
    bad = [(r["name"], r.get("error"), [c for c in r["checks"] if not c["passed"]]) for r in results if not r["passed"]]
    assert not bad, bad


def test_plan_ordering():
    assert plan(["recall", "new", "existing"]) == ["recall_baseline", "existing", "new", "recall_after_changes"]
    assert plan(["existing"]) == ["existing"]
    assert plan(["recall", "existing"]) == ["recall_baseline", "existing"]
    with pytest.raises(ValueError):
        plan(["bogus"])


def test_all_scenarios_pass_on_local_store(sample):
    results = run_all(make_wb(sample, lambda cfg: LocalStore()), sample)
    assert [r["name"] for r in results] == ["recall_baseline", "existing", "new", "updated", "deleted", "recall_after_changes"]
    assert_all_pass(results)


def test_all_scenarios_pass_on_managed_emulator(sample):
    fake = FakeOS()
    wb = make_wb(sample, os_store(fake, service="es", id_mode="custom"))
    results = run_all(wb, sample)
    assert_all_pass(results)
    assert fake.refreshes >= 1


def test_all_scenarios_pass_on_serverless_emulator_with_visibility_lag(sample):
    fake = FakeOS(lag=2)
    wb = make_wb(sample, os_store(fake, service="aoss", id_mode="auto"))
    results = run_all(wb, sample)
    assert_all_pass(results)
    new = next(r for r in results if r["name"] == "new")
    assert new["metrics"]["time_to_searchable_p50_s"] > 0  # the lag was observed and measured
    # state in the index matches what the framework believes: no duplicate doc_ids
    ids = [s["doc_id"] for s in fake.docs.values()]
    assert len(ids) == len(set(ids)) == len(wb.records)


def test_updated_scenario_flags_stale_copy_when_old_version_cannot_be_deleted(sample):
    fake = FakeOS(broken_term_lookup=True)
    wb = make_wb(sample, os_store(fake, service="aoss", id_mode="auto"))
    (res,) = run_all(wb, sample, names=("updated",))
    assert res["passed"] is False
    failed = {c["metric"] for c in res["checks"] if not c["passed"]}
    assert "old_vector_served" in failed
    assert res["metrics"]["old_vector_served"] > 0


def test_new_scenario_scores_expected_neighbor(sample):
    wb = make_wb(sample, lambda cfg: LocalStore())
    (res,) = run_all(wb, sample, names=("new",))
    m = res["metrics"]
    assert m["new_cases"] == 40 and m["neighbor_queries"] > 0
    assert m["post_index_hit_at_1"] == 1.0
    assert len(wb.records) == 340 and wb.mirror.count() == 340


def test_updated_scenario_updates_versions_and_mirror(sample):
    wb = make_wb(sample, lambda cfg: LocalStore())
    (res,) = run_all(wb, sample, names=("updated",))
    assert res["metrics"]["updated_cases"] == 20
    updated_ids = {u["id"] for u in read_jsonl(str(sample / "updates.jsonl"))}
    assert all(wb.records[i]["version"] == 2 and wb.mirror.get_version(i) == 2 for i in updated_ids)


def test_scenario_exception_is_captured_not_raised(sample):
    class Broken(LocalStore):
        def search(self, vector, k=10):
            raise RuntimeError("boom")

    wb = make_wb(sample, lambda cfg: Broken())
    (res,) = run_all(wb, sample, names=("existing",))
    assert res["passed"] is False and "boom" in res["error"]


def test_ensure_populated_aborts_when_index_state_is_unknown(sample):
    from simtest.cli import _ensure_populated

    wb = make_wb(sample, lambda cfg: LocalStore())
    wb.store.delete(wb.store.search(np.ones(DIM, dtype=np.float32), 1)[0].id)  # index now differs from corpus
    with pytest.raises(SystemExit, match="--reingest"):
        _ensure_populated(wb, {"kind": "opensearch"}, reingest=False)
    _ensure_populated(wb, {"kind": "opensearch"}, reingest=True)  # rebuilds
    assert wb.store.count() == wb.mirror.count()


# ---- load test -------------------------------------------------------------------------------
def test_load_stages_and_writers(sample):
    wb = make_wb(sample, lambda cfg: LocalStore())
    vecs = make_query_vectors(wb, 30)
    res = run_load(wb.store, vecs, [2, 4], duration=0.3, warmup=0.0, k=5, batch=1, thresholds={"load_max_p99_ms": 1000, "load_max_error_rate": 0.01},
                   writer_docs=wb.docs()[:20], writers=2, pause=0, log=lambda m: None)
    assert [s["clients"] for s in res["stages"]] == [2, 4]
    assert all(s["qps"] > 0 and s["errors"] == 0 and s["ok"] for s in res["stages"])
    assert res["stages"][0]["writers"]["writes"] > 0
    assert res["verdict"]["last_ok_clients"] == 4 and res["verdict"]["first_degraded_clients"] is None


def test_load_counts_errors_and_429s_and_flags_degradation(sample):
    class Throttled(LocalStore):
        def search_with_took(self, vector, k=10):
            err = StoreError("rejected")
            err.status_code = 429
            raise err

    wb = make_wb(sample, lambda cfg: Throttled())
    res = run_load(wb.store, make_query_vectors(wb, 10), [2], duration=0.2, warmup=0.0, k=5, batch=1,
                   thresholds={"load_max_p99_ms": 500, "load_max_error_rate": 0.01}, pause=0, log=lambda m: None)
    st = res["stages"][0]
    assert st["errors"] > 0 and st["throttled_429"] == st["errors"] and st["ok"] is False
    assert res["verdict"]["first_degraded_clients"] == 2


def test_load_batch_mode_counts_queries_per_request(sample):
    wb = make_wb(sample, lambda cfg: LocalStore())
    res = run_load(wb.store, make_query_vectors(wb, 10), [1], duration=0.2, warmup=0.0, k=3, batch=5,
                   thresholds={}, pause=0, log=lambda m: None)
    st = res["stages"][0]
    assert st["queries"] == st["requests"] * 5


# ---- dataset matching ------------------------------------------------------------------------
def test_match_writes_csv_and_scores_labelled_matches(sample, tmp_path):
    wb = make_wb(sample, lambda cfg: LocalStore())
    recs = read_jsonl(str(sample / "dataset_a.jsonl"))
    out = tmp_path / "m.csv"
    m = run_match(wb, recs, k=5, min_sim=0.0, out_csv=str(out), batch=16, workers=2, log=lambda x: None)
    rows = list(csv.DictReader(open(out)))
    assert len(rows) == len(recs) * 5 == m["rows_written"]
    assert {"query_id", "rank", "match_id", "similarity"} <= set(rows[0])
    assert m["labelled_positive"] > 0 and m["hit_at_k"] >= 0.5  # lexical embedder: loose lower bound
    # works through the OpenSearch store's _msearch path too
    wb2 = make_wb(sample, os_store(FakeOS(), service="es"))
    m2 = run_match(wb2, recs, k=5, min_sim=0.0, out_csv=str(tmp_path / "m2.csv"), batch=16, workers=2, log=lambda x: None)
    assert m2["hit_at_k"] == m["hit_at_k"]
