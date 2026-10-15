import json

import pytest

from simtest.meta import config_snapshot
from simtest.multireport import (
    _best_flags, build_columns, build_report, direction, load_results, render_html, render_markdown, write_report,
)
from simtest.sweep import apply_overrides, parse_vary, plan


def run_doc(label, recall, p99, cfg=None, endpoint="search-x.example.com"):
    return {
        "meta": {
            "label": label, "target": "managed", "timestamp": "2026-01-01T10:00:00", "embedder": "titan", "dim": 1024,
            "corpus_size": 100, "k": 10, "target_info": {"endpoint": endpoint, "service": "es"},
            "embedder_stats": {"calls": 10, "cache_hits": 0, "throttles": 0, "latency_ms": {"p50": 5}},
            "ingest": {"seconds": 2.0, "docs_per_sec": 50, "errors": 0, "throttled": 0},
            "config": cfg or {"target": {"m": 16, "endpoint": endpoint}},
        },
        "scenarios": [{
            "name": "recall_baseline", "seconds": 1.0, "passed": recall >= 0.9,
            "metrics": {"recall_at_k": recall, "latency_ms": {"p50": 3, "p99": p99, "count": 5}},
            "checks": [{"metric": "recall_at_k", "op": ">=", "threshold": 0.9, "value": recall, "passed": recall >= 0.9}],
        }],
        "passed": True,
    }


def load_doc(label, qps_by_clients):
    stages = [{"clients": c, "qps": q, "latency_ms": {"p50": 1, "p99": 400 / q * 100}, "errors": 0, "throttled_429": 0, "ok": True}
              for c, q in qps_by_clients.items()]
    return {"meta": {"label": label, "target": "managed", "timestamp": "2026-01-01T10:05:00"},
            "load": {"stages": stages, "verdict": {"peak_qps": max(qps_by_clients.values()), "last_ok_clients": 8, "first_degraded_clients": None}}}


def test_direction_rules():
    assert direction("recall_baseline.recall_at_k") == "max"
    assert direction("recall_baseline.latency_ms.p99") == "min"  # the scenario name contains 'recall'
    assert direction("load[8].qps") == "max" and direction("load[8].p99_ms") == "min"
    assert direction("new.time_to_searchable_p95_s") == "min"
    assert direction("updated.old_vector_served") == "min"
    assert direction("updated.hit_at_1") == "max"
    assert direction("recall_baseline.queries") is None and direction("existing.k") is None


def test_best_flags_ignore_noise_and_none():
    assert _best_flags([1.0, 0.999], "max") == [False, False]  # within 2%: a tie
    assert _best_flags([0.95, 0.80], "max") == [True, False]
    assert _best_flags([5, 9, 5.05], "min") == [True, False, True]  # near-equal leaders both count
    assert _best_flags([1, None], "max") == [False, False]  # one run: nothing to compare
    assert _best_flags([1, 2], None) == [False, False]


def test_columns_merge_by_label_and_disambiguate():
    docs = [("a", run_doc("m16", 0.9, 30)), ("b", load_doc("m16", {8: 100})), ("c", run_doc("m16", 0.8, 40))]
    cols = build_columns(docs)
    assert [c["label"] for c in cols] == ["m16", "m16#2"]
    assert set(cols[0]["kinds"]) == {"run", "load"} and cols[0]["load"]["stages"]
    assert build_columns(docs, labels=["x", "y", "z"])[0]["label"] == "x"


def test_report_marks_best_and_lists_only_config_differences(tmp_path):
    cfg_a = {"target": {"m": 16, "ef_search": 100}, "run": {"k": 10}}
    cfg_b = {"target": {"m": 32, "ef_search": 100}, "run": {"k": 10}}
    cols = build_columns([("a", run_doc("m16", 0.80, 30, cfg_a)), ("b", run_doc("m32", 0.95, 20, cfg_b))])
    rep = build_report(cols)
    md = render_markdown(rep)
    assert "| Recall@k, baseline | 0.8 | **0.95** |" in md
    assert "| Query latency p99 (ms) | 30 | **20** |" in md
    assert "target.m" in md and "ef_search" not in md.split("What differs")[1]
    assert "Best on the most metrics: m32" in md
    assert "FAIL" in md and "recall_at_k >= 0.9 (got 0.8)" in md


def test_redact_hides_endpoints():
    cols = build_columns([("a", run_doc("A", 0.9, 30, endpoint="secret-host-a")), ("b", run_doc("B", 0.9, 30, endpoint="secret-host-b"))])
    assert "secret-host-a" in render_markdown(build_report(cols))
    red = render_markdown(build_report(cols, redact=True)) + render_html(build_report(cols, redact=True))
    assert "secret-host" not in red


def test_html_has_charts_legend_and_escapes_labels():
    docs = [("a", run_doc("<b>x</b>", 0.9, 30)), ("l1", load_doc("<b>x</b>", {8: 100, 16: 150})),
            ("b", run_doc("y", 0.9, 30)), ("l2", load_doc("y", {8: 90, 16: 120}))]
    page = render_html(build_report(build_columns(docs)))
    assert "<polyline" in page and page.count("<polyline") == 2 * 2  # two series in each of two charts
    assert "&lt;b&gt;x&lt;/b&gt;" in page and "<b>x</b>" not in page
    assert 'class="legend"' in page and "prefers-color-scheme:dark" in page


def test_single_run_has_no_legend_or_leader():
    page = render_html(build_report(build_columns([("a", run_doc("only", 0.9, 30)), ("l", load_doc("only", {8: 100, 16: 120}))])))
    assert 'class="legend"' not in page and "Best on the most metrics" not in page


def test_load_results_skips_bad_files(tmp_path):
    good, junk, notjson = tmp_path / "g.json", tmp_path / "j.json", tmp_path / "n.json"
    good.write_text(json.dumps(run_doc("a", 0.9, 30)))
    junk.write_text(json.dumps({"hello": 1}))
    notjson.write_text("{")
    docs, skipped = load_results([str(tmp_path)])
    assert len(docs) == 1 and len(skipped) == 2
    with pytest.raises(ValueError):
        write_report([str(junk)], str(tmp_path / "out"))


def test_config_snapshot_masks_secrets():
    cfg = {"embedder": {"model_id": "m"}, "run": {}, "thresholds": {}}
    snap = config_snapshot(cfg, {"name": "t", "index": "i", "headers": {"Authorization": "Bearer x"}, "client_key": "/k.pem", "m": 16})
    assert snap["target"]["headers"] == "***" and snap["target"]["client_key"] == "***"
    assert snap["target"]["m"] == 16 and "name" not in snap["target"]


def test_sweep_parse_and_plan():
    vary = parse_vary(["m=16,32", "query_ef_search=50,100"])
    assert vary == [("m", [16, 32]), ("query_ef_search", [50, 100])]
    pts = plan(vary, ["recall", "existing"], "managed")
    assert [p["reingest"] for p in pts] == [True, False, True, False]  # query-only changes reuse the index
    assert pts[0]["label"] == "managed m=16,query_ef_search=50"
    assert all(p["reingest"] for p in plan(vary, ["recall", "new"]))  # mutating scenarios always rebuild
    for bad in (["m"], ["m="], ["m=1", "m=2"]):
        with pytest.raises(ValueError):
            parse_vary(bad)


def test_apply_overrides_routes_keys():
    base = {"embedder": {"dimensions": 1024}, "run": {"k": 10}, "targets": {"t": {"m": 16}}}
    cfg = apply_overrides(base, "t", {"embedder.dimensions": 256, "run.k": 5, "m": 32, "ef_search": 7})
    assert cfg["embedder"]["dimensions"] == 256 and cfg["run"]["k"] == 5
    assert cfg["targets"]["t"] == {"m": 32, "ef_search": 7} and base["targets"]["t"] == {"m": 16}
