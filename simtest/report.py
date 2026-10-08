"""Console reporting and result comparison."""
from __future__ import annotations

import json
from typing import Any


def table(rows: list[list[Any]], headers: list[str]) -> str:
    cells = [[str(c) for c in headers]] + [[("" if c is None else str(c)) for c in r] for r in rows]
    widths = [max(len(r[i]) for r in cells) for i in range(len(headers))]
    lines = ["  ".join(c.ljust(widths[i]) for i, c in enumerate(r)) for r in cells]
    lines.insert(1, "  ".join("-" * w for w in widths))
    return "\n".join(lines)


def _lat(d: dict) -> str:
    if not d or not d.get("count"):
        return "-"
    return f"p50={d['p50']}ms p95={d['p95']}ms p99={d['p99']}ms"


def print_run(results: list[dict], meta: dict) -> None:
    print("\n" + "=" * 78)
    print(f"Target: {meta.get('target')}   index: {meta.get('index')}   embedder: {meta.get('embedder')} (dim {meta.get('dim')})")
    print(f"Corpus: {meta.get('corpus_size')} cases   k={meta.get('k')}")
    print("=" * 78)
    for r in results:
        status = "PASS" if r["passed"] else "FAIL"
        print(f"\n[{status}] {r['name']}  ({r['seconds']}s)")
        if r.get("error"):
            print(f"   ERROR: {r['error']}")
        for c in r["checks"]:
            mark = "ok " if c["passed"] else "BAD"
            print(f"   {mark} {c['metric']} {c['op']} {c['threshold']}   (got {c['value']})")
        for key, val in r["metrics"].items():
            if isinstance(val, dict):
                if "p50" in val:
                    print(f"      {key}: {_lat(val)}")
                elif val:
                    print(f"      {key}: {json.dumps(val)}")
            elif val is not None and key not in {c["metric"] for c in r["checks"]}:
                print(f"      {key}: {val}")
    overall = all(r["passed"] for r in results)
    print("\nOVERALL:", "PASS" if overall else "FAIL")


def print_load(res: dict) -> None:
    rows = []
    for s in res["stages"]:
        lat = s["latency_ms"]
        took = s.get("server_took_ms") or {}
        w = s.get("writers") or {}
        rows.append(
            [
                s["clients"],
                s["qps"],
                lat.get("p50"),
                lat.get("p95"),
                lat.get("p99"),
                lat.get("max"),
                took.get("p50"),
                took.get("p99"),
                s["errors"],
                s["throttled_429"],
                w.get("writes_per_sec", ""),
                "ok" if s["ok"] else "DEGRADED",
            ]
        )
    print()
    print(
        table(
            rows,
            ["clients", "qps", "p50ms", "p95ms", "p99ms", "maxms", "srv p50", "srv p99", "errors", "429s", "writes/s", "verdict"],
        )
    )
    v = res["verdict"]
    print(
        f"\nPeak QPS {v['peak_qps']}; last healthy stage: {v['last_ok_clients']} clients; "
        f"first degraded: {v['first_degraded_clients']} (limits: p99 <= {v['max_p99_ms']}ms, error rate <= {v['max_error_rate']})"
    )


def flatten(obj: Any, prefix: str = "") -> dict:
    out: dict = {}
    if isinstance(obj, dict):
        for k, v in obj.items():
            out.update(flatten(v, f"{prefix}{k}."))
    else:
        out[prefix.rstrip(".")] = obj
    return out


def _numeric_view(doc: dict) -> dict:
    flat: dict = {}
    if "scenarios" in doc:
        for s in doc["scenarios"]:
            for k, v in flatten(s.get("metrics", {})).items():
                flat[f"{s['name']}.{k}"] = v
            flat[f"{s['name']}.passed"] = s["passed"]
    if "load" in doc:
        for s in doc["load"]["stages"]:
            c = s["clients"]
            flat[f"load[{c}].qps"] = s["qps"]
            flat[f"load[{c}].p50_ms"] = s["latency_ms"].get("p50")
            flat[f"load[{c}].p99_ms"] = s["latency_ms"].get("p99")
            flat[f"load[{c}].errors"] = s["errors"]
            flat[f"load[{c}].throttled_429"] = s["throttled_429"]
    if "match" in doc:
        for k, v in flatten(doc["match"]).items():
            flat[f"match.{k}"] = v
    return flat


def compare(a: dict, b: dict, label_a: str, label_b: str) -> str:
    fa, fb = _numeric_view(a), _numeric_view(b)
    rows = []
    for key in sorted(set(fa) | set(fb)):
        va, vb = fa.get(key), fb.get(key)
        ratio = ""
        if isinstance(va, (int, float)) and isinstance(vb, (int, float)) and not isinstance(va, bool) and va:
            ratio = f"{vb / va:.2f}x"
        rows.append([key, va, vb, ratio])
    return table(rows, ["metric", label_a, label_b, f"{label_b}/{label_a}"])
