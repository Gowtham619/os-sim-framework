"""Multi-run comparison report: N result files in, one Markdown and/or self-contained HTML report out.

Runs that share a label (for example the `run` and `load` results of one sweep point) become one column.
Every table is built once as plain data and rendered to both formats, so they always agree.
"""
from __future__ import annotations

import glob
import html
import json
import math
import os
from datetime import datetime
from typing import Any, Optional

from .report import _numeric_view, flatten

# Keys hidden by --redact (hosts, accounts and anything that identifies the environment).
REDACT_KEYS = {"endpoint", "profile", "ca_certs", "client_cert", "client_key", "headers", "url_prefix", "host", "port"}
_NOISE_KEYS = {"cache_path"}
_NEUTRAL_LAST = {
    "count", "queries", "k", "passed", "new_cases", "updated_cases", "timed_individually", "neighbor_queries",
    "rows_written", "skipped_not_in_index", "old_vector_checked", "labelled_positive", "labelled_negative",
    "top1_above_min_similarity", "indexed", "csv", "recall_min",
}
_MAX_HINTS = ("recall", "hit_at", "mrr", "qps", "docs_per_sec", "self_similarity")
_MIN_HINTS = ("seconds", "error", "throttl", "429", "duplicate", "stale", "old_vector_served", "leftover", "timeout",
              "retries", "false_positive")

# (title, metric key) - shown only when at least one run has a value.
QUALITY_ROWS = [
    ("Recall@k, baseline", "recall_baseline.recall_at_k"),
    ("Recall@k, after changes", "recall_after_changes.recall_at_k"),
    ("Exact text -> itself at rank 1", "existing.exact_hit_at_1"),
    ("Re-worded text -> itself in top k", "existing.perturbed_hit_at_k"),
    ("New case findable after indexing (hit@1)", "new.post_index_hit_at_1"),
    ("New case finds expected neighbour (hit@k)", "new.neighbor_hit_at_k"),
    ("Updated case found at rank 1", "updated.hit_at_1"),
    ("Old embedding still served after update", "updated.old_vector_served"),
    ("Duplicates after update", "updated.duplicates"),
    ("Stale versions after update", "updated.stale_versions"),
    ("Dataset match hit@1", "match.hit_at_1"),
    ("Dataset match hit@k", "match.hit_at_k"),
    ("Dataset match MRR", "match.mrr"),
]
SPEED_ROWS = [
    ("Query latency p50 (ms)", "recall_baseline.latency_ms.p50"),
    ("Query latency p95 (ms)", "recall_baseline.latency_ms.p95"),
    ("Query latency p99 (ms)", "recall_baseline.latency_ms.p99"),
    ("Time to searchable, new case, p95 (s)", "new.time_to_searchable_p95_s"),
    ("Time to searchable, updated case, p95 (s)", "updated.time_to_searchable_p95_s"),
    ("Dataset match throughput (queries/s)", "match.search_qps"),
]
# (title, dotted path into meta, direction)
EFFORT_ROWS = [
    ("Embedding calls made (not cached)", "embedder_stats.calls", None),
    ("Embedding cache hits", "embedder_stats.cache_hits", None),
    ("Bedrock throttles", "embedder_stats.throttles", "min"),
    ("Embedding latency p50 (ms)", "embedder_stats.latency_ms.p50", "min"),
    ("Ingest time (s)", "ingest.seconds", "min"),
    ("Ingest rate (docs/s)", "ingest.docs_per_sec", "max"),
    ("Ingest errors", "ingest.errors", "min"),
    ("Ingest throttles (429)", "ingest.throttled", "min"),
]

# Reference categorical palette, slots 1-8 in their validated order.
SERIES_LIGHT = ["#2a78d6", "#eb6834", "#1baf7a", "#eda100", "#e87ba4", "#008300", "#4a3aa7", "#e34948"]
SERIES_DARK = ["#3987e5", "#d95926", "#199e70", "#c98500", "#d55181", "#008300", "#9085e9", "#e66767"]
MAX_SERIES = len(SERIES_LIGHT)


# ------------------------------------------------------------------------------------------------
def direction(key: str) -> Optional[str]:
    """'max' if higher is better, 'min' if lower is better, None if the metric is only informational."""
    rest = key.split(".", 1)[1] if "." in key else key
    if rest.rsplit(".", 1)[-1] in _NEUTRAL_LAST:
        return None
    if "latency" in rest or rest.endswith("_ms") or "took" in rest:
        return "min"
    if any(h in rest for h in _MAX_HINTS):
        return "max"
    if any(h in rest for h in _MIN_HINTS) or rest.endswith("_s"):
        return "min"
    return None


def fmt(v: Any) -> str:
    if v is None or v == "":
        return "–"
    if isinstance(v, bool):
        return "yes" if v else "no"
    if isinstance(v, int):
        return f"{v:,}"
    if isinstance(v, float):
        return f"{v:,.0f}" if abs(v) >= 1000 else f"{v:.4g}"
    return str(v)


def _dig(d: Any, path: str) -> Any:
    for part in path.split("."):
        if not isinstance(d, dict):
            return None
        d = d.get(part)
    return d


def _num(v: Any) -> bool:
    return isinstance(v, (int, float)) and not isinstance(v, bool)


# ------------------------------------------------------------------------------------------------
def collect_files(paths: list[str]) -> list[str]:
    out: list[str] = []
    for p in paths:
        if os.path.isdir(p):
            out += sorted(glob.glob(os.path.join(p, "*.json")))
        else:
            out.append(p)
    return out


def load_results(paths: list[str]) -> tuple[list[tuple[str, dict]], list[str]]:
    docs, skipped = [], []
    for p in collect_files(paths):
        try:
            with open(p, "r", encoding="utf-8") as fh:
                d = json.load(fh)
        except (OSError, ValueError) as exc:
            skipped.append(f"{p}: {type(exc).__name__}")
            continue
        if isinstance(d, dict) and "meta" in d and any(k in d for k in ("scenarios", "load", "match")):
            docs.append((p, d))
        else:
            skipped.append(f"{p}: not a simtest result file")
    return docs, skipped


def _kind(doc: dict) -> str:
    return "run" if "scenarios" in doc else "load" if "load" in doc else "match"


def build_columns(docs: list[tuple[str, dict]], labels: Optional[list[str]] = None) -> list[dict]:
    """One column per label. Results sharing a label merge; a repeated kind under one label becomes `label#2`."""
    cols: dict[str, dict] = {}
    order: list[str] = []
    for i, (path, doc) in enumerate(docs):
        meta = doc.get("meta") or {}
        base = (labels[i] if labels and i < len(labels) else None) or meta.get("label")
        if not base:
            base = f"{meta.get('target', 'run')}@{str(meta.get('timestamp', i)).replace('T', ' ')}"
        kind, label, n = _kind(doc), base, 1
        while label in cols and kind in cols[label]["kinds"]:
            n += 1
            label = f"{base}#{n}"
        col = cols.get(label)
        if col is None:
            col = cols[label] = {"label": label, "kinds": {}, "meta": meta, "paths": []}
            order.append(label)
        col["kinds"][kind] = doc
        col["paths"].append(path)
        if kind == "run" or not col.get("meta"):
            col["meta"] = meta
    out = []
    for label in order:
        c = cols[label]
        metrics: dict = {}
        for d in c["kinds"].values():
            metrics.update(_numeric_view(d))
        c["metrics"] = metrics
        c["scenarios"] = (c["kinds"].get("run") or {}).get("scenarios") or []
        c["load"] = (c["kinds"].get("load") or {}).get("load")
        out.append(c)
    return out


# ------------------------------------------------------------------------------------------------
TIE_TOLERANCE = 0.02  # values within 2% of the best count as tied, so measurement noise is not called a win


def _best_flags(values: list, how: Optional[str]) -> list[bool]:
    nums = [v for v in values if _num(v)]
    if how is None or len(nums) < 2:
        return [False] * len(values)
    target = max(nums) if how == "max" else min(nums)
    close = [_num(v) and abs(v - target) <= TIE_TOLERANCE * max(abs(target), abs(v), 1e-12) for v in values]
    if all(close[i] for i, v in enumerate(values) if _num(v)):
        return [False] * len(values)  # everything is within tolerance: no meaningful difference
    return close


def _table(headers: list[str], rows: list[list], best: Optional[list[list[bool]]] = None) -> dict:
    return {"headers": headers, "rows": rows, "best": best or [[False] * len(r) for r in rows]}


def _metric_table(cols: list[dict], spec: list[tuple[str, str]], lead: str, wins: dict) -> Optional[dict]:
    rows, best = [], []
    for title, key in spec:
        vals = [c["metrics"].get(key) for c in cols]
        if all(v is None for v in vals):
            continue
        flags = _best_flags(vals, direction(key))
        for c, f in zip(cols, flags):
            if f:
                wins[c["label"]] = wins.get(c["label"], 0) + 1
        wins["_total"] = wins.get("_total", 0) + (1 if any(flags) else 0)
        rows.append([title] + [fmt(v) for v in vals])
        best.append([False] + flags)
    return _table([lead] + [c["label"] for c in cols], rows, best) if rows else None


def _effort_table(cols: list[dict]) -> Optional[dict]:
    rows, best = [], []
    for title, path, how in EFFORT_ROWS:
        vals = [_dig(c["meta"], path) for c in cols]
        if all(v is None for v in vals):
            continue
        rows.append([title] + [fmt(v) for v in vals])
        best.append([False] + _best_flags(vals, how))
    secs = [round(sum(s.get("seconds", 0) or 0 for s in c["scenarios"]), 2) if c["scenarios"] else None for c in cols]
    if any(v is not None for v in secs):
        rows.append(["Scenario run time (s)"] + [fmt(v) for v in secs])
        best.append([False] + _best_flags(secs, "min"))
    return _table(["Cost and effort"] + [c["label"] for c in cols], rows, best) if rows else None


def _scenario_tables(cols: list[dict]) -> tuple[Optional[dict], list[str]]:
    names: list[str] = []
    for c in cols:
        for s in c["scenarios"]:
            if s["name"] not in names:
                names.append(s["name"])
    if not names:
        return None, []
    rows = []
    for n in names:
        row = [n]
        for c in cols:
            s = next((x for x in c["scenarios"] if x["name"] == n), None)
            row.append("–" if s is None else "PASS" if s["passed"] else "FAIL")
        rows.append(row)
    failed: list[str] = []
    for c in cols:
        for s in c["scenarios"]:
            if s.get("error"):
                failed.append(f"{c['label']} / {s['name']}: ERROR {s['error']}")
            for chk in s.get("checks", []):
                if not chk["passed"]:
                    failed.append(f"{c['label']} / {s['name']}: {chk['metric']} {chk['op']} {chk['threshold']} (got {fmt(chk['value'])})")
    return _table(["Scenario"] + [c["label"] for c in cols], rows), failed


def _config_diff(cols: list[dict], redact: bool) -> tuple[Optional[dict], list[str]]:
    with_cfg = [c for c in cols if (c["meta"] or {}).get("config")]
    missing = [c["label"] for c in cols if c not in with_cfg]
    if len(with_cfg) < 2:
        return None, missing
    flat = [flatten(c["meta"]["config"]) for c in with_cfg]
    rows = []
    for key in sorted(set().union(*flat)):
        last = key.rsplit(".", 1)[-1]
        if last in _NOISE_KEYS or (redact and last in REDACT_KEYS):
            continue
        vals = [f.get(key) for f in flat]
        if len({json.dumps(v, sort_keys=True, default=str) for v in vals}) > 1:
            rows.append([key] + [fmt(v) for v in vals])
    return (_table(["Setting that differs"] + [c["label"] for c in with_cfg], rows) if rows else _table(["Setting that differs"], [["(no differences)"]])), missing


def _run_info(cols: list[dict], redact: bool) -> dict:
    rows = []
    for c in cols:
        m = c["meta"] or {}
        info = m.get("target_info") or {}
        row = [
            c["label"], "+".join(k for k in ("run", "load", "match") if k in c["kinds"]), m.get("target", ""), info.get("service") or info.get("kind", ""),
            m.get("embedder", ""), m.get("dim", ""), m.get("corpus_size", ""), m.get("k", ""), str(m.get("timestamp", "")).replace("T", " "),
        ]
        if not redact:
            row.insert(4, info.get("endpoint", ""))
        rows.append([fmt(x) if not isinstance(x, str) else (x or "–") for x in row])
    heads = ["Run", "Results", "Target", "Service", "Embedder", "Dim", "Corpus", "k", "When"]
    if not redact:
        heads.insert(4, "Endpoint")
    return _table(heads, rows)


def _load_section(cols: list[dict]) -> Optional[dict]:
    lc = [c for c in cols if c["load"] and c["load"].get("stages")]
    if not lc:
        return None
    clients = sorted({s["clients"] for c in lc for s in c["load"]["stages"]})

    def grid(getter, how):
        rows, best = [], []
        for n in clients:
            vals = []
            for c in lc:
                st = next((s for s in c["load"]["stages"] if s["clients"] == n), None)
                vals.append(getter(st) if st else None)
            rows.append([str(n)] + [fmt(v) for v in vals])
            best.append([False] + _best_flags(vals, how))
        return _table(["Concurrent clients"] + [c["label"] for c in lc], rows, best)

    def problems(st):
        return None if st is None else (st.get("errors") or 0) + (st.get("throttled_429") or 0)

    verdict_rows = []
    for c in lc:
        v = c["load"].get("verdict") or {}
        verdict_rows.append([c["label"], fmt(v.get("peak_qps")), fmt(v.get("last_ok_clients")), fmt(v.get("first_degraded_clients"))])
    series = {
        "qps": [(c["label"], [next((s["qps"] for s in c["load"]["stages"] if s["clients"] == n), None) for n in clients]) for c in lc],
        "p99": [(c["label"], [next((s["latency_ms"].get("p99") for s in c["load"]["stages"] if s["clients"] == n), None) for n in clients]) for c in lc],
    }
    return {
        "clients": clients,
        "series": series,
        "tables": [
            ("Throughput (queries/s), higher is better", grid(lambda s: s["qps"], "max")),
            ("p99 latency (ms), lower is better", grid(lambda s: s["latency_ms"].get("p99"), "min")),
            ("p50 latency (ms), lower is better", grid(lambda s: s["latency_ms"].get("p50"), "min")),
            ("Errors plus 429 throttles, lower is better", grid(problems, "min")),
        ],
        "verdict": _table(["Run", "Peak QPS", "Last healthy clients", "First degraded clients"], verdict_rows),
    }


def build_report(cols: list[dict], redact: bool = False, title: str = "Similarity search comparison") -> dict:
    wins: dict = {}
    quality = _metric_table(cols, QUALITY_ROWS, "Quality", wins)
    speed = _metric_table(cols, SPEED_ROWS, "Speed", wins)
    scen, failed = _scenario_tables(cols)
    cfg, no_cfg = _config_diff(cols, redact)
    leaders = ""
    if len(cols) >= 2 and wins.get("_total"):
        ranked = sorted(((wins.get(c["label"], 0), c["label"]) for c in cols), reverse=True)
        top = ranked[0]
        tied = [lab for n, lab in ranked if n == top[0]]
        total = wins["_total"]
        if top[0] and len(tied) == 1:
            leaders = f"Best on the most metrics: {tied[0]} ({top[0]} of {total} metrics with a clear difference)."
        elif top[0]:
            leaders = f"No clear overall leader: {', '.join(tied)} are level at {top[0]} of {total} metrics each."
    return {
        "title": title,
        "generated": datetime.now().isoformat(timespec="seconds"),
        "redacted": redact,
        "runs": _run_info(cols, redact),
        "leaders": leaders,
        "scenarios": scen,
        "failed_checks": failed,
        "quality": quality,
        "speed": speed,
        "load": _load_section(cols),
        "effort": _effort_table(cols),
        "config": cfg,
        "no_config": no_cfg,
        "n_runs": len(cols),
    }


# ------------------------------------------------------------------------------------------------
# Markdown
def _md_cell(s: str) -> str:
    return str(s).replace("|", "\\|").replace("\n", " ")


def _md_table(t: dict) -> str:
    lines = ["| " + " | ".join(_md_cell(h) for h in t["headers"]) + " |", "|" + "|".join("---" for _ in t["headers"]) + "|"]
    for row, flags in zip(t["rows"], t["best"]):
        cells = [f"**{_md_cell(c)}**" if f and c not in ("", "–") else _md_cell(c) for c, f in zip(row, flags)]
        lines.append("| " + " | ".join(cells) + " |")
    return "\n".join(lines)


def render_markdown(rep: dict) -> str:
    out = [f"# {rep['title']}", "", f"Generated {rep['generated'].replace('T', ' ')}. {rep['n_runs']} run(s). "
           "Bold marks the best value in a row, only where higher or lower is clearly better."]
    if rep["redacted"]:
        out.append("Endpoints, profiles and other environment identifiers are hidden.")
    out += ["", "## Runs", "", _md_table(rep["runs"])]
    if rep["leaders"]:
        out += ["", f"**{rep['leaders']}** Check the individual rows before choosing: the metrics are not equally important."]
    if rep["scenarios"]:
        out += ["", "## Scenario results", "", _md_table(rep["scenarios"])]
        if rep["failed_checks"]:
            out += ["", "Failed checks:", ""] + [f"- {_md_cell(x)}" for x in rep["failed_checks"]]
    for key, head in (("quality", "Quality"), ("speed", "Speed")):
        if rep[key]:
            out += ["", f"## {head}", "", _md_table(rep[key])]
    if rep["load"]:
        out += ["", "## Load test", "", _md_table(rep["load"]["verdict"])]
        for name, t in rep["load"]["tables"]:
            out += ["", f"### {name}", "", _md_table(t)]
    if rep["effort"]:
        out += ["", "## Cost and effort", "", _md_table(rep["effort"])]
    if rep["config"]:
        out += ["", "## What differs between the runs", "", _md_table(rep["config"])]
    if rep["no_config"]:
        out += ["", "No config was recorded for: " + ", ".join(rep["no_config"]) + " (older results)."]
    return "\n".join(out) + "\n"


# ------------------------------------------------------------------------------------------------
# HTML
_CSS = """
:root{color-scheme:light;--surface:#fcfcfb;--panel:#f4f3f0;--text:#0b0b0b;--text-2:#52514e;--text-3:#7a7974;--grid:#e3e2de;--line:#d6d5d0;--best:#e4effc;--fail:#b3261e;--pass:#1d6b3a;
%LIGHT%}
@media (prefers-color-scheme:dark){:root:not([data-theme="light"]){color-scheme:dark;--surface:#1a1a19;--panel:#242422;--text:#fff;--text-2:#c3c2b7;--text-3:#96958c;--grid:#34342f;--line:#43433d;--best:#1d3350;--fail:#f2918a;--pass:#7fd29b;
%DARK%}}
:root[data-theme="dark"]{color-scheme:dark;--surface:#1a1a19;--panel:#242422;--text:#fff;--text-2:#c3c2b7;--text-3:#96958c;--grid:#34342f;--line:#43433d;--best:#1d3350;--fail:#f2918a;--pass:#7fd29b;
%DARK%}
*{box-sizing:border-box}
body{margin:0;background:var(--surface);color:var(--text);font:15px/1.5 system-ui,-apple-system,"Segoe UI",Roboto,sans-serif}
main{max-width:1100px;margin:0 auto;padding:24px 16px 64px}
h1{font-size:26px;margin:0 0 4px}h2{font-size:18px;margin:36px 0 8px}h3{font-size:15px;margin:22px 0 6px;color:var(--text-2)}
p.sub,p.note{color:var(--text-2);margin:4px 0 12px}.leader{background:var(--panel);border-left:4px solid var(--series-1);padding:10px 14px;margin:16px 0}
.wrap{overflow-x:auto}table{border-collapse:collapse;width:100%;font-size:14px;font-variant-numeric:tabular-nums}
th,td{padding:7px 12px;border-bottom:1px solid var(--line);text-align:right;white-space:nowrap}
th:first-child,td:first-child{text-align:left;white-space:normal}th{color:var(--text-2);font-weight:600;border-bottom:2px solid var(--line)}
td.best{font-weight:700;background:var(--best)}td.pass{color:var(--pass);font-weight:600}td.fail{color:var(--fail);font-weight:700}
ul.fails{color:var(--fail);margin:8px 0;padding-left:20px}
.legend{display:flex;flex-wrap:wrap;gap:6px 18px;margin:8px 0 2px;color:var(--text-2);font-size:13px}
.legend i{display:inline-block;width:14px;height:3px;border-radius:2px;margin-right:6px;vertical-align:middle}
.charts{display:grid;grid-template-columns:repeat(auto-fit,minmax(320px,1fr));gap:20px;margin:8px 0}
svg.chart{width:100%;height:auto;display:block}svg.chart text{fill:var(--text-3);font:12px system-ui,sans-serif}
svg.chart .grid{stroke:var(--grid);stroke-width:1}svg.chart .ttl{fill:var(--text-2);font-size:13px;font-weight:600}
svg.chart .ln{fill:none;stroke-width:2;stroke-linejoin:round;stroke-linecap:round}
svg.chart .dot{stroke:var(--surface);stroke-width:2}svg.chart .hit{fill:transparent}
"""


def _css() -> str:
    light = "".join(f"--series-{i + 1}:{c};" for i, c in enumerate(SERIES_LIGHT))
    dark = "".join(f"--series-{i + 1}:{c};" for i, c in enumerate(SERIES_DARK))
    return _CSS.replace("%LIGHT%", light).replace("%DARK%", dark)


def _nice_max(v: float) -> float:
    if v <= 0:
        return 1.0
    exp = 10 ** math.floor(math.log10(v))
    for m in (1, 2, 3, 4, 6, 8, 10):  # quarter steps stay clean: 0.25, 0.5, 0.75, 1, 1.5, 2, 2.5
        if v <= m * exp:
            return m * exp
    return 10 * exp


def _line_chart(title: str, unit: str, xs: list, series: list[tuple[str, list]]) -> str:
    W, H, ml, mr, mt, mb = 420, 250, 50, 12, 28, 42
    pw, ph = W - ml - mr, H - mt - mb
    series = series[:MAX_SERIES]
    vals = [v for _, ys in series for v in ys if _num(v)]
    ymax = _nice_max(max(vals) if vals else 1)
    inset = 26  # keeps the first and last markers off the plot edge
    px = lambda i: ml + (pw / 2 if len(xs) == 1 else inset + (pw - 2 * inset) * i / (len(xs) - 1))
    py = lambda v: mt + ph * (1 - v / ymax)
    g = [f'<svg class="chart" viewBox="0 0 {W} {H}" role="img" aria-label="{html.escape(title)}">',
         f'<text class="ttl" x="{ml}" y="14">{html.escape(title)}</text>']
    for t in range(5):
        v = ymax * t / 4
        y = py(v)
        g.append(f'<line class="grid" x1="{ml}" x2="{W - mr}" y1="{y:.1f}" y2="{y:.1f}"/>')
        g.append(f'<text x="{ml - 6}" y="{y + 4:.1f}" text-anchor="end">{fmt(round(v, 2))}</text>')
    for i, x in enumerate(xs):
        g.append(f'<text x="{px(i):.1f}" y="{H - mb + 16}" text-anchor="middle">{html.escape(str(x))}</text>')
    g.append(f'<text x="{ml + pw / 2:.1f}" y="{H - 6}" text-anchor="middle">concurrent clients</text>')
    for si, (label, ys) in enumerate(series):
        color = f"var(--series-{si + 1})"
        pts = [(px(i), py(v), xs[i], v) for i, v in enumerate(ys) if _num(v)]
        if len(pts) > 1:
            g.append(f'<polyline class="ln" style="stroke:{color}" points="{" ".join(f"{x:.1f},{y:.1f}" for x, y, _, _ in pts)}"/>')
        for x, y, xv, v in pts:
            tip = html.escape(f"{label}: {fmt(v)}{unit} at {xv} clients")
            g.append(f'<circle class="dot" cx="{x:.1f}" cy="{y:.1f}" r="4" style="fill:{color}"><title>{tip}</title></circle>')
            g.append(f'<circle class="hit" cx="{x:.1f}" cy="{y:.1f}" r="11"><title>{tip}</title></circle>')
    g.append("</svg>")
    return "".join(g)


def _html_table(t: dict) -> str:
    head = "".join(f"<th>{html.escape(str(h))}</th>" for h in t["headers"])
    body = []
    for row, flags in zip(t["rows"], t["best"]):
        cells = []
        for c, f in zip(row, flags):
            cls = "best" if f else "pass" if c == "PASS" else "fail" if c == "FAIL" else ""
            cells.append(f'<td class="{cls}">{html.escape(str(c))}</td>' if cls else f"<td>{html.escape(str(c))}</td>")
        body.append("<tr>" + "".join(cells) + "</tr>")
    return f'<div class="wrap"><table><thead><tr>{head}</tr></thead><tbody>{"".join(body)}</tbody></table></div>'


def _legend(labels: list[str]) -> str:
    if len(labels) < 2:
        return ""
    return '<div class="legend">' + "".join(
        f'<span><i style="background:var(--series-{i + 1})"></i>{html.escape(l)}</span>' for i, l in enumerate(labels[:MAX_SERIES])
    ) + "</div>"


def render_html(rep: dict) -> str:
    h = [f'<!doctype html><html lang="en"><head><meta charset="utf-8"><meta name="viewport" content="width=device-width,initial-scale=1">'
         f"<title>{html.escape(rep['title'])}</title><style>{_css()}</style></head><body><main>",
         f"<h1>{html.escape(rep['title'])}</h1>",
         f'<p class="sub">Generated {html.escape(rep["generated"].replace("T", " "))} &middot; {rep["n_runs"]} run(s). '
         "Bold, shaded cells mark the best value in a row, only where higher or lower is clearly better.</p>"]
    if rep["redacted"]:
        h.append('<p class="note">Endpoints, profiles and other environment identifiers are hidden.</p>')
    h += ["<h2>Runs</h2>", _html_table(rep["runs"])]
    if rep["leaders"]:
        h.append(f'<div class="leader"><strong>{html.escape(rep["leaders"])}</strong> '
                 "Check the individual rows before choosing: the metrics are not equally important.</div>")
    if rep["scenarios"]:
        h += ["<h2>Scenario results</h2>", _html_table(rep["scenarios"])]
        if rep["failed_checks"]:
            h.append('<p class="note">Failed checks:</p><ul class="fails">' + "".join(f"<li>{html.escape(x)}</li>" for x in rep["failed_checks"]) + "</ul>")
    for key, head in (("quality", "Quality"), ("speed", "Speed")):
        if rep[key]:
            h += [f"<h2>{head}</h2>", _html_table(rep[key])]
    if rep["load"]:
        ld = rep["load"]
        labels = [l for l, _ in ld["series"]["qps"]]
        h += ["<h2>Load test</h2>", _html_table(ld["verdict"]), _legend(labels), '<div class="charts">',
              _line_chart("Throughput vs concurrency", " queries/s", ld["clients"], ld["series"]["qps"]),
              _line_chart("p99 latency vs concurrency", " ms", ld["clients"], ld["series"]["p99"]), "</div>"]
        if len(labels) > MAX_SERIES:
            h.append(f'<p class="note">Charts show the first {MAX_SERIES} runs; the tables below include all.</p>')
        for name, t in ld["tables"]:
            h += [f"<h3>{html.escape(name)}</h3>", _html_table(t)]
    if rep["effort"]:
        h += ["<h2>Cost and effort</h2>", _html_table(rep["effort"])]
    if rep["config"]:
        h += ["<h2>What differs between the runs</h2>", _html_table(rep["config"])]
    if rep["no_config"]:
        h.append(f'<p class="note">No config was recorded for: {html.escape(", ".join(rep["no_config"]))} (older results).</p>')
    h.append("</main></body></html>")
    return "".join(h)


# ------------------------------------------------------------------------------------------------
def write_report(
    paths: list[str],
    out_prefix: str,
    fmt_: str = "both",
    labels: Optional[list[str]] = None,
    redact: bool = False,
    title: str = "Similarity search comparison",
) -> tuple[list[str], list[str]]:
    docs, skipped = load_results(paths)
    if not docs:
        raise ValueError("no simtest result files found in: " + ", ".join(paths))
    rep = build_report(build_columns(docs, labels), redact=redact, title=title)
    os.makedirs(os.path.dirname(os.path.abspath(out_prefix)), exist_ok=True)
    written = []
    if fmt_ in ("md", "both"):
        with open(out_prefix + ".md", "w", encoding="utf-8") as fh:
            fh.write(render_markdown(rep))
        written.append(out_prefix + ".md")
    if fmt_ in ("html", "both"):
        with open(out_prefix + ".html", "w", encoding="utf-8") as fh:
            fh.write(render_html(rep))
        written.append(out_prefix + ".html")
    return written, skipped
