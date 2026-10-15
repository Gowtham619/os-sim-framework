"""Command line interface:  python -m simtest <command> [options]"""
from __future__ import annotations

import argparse
import json
import os
import sys
import time
from datetime import datetime

from . import __version__
from .config import get_target, load_config
from .data import generate_sample, read_jsonl, synthetic_texts
from .embedder import make_embedder
from .loadtest import make_query_vectors, run_load
from .match import run_match
from .metrics import summarize_latency
from .report import compare, print_load, print_run, table
from .scenarios import CANONICAL, Workbench, run_scenarios
from .store import make_store


def log(msg: str) -> None:
    print(msg, flush=True)


def _ts() -> str:
    return datetime.now().strftime("%Y%m%d-%H%M%S")


def _save(out_dir: str, target: str, kind: str, doc: dict) -> str:
    os.makedirs(out_dir, exist_ok=True)
    ts = _ts()
    path = os.path.join(out_dir, f"{ts}_{target}_{kind}.json")
    n = 1
    while os.path.exists(path):  # never overwrite an earlier result from the same second
        n += 1
        path = os.path.join(out_dir, f"{ts}-{n}_{target}_{kind}.json")
    with open(path, "w", encoding="utf-8") as fh:
        json.dump(doc, fh, indent=2, default=str)
    return path


def _setup(args):
    cfg = load_config(args.config)
    if getattr(args, "k", None):
        cfg["run"]["k"] = args.k
    if getattr(args, "n_queries", None):
        cfg["run"]["n_queries"] = args.n_queries
    embedder = make_embedder(cfg)
    tcfg = get_target(cfg, args.target)
    if getattr(args, "cmd", None) == "check":
        tcfg["timeout"] = min(int(tcfg.get("timeout", 60)), 15)  # fail fast instead of hanging on a blocked network
        tcfg["max_retries"] = 0
    store = make_store(tcfg, embedder.dim)
    wb = Workbench(cfg, store, embedder)
    return cfg, tcfg, wb


def _load_corpus(wb: Workbench, cfg: dict, path: str | None = None, limit: int | None = None) -> None:
    path = path or cfg["data"]["corpus"]
    recs = read_jsonl(path, limit=limit)
    log(f"Loading {len(recs)} corpus cases from {path} and embedding them (cached after the first run)...")
    wb.load_corpus(recs, progress=True)


def _ensure_populated(wb: Workbench, tcfg: dict, reingest: bool) -> None:
    expected = wb.mirror.count()
    if getattr(wb.store, "read_only", False):
        if reingest:
            raise SystemExit("--reingest would recreate the index, but this target is read_only.")
        try:
            n = wb.store.count()
        except Exception as exc:
            raise SystemExit(f"Cannot read the index ({type(exc).__name__}: {str(exc)[:200]}).")
        if n != expected:
            log(
                f"WARNING: read_only index holds {n} docs but the corpus file has {expected}. Recall and self-retrieval "
                "numbers are only meaningful if the corpus file is the data that is in the index (use --limit/a sample "
                "file that is a subset of it)."
            )
        return
    if reingest:
        log("Re-creating index and ingesting corpus...")
        stats = wb.ingest(recreate=True)
        log(f"  indexed {stats['indexed']} docs ({stats.get('docs_per_sec')} docs/s), errors={stats['errors']}")
        return
    try:
        n = wb.store.count()
    except Exception:
        n = None
    if n == expected:
        return
    if tcfg.get("kind") == "local":
        wb.ingest(recreate=True, progress=False)  # the local store lives in memory, so populate it per process
        return
    if not n:
        raise SystemExit("The index is empty or missing. Run `ingest` first (or pass --reingest).")
    raise SystemExit(
        f"The index holds {n} docs but the corpus file has {expected}. Earlier runs mutate the index; "
        "pass --reingest to rebuild it from the corpus so the test starts from a known state."
    )


# -------------------------------------------------------------------------------------------------
def cmd_init_sample(args) -> int:
    counts = generate_sample(args.out, args.corpus, args.new, args.updates, args.dataset_a, args.seed)
    log(f"Wrote sample data to {args.out}/: {counts}")
    return 0


def _whoami(profile, region: str | None = None) -> str:
    """Which AWS account / role a profile resolves to, so the wrong SAML login is obvious."""
    try:
        import boto3

        from botocore.config import Config

        session = boto3.Session(profile_name=profile) if profile else boto3.Session()
        quick = Config(connect_timeout=5, read_timeout=10, retries={"max_attempts": 1})
        ident = session.client("sts", region_name=region, config=quick).get_caller_identity()
        return f"account {ident['Account']}, {ident['Arn']}"
    except Exception as exc:
        return f"unresolved ({type(exc).__name__}: {str(exc)[:120]})"


def cmd_check(args) -> int:
    cfg, tcfg, wb = _setup(args)
    if cfg["embedder"].get("type") == "titan":
        log(f"Bedrock identity (profile={cfg['embedder'].get('profile')}): "
            f"{_whoami(cfg['embedder'].get('profile'), cfg['embedder'].get('region'))}")
    if tcfg.get("kind") == "opensearch" and tcfg.get("auth", "sigv4") == "sigv4":
        log(f"Target identity (profile={tcfg.get('profile')}): {_whoami(tcfg.get('profile'), tcfg.get('region'))}")
    t0 = time.perf_counter()
    vec = wb.embedder.embed("connectivity check", use_cache=False)
    log(f"Embedder OK: {wb.embedder.model_id}, dim={len(vec)}, {round((time.perf_counter() - t0) * 1000)} ms")
    log(f"Target '{args.target}': {json.dumps(wb.store.describe())}")
    try:
        log(f"Index document count: {wb.store.count()}")
    except Exception as exc:
        log(f"Index not reachable or not created yet ({type(exc).__name__}: {str(exc)[:200]})")
    return 0


def cmd_embed_bench(args) -> int:
    cfg = load_config(args.config)
    emb = make_embedder(cfg)
    texts = synthetic_texts(args.count)
    rows = []
    for w in [int(x) for x in args.workers.split(",")]:
        before = dict(emb.stats)
        emb.latencies_ms.clear()
        t0 = time.perf_counter()
        emb.embed_many(texts, workers=w, use_cache=False)
        dt = time.perf_counter() - t0
        lat = summarize_latency(emb.latencies_ms)
        rows.append(
            [
                w,
                round(args.count / dt, 1),
                lat.get("p50"),
                lat.get("p95"),
                lat.get("p99"),
                emb.stats["throttles"] - before["throttles"],
                emb.stats["retries"] - before["retries"],
            ]
        )
        log(f"  workers={w}: {rows[-1][1]} embeddings/s")
    print()
    print(table(rows, ["workers", "emb/s", "p50ms", "p95ms", "p99ms", "throttles", "retries"]))
    print("\nIf throttles climb as workers increase you've hit your Bedrock quota for this model/region;")
    print("use that emb/s figure to estimate how long embedding your full dataset will take.")
    return 0


def _checkpoint_path(args, tcfg: dict) -> str:
    safe = "".join(c if c.isalnum() or c in "-_" else "_" for c in f"{args.target}-{tcfg.get('index', 'mem')}")
    return os.path.join(".cache", f"ingest-{safe}.json")


def _fmt_secs(x: float) -> str:
    x = int(x)
    return f"{x // 3600}h{x % 3600 // 60:02d}m" if x >= 3600 else f"{x // 60}m{x % 60:02d}s"


def _ingest_dry_run(args, cfg, tcfg, wb, records) -> int:
    emb = wb.embedder
    texts = list(dict.fromkeys(r["text"] for r in records))
    cached = sum(1 for t in texts if emb.cache.has(emb._key(t)))
    todo = [t for t in texts if not emb.cache.has(emb._key(t))]
    chars = sum(len(t) for t in todo)
    dim = emb.dim
    log("DRY RUN: nothing was embedded, written or deleted.")
    log(f"  corpus cases:        {len(records)} ({len(texts)} unique texts)")
    log(f"  already cached:      {cached}   still to embed: {len(todo)}")
    log(f"  est. input tokens:   ~{chars // 4} (about 4 characters per token; the real count comes from Bedrock)")
    if args.rate:
        log(f"  est. embedding time: ~{_fmt_secs(len(todo) / args.rate)} at {args.rate} embeddings/s")
    else:
        log("  est. embedding time: pass --rate <emb/s> from `embed-bench` to get an estimate")
    log(f"  raw vector size:     ~{round(len(records) * dim * 4 / 1e6, 1)} MB in the index, same again in the local cache")
    log(f"  memory while ingest: flat (one chunk of {args.chunk} at a time)")
    mode = "append to" if args.append or args.resume else "DELETE and recreate"
    log(f"  index:               would {mode} '{tcfg.get('index', 'memory')}' on target '{args.target}'")
    if getattr(wb.store, "read_only", False):
        log("  NOTE: this target is read_only, so a real ingest would be refused.")
    return 0


def cmd_ingest(args) -> int:
    cfg, tcfg, wb = _setup(args)
    if getattr(wb.store, "read_only", False) and not args.dry_run:
        raise SystemExit(f"Target '{args.target}' is read_only; ingest would modify it. Use a copy of the index.")
    path = args.data or cfg["data"]["corpus"]
    records = read_jsonl(path, limit=args.limit)
    if args.dry_run:
        return _ingest_dry_run(args, cfg, tcfg, wb, records)

    ckpt = _checkpoint_path(args, tcfg)
    start = 0
    if args.resume:
        try:
            state = json.load(open(ckpt))
        except (OSError, ValueError):
            raise SystemExit(f"No checkpoint at {ckpt}; nothing to resume. Run without --resume.")
        if state.get("source") != path or state.get("total") != len(records):
            raise SystemExit(f"Checkpoint {ckpt} is for a different corpus ({state.get('source')}, {state.get('total')} cases).")
        start = int(state["done"])
        log(f"Resuming at case {start}/{len(records)}.")
    recreate = not (args.append or args.resume)
    log(f"Embedding and indexing {len(records) - start} cases into '{args.target}' "
        f"({wb.store.describe().get('index', 'memory')}) in chunks of {args.chunk}...")

    t_start = time.perf_counter()

    def on_chunk(done: int, total: int, stats: dict) -> None:
        os.makedirs(os.path.dirname(ckpt), exist_ok=True)
        json.dump({"source": path, "total": total, "done": done}, open(ckpt, "w"))
        elapsed = time.perf_counter() - t_start
        rate = (done - start) / elapsed if elapsed else 0
        eta = _fmt_secs((total - done) / rate) if rate else "?"
        log(f"  {done}/{total} ({round(100 * done / total)}%)  {round(rate, 1)} cases/s  ETA {eta}  "
            f"429s={wb.embedder.stats['throttles'] + stats['throttled']}")

    stats = wb.ingest_streaming(records, recreate=recreate, start=start, chunk_size=args.chunk, on_chunk=on_chunk)
    log(
        f"Bulk indexed {stats['indexed']} docs in {round(stats['seconds'], 1)}s "
        f"({stats.get('docs_per_sec')} docs/s), errors={stats['errors']}, 429s={stats['throttled']}, retries={stats['retries']}"
    )
    if stats["stopped_at"] is not None:
        log(f"STOPPED: indexing errors in the chunk starting at case {stats['stopped_at']}. "
            "Fix the cause, then continue with `ingest --resume` (same options).")
        return 1
    try:
        os.remove(ckpt)
    except OSError:
        pass
    t0 = time.perf_counter()
    deadline = t0 + float(cfg["run"]["visibility_timeout"]) * 3
    expected = start + stats["indexed"]
    n = 0
    while time.perf_counter() < deadline:
        try:
            n = wb.store.count()
        except Exception:
            n = 0
        if n >= expected:
            break
        time.sleep(2)
    if n >= expected:
        log(f"All {n} docs countable {round(time.perf_counter() - t0, 1)}s after the last bulk call.")
    else:
        log(f"WARNING: only {n}/{expected} docs countable after waiting; Serverless can lag, re-check later with `check`.")
    if tcfg.get("kind") == "local":
        log("(local target is in-memory: it is rebuilt automatically by each command, so this step is only a dry run)")
    return 0


def cmd_run(args) -> int:
    cfg, tcfg, wb = _setup(args)
    _load_corpus(wb, cfg, limit=args.limit)
    _ensure_populated(wb, tcfg, args.reingest)
    names = [s.strip() for s in args.scenarios.split(",") if s.strip()]
    if getattr(wb.store, "read_only", False):
        blocked = [n for n in names if n in ("new", "updated", "deleted")]
        if blocked:
            log(f"Target is read_only: skipping scenarios that write to the index: {', '.join(blocked)}")
            names = [n for n in names if n not in blocked]
    new_recs = read_jsonl(cfg["data"]["new_cases"]) if "new" in names else []
    upd_recs = read_jsonl(cfg["data"]["updates"]) if "updated" in names else []
    if args.max_new:
        new_recs = new_recs[: args.max_new]
    start_size = len(wb.records)
    results = run_scenarios(wb, names, new_recs, upd_recs, log=log)
    meta = {
        "timestamp": datetime.now().isoformat(timespec="seconds"),
        "target": args.target,
        "target_info": wb.store.describe(),
        "index": wb.store.describe().get("index"),
        "embedder": wb.embedder.model_id,
        "dim": wb.embedder.dim,
        "corpus_size": start_size,
        "k": wb.k,
        "embedder_stats": wb.embedder.summary(),
    }
    print_run(results, meta)
    path = _save(args.out, args.target, "run", {"meta": meta, "scenarios": results, "passed": all(r["passed"] for r in results)})
    log(f"\nSaved {path}")
    return 0 if all(r["passed"] for r in results) else 1


def cmd_load(args) -> int:
    cfg, tcfg, wb = _setup(args)
    _load_corpus(wb, cfg, limit=args.limit)
    _ensure_populated(wb, tcfg, args.reingest)
    log(f"Preparing {args.queries} query vectors (embedded once, outside the measured path)...")
    vectors = make_query_vectors(wb, args.queries)
    writer_docs = wb.docs()[:500] if args.writers else None
    stages = [int(x) for x in args.stages.split(",")]
    res = run_load(
        wb.store, vectors, stages, args.duration, args.warmup, wb.k, args.batch, cfg["thresholds"],
        writer_docs=writer_docs, writers=args.writers, pause=args.pause, log=log,
    )
    print_load(res)
    meta = {"timestamp": datetime.now().isoformat(timespec="seconds"), "target": args.target, "target_info": wb.store.describe(),
            "dim": wb.embedder.dim, "corpus_size": len(wb.records), "k": wb.k, "writers": args.writers}
    path = _save(args.out, args.target, "load", {"meta": meta, "load": res})
    log(f"\nSaved {path}")
    if args.writers:
        log("Note: writers re-index existing cases with unchanged content (version bumps only).")
    return 0


def cmd_match(args) -> int:
    cfg, tcfg, wb = _setup(args)
    _load_corpus(wb, cfg, limit=args.limit)
    _ensure_populated(wb, tcfg, args.reingest)
    path = args.queries or cfg["data"]["queries"]
    recs = read_jsonl(path)
    out_csv = args.csv or os.path.join(args.out, f"{_ts()}_{args.target}_matches.csv")
    log(f"Matching {len(recs)} records from {path} against the indexed corpus (top {wb.k})...")
    metrics = run_match(wb, recs, wb.k, args.min_sim, out_csv, batch=args.batch, workers=args.workers, log=log)
    print(json.dumps(metrics, indent=2))
    meta = {"timestamp": datetime.now().isoformat(timespec="seconds"), "target": args.target, "target_info": wb.store.describe()}
    p = _save(args.out, args.target, "match", {"meta": meta, "match": metrics})
    log(f"\nSaved {p}")
    return 0


def cmd_compare(args) -> int:
    with open(args.a, encoding="utf-8") as fh:
        a = json.load(fh)
    with open(args.b, encoding="utf-8") as fh:
        b = json.load(fh)
    la = args.label_a or a.get("meta", {}).get("target", "A")
    lb = args.label_b or b.get("meta", {}).get("target", "B")
    print(compare(a, b, la, lb))
    return 0


# -------------------------------------------------------------------------------------------------
def build_parser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser(prog="simtest", description="Similarity-search test framework (Bedrock Titan + OpenSearch)")
    p.add_argument("--version", action="version", version=__version__)
    sub = p.add_subparsers(dest="cmd", required=True)

    def common(sp, target=True):
        sp.add_argument("--config", "-c", default="config.yaml", help="config file (default: config.yaml)")
        if target:
            sp.add_argument("--target", "-t", required=True, help="target name from the config (e.g. managed, serverless, local)")
        sp.add_argument("--k", type=int, help="top-k (overrides config run.k)")
        sp.add_argument("--limit", type=int, help="only use the first N corpus cases (quick smoke tests)")
        sp.add_argument("--out", default="results", help="results directory")

    s = sub.add_parser("init-sample", help="generate synthetic corpus / new / updated / dataset-A files")
    s.add_argument("--out", default="data")
    s.add_argument("--corpus", type=int, default=2000)
    s.add_argument("--new", type=int, default=100)
    s.add_argument("--updates", type=int, default=100)
    s.add_argument("--dataset-a", type=int, default=200)
    s.add_argument("--seed", type=int, default=7)
    s.set_defaults(fn=cmd_init_sample)

    s = sub.add_parser("check", help="verify Bedrock + OpenSearch connectivity and credentials")
    common(s)
    s.set_defaults(fn=cmd_check)

    s = sub.add_parser("embed-bench", help="measure Bedrock Titan embedding throughput / throttling at several concurrency levels")
    s.add_argument("--config", "-c", default="config.yaml")
    s.add_argument("--count", type=int, default=200)
    s.add_argument("--workers", default="1,4,8,16")
    s.set_defaults(fn=cmd_embed_bench)

    s = sub.add_parser("ingest", help="embed the corpus and bulk-index it (recreates the index unless --append)")
    common(s)
    s.add_argument("--data", help="corpus JSONL (default: config data.corpus)")
    s.add_argument("--append", action="store_true", help="do not recreate the index")
    s.add_argument("--chunk", type=int, default=2000, help="cases embedded and indexed per step (memory stays flat)")
    s.add_argument("--resume", action="store_true", help="continue an interrupted ingest from its last completed chunk")
    s.add_argument("--dry-run", action="store_true", help="estimate tokens, time and size; embed, write and delete nothing")
    s.add_argument("--rate", type=float, help="embeddings/s from embed-bench, used by --dry-run for a time estimate")
    s.set_defaults(fn=cmd_ingest)

    s = sub.add_parser("run", help="run the scenarios: recall, existing, new, updated, deleted")
    common(s)
    s.add_argument("--scenarios", default=",".join(CANONICAL), help=f"comma list from: {', '.join(CANONICAL)}")
    s.add_argument("--n-queries", type=int, help="queries per scenario (overrides config run.n_queries)")
    s.add_argument("--max-new", type=int, help="limit the number of new cases used")
    s.add_argument("--reingest", action="store_true", help="recreate the index from the corpus before running")
    s.set_defaults(fn=cmd_run)

    s = sub.add_parser("load", help="stepped concurrency load test against the target")
    common(s)
    s.add_argument("--stages", default="8,16,32,64", help="client counts, comma separated")
    s.add_argument("--duration", type=float, default=60, help="seconds per stage (measured)")
    s.add_argument("--warmup", type=float, default=5, help="seconds of warm-up per stage (not measured)")
    s.add_argument("--batch", type=int, default=1, help="queries per request (>1 uses _msearch)")
    s.add_argument("--queries", type=int, default=500, help="distinct query vectors to cycle through")
    s.add_argument("--writers", type=int, default=0, help="concurrent writer threads (mixed read/write load)")
    s.add_argument("--pause", type=float, default=3.0, help="seconds between stages")
    s.add_argument("--reingest", action="store_true")
    s.set_defaults(fn=cmd_load)

    s = sub.add_parser("match", help="search every record of another dataset against the indexed corpus, write CSV")
    common(s)
    s.add_argument("--queries", help="dataset A JSONL (default: config data.queries)")
    s.add_argument("--min-sim", type=float, default=0.0, help="similarity threshold for the above_min_similarity column")
    s.add_argument("--batch", type=int, default=50, help="queries per _msearch request")
    s.add_argument("--workers", type=int, default=4)
    s.add_argument("--csv", help="output CSV path")
    s.add_argument("--reingest", action="store_true")
    s.set_defaults(fn=cmd_match)

    s = sub.add_parser("compare", help="compare two result JSON files side by side (e.g. managed vs serverless)")
    s.add_argument("a")
    s.add_argument("b")
    s.add_argument("--label-a")
    s.add_argument("--label-b")
    s.set_defaults(fn=cmd_compare)
    return p


def main(argv=None) -> int:
    args = build_parser().parse_args(argv)
    try:
        return int(args.fn(args) or 0)
    except KeyboardInterrupt:
        return 130


if __name__ == "__main__":
    sys.exit(main())
