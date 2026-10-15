import json
import os
import subprocess
import sys

import yaml

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))


def sh(args, cwd=ROOT, expect=0):
    p = subprocess.run([sys.executable, "-m", "simtest", *args], cwd=cwd, capture_output=True, text=True, timeout=300)
    assert p.returncode == expect, f"exit {p.returncode}\nSTDOUT:\n{p.stdout[-3000:]}\nSTDERR:\n{p.stderr[-3000:]}"
    return p.stdout


def test_cli_end_to_end_on_local_target(tmp_path):
    data, results = tmp_path / "data", tmp_path / "results"
    cfg = {
        "embedder": {"type": "hash", "dimensions": 128, "cache_path": str(tmp_path / "cache.sqlite")},
        "targets": {"local": {"kind": "local"}},
        "data": {
            "corpus": str(data / "corpus.jsonl"),
            "new_cases": str(data / "new_cases.jsonl"),
            "updates": str(data / "updates.jsonl"),
            "queries": str(data / "dataset_a.jsonl"),
        },
        "run": {"n_queries": 30, "visibility_sample": 4, "delete_sample": 4},
        "thresholds": {"new_neighbor_hit_at_k": 0.3},  # hash embedder is lexical, not semantic
    }
    cfg_path = tmp_path / "config.yaml"
    cfg_path.write_text(yaml.safe_dump(cfg))
    common = ["-c", str(cfg_path), "-t", "local", "--out", str(results)]

    sh(["init-sample", "--out", str(data), "--corpus", "200", "--new", "20", "--updates", "10", "--dataset-a", "30"])
    assert (data / "corpus.jsonl").exists()

    assert "Embedder OK" in sh(["check", *common[:4]])
    assert "emb/s" in sh(["embed-bench", "-c", str(cfg_path), "--count", "20", "--workers", "1,2"])
    assert "Bulk indexed 200 docs" in sh(["ingest", *common])

    out = sh(["run", *common])
    assert "OVERALL: PASS" in out
    run_files = sorted(f for f in os.listdir(results) if f.endswith("_run.json"))
    doc = json.load(open(results / run_files[-1]))
    assert doc["passed"] is True and [s["name"] for s in doc["scenarios"]][0] == "recall_baseline"
    assert doc["meta"]["corpus_size"] == 200

    out = sh(["run", *common, "--scenarios", "existing,updated", "--n-queries", "10"])
    assert len([f for f in os.listdir(results) if f.endswith("_run.json")]) == 2  # second run didn't overwrite the first
    assert "[PASS] existing" in out and "[PASS] updated" in out and "recall_baseline" not in out

    out = sh(["load", *common, "--stages", "2,3", "--duration", "0.3", "--warmup", "0", "--pause", "0", "--queries", "20"])
    assert "Peak QPS" in out

    out = sh(["match", *common, "--k", "3"])
    assert '"rows_written": 90' in out
    assert any(f.endswith("_matches.csv") for f in os.listdir(results))

    run_a = str(results / run_files[0])
    assert "recall_baseline.recall_at_k" in sh(["compare", run_a, run_a, "--label-a", "x", "--label-b", "y"])


def test_cli_run_exits_nonzero_when_a_check_fails(tmp_path):
    data = tmp_path / "data"
    cfg = {
        "embedder": {"type": "hash", "dimensions": 128, "cache_path": None},
        "targets": {"local": {"kind": "local"}},
        "data": {"corpus": str(data / "corpus.jsonl"), "new_cases": str(data / "new_cases.jsonl"), "updates": str(data / "updates.jsonl")},
        "run": {"n_queries": 20},
        "thresholds": {"existing_exact_hit_at_1": 1.5},  # impossible on purpose
    }
    cfg_path = tmp_path / "c.yaml"
    cfg_path.write_text(yaml.safe_dump(cfg))
    sh(["init-sample", "--out", str(data), "--corpus", "100", "--new", "5", "--updates", "5", "--dataset-a", "5"])
    out = sh(["run", "-c", str(cfg_path), "-t", "local", "--out", str(tmp_path / "r"), "--scenarios", "existing"], expect=1)
    assert "[FAIL] existing" in out and "OVERALL: FAIL" in out


def test_sweep_and_report_end_to_end(tmp_path):
    data = tmp_path / "data"
    cfg = {
        "embedder": {"type": "hash", "dimensions": 64, "cache_path": str(tmp_path / "cache.sqlite")},
        "targets": {"local": {"kind": "local"}},
        "data": {"corpus": str(data / "corpus.jsonl"), "new_cases": str(data / "new_cases.jsonl"), "updates": str(data / "updates.jsonl")},
        "run": {"n_queries": 20},
    }
    cfg_path = tmp_path / "c.yaml"
    cfg_path.write_text(yaml.safe_dump(cfg))
    sh(["init-sample", "--out", str(data), "--corpus", "100", "--new", "5", "--updates", "5", "--dataset-a", "5"])
    out = tmp_path / "sw"
    text = sh(["sweep", "-c", str(cfg_path), "-t", "local", "--vary", "embedder.dimensions=64,128", "--limit", "100",
               "--stages", "2", "--duration", "0.2", "--warmup", "0", "--pause", "0", "--queries", "10", "--out", str(out)])
    assert "Sweep done: 4 result files" in text
    md = (out / "sweep_report.md").read_text()
    assert "embedder.dimensions=64" in md and "embedder.dimensions=128" in md and "## Load test" in md
    assert (out / "sweep_report.html").exists() and (out / "sweep.json").exists()

    # the standalone report command accepts the same folder and honours --format
    sh(["report", str(out), "--format", "md", "--out", str(tmp_path / "again")])
    assert (tmp_path / "again.md").exists() and not (tmp_path / "again.html").exists()

    plan_only = sh(["sweep", "-c", str(cfg_path), "-t", "local", "--vary", "m=16,32", "--dry-run"])
    assert "2 points" in plan_only
