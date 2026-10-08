"""JSONL helpers, text perturbation, and a synthetic support-case generator for dry runs.

Real data format (one JSON object per line):
    {"id": "case-1", "text": "free text that will be embedded", ...optional metadata...}
new_cases.jsonl may add   "expected_neighbor": "<existing id>"   (used to score 'find similar existing case')
updates.jsonl   is        {"id": "<existing id>", "text": "<new text>"}
dataset_a.jsonl may add   "expected_match": "<corpus id>" | null (used to score the `match` command)
"""
from __future__ import annotations

import json
import os
import random
from typing import Iterable


def read_jsonl(path: str, limit: int | None = None) -> list[dict]:
    out = []
    with open(path, "r", encoding="utf-8") as fh:
        for line in fh:
            line = line.strip()
            if not line:
                continue
            out.append(json.loads(line))
            if limit and len(out) >= limit:
                break
    return out


def write_jsonl(path: str, records: Iterable[dict]) -> int:
    os.makedirs(os.path.dirname(os.path.abspath(path)), exist_ok=True)
    n = 0
    with open(path, "w", encoding="utf-8") as fh:
        for r in records:
            fh.write(json.dumps(r, ensure_ascii=False) + "\n")
            n += 1
    return n


def perturb_text(text: str, rng: random.Random, drop: float = 0.15) -> str:
    """Drop a fraction of words (never the first/last) to simulate a partially re-worded query."""
    words = text.split()
    if len(words) <= 4:
        return text
    kept = [w for i, w in enumerate(words) if i in (0, len(words) - 1) or rng.random() > drop]
    if len(kept) < 3:
        return text
    return " ".join(kept)


# ------------------------------------------------------------------------------------------------
# Synthetic cases
# ------------------------------------------------------------------------------------------------
_PRODUCTS = [
    "billing portal", "mobile app", "payment gateway", "search service", "report builder", "data sync agent",
    "identity provider", "notification service", "admin console", "export tool", "api gateway", "job scheduler",
    "document viewer", "audit logger", "checkout flow", "inventory service",
]
_COMPONENTS = [
    "login page", "upload dialog", "invoice generator", "webhook handler", "cache layer", "pdf renderer",
    "token refresh", "email queue", "csv importer", "dashboard widget", "rate limiter", "session manager",
    "retry worker", "settings panel",
]
_SYMPTOMS = [
    ("times out", "hangs and eventually times out"),
    ("returns a blank screen", "shows nothing but an empty page"),
    ("crashes", "terminates unexpectedly"),
    ("duplicates records", "creates the same record twice"),
    ("drops messages", "loses some messages silently"),
    ("shows stale data", "keeps displaying outdated information"),
    ("rejects valid input", "refuses input that should be accepted"),
    ("runs very slowly", "has severe performance degradation"),
]
_TRIGGERS = [
    ("uploading a large file", "a big file upload is attempted"),
    ("running a nightly batch", "the overnight batch job starts"),
    ("switching accounts", "the user changes to another account"),
    ("applying a coupon", "a discount code is applied"),
    ("restoring a backup", "a backup restore is started"),
    ("enabling two-factor login", "multi-factor authentication is turned on"),
    ("exporting a monthly report", "the month-end report export runs"),
]
_ENVS = [
    ("production", "the prod environment"), ("staging", "the staging environment"),
    ("the EU region", "the european region"), ("the on-prem install", "the self-hosted deployment"),
]
_WORKAROUNDS = [
    ("restarted the service", "a service restart"), ("cleared the cache", "flushing the cache"),
    ("rolled back the release", "reverting to the previous version"), ("re-ran the job", "running the job again"),
    ("increased the timeout", "raising the timeout value"),
]


def _fields(rng: random.Random, serial: int) -> dict:
    return {
        "product": rng.choice(_PRODUCTS),
        "component": rng.choice(_COMPONENTS),
        "sym": rng.choice(_SYMPTOMS),
        "trig": rng.choice(_TRIGGERS),
        "env": rng.choice(_ENVS),
        "wk": rng.choice(_WORKAROUNDS),
        "code": f"ERR-{rng.randint(1000, 9999)}-{serial}",
    }


def _text_a(f: dict) -> str:
    return (
        f"{f['product']}: the {f['component']} {f['sym'][0]} with error {f['code']} when {f['trig'][0]} "
        f"in {f['env'][0]}. Workaround tried: {f['wk'][0]}."
    )


def _text_b(f: dict) -> str:
    return (
        f"In {f['env'][1]}, {f['trig'][1]} and the {f['component']} of the {f['product']} {f['sym'][1]} "
        f"({f['code']}). The customer attempted {f['wk'][1]}."
    )


def synthetic_texts(n: int, seed: int = 99) -> list[str]:
    """Unique, realistic-length texts (used by the embedding throughput benchmark)."""
    rng = random.Random(seed)
    return [_text_a(_fields(rng, 30_000_000 + i)) for i in range(n)]


def generate_sample(
    out_dir: str, n_corpus: int = 2000, n_new: int = 100, n_updates: int = 100, n_dataset_a: int = 200, seed: int = 7
) -> dict:
    rng = random.Random(seed)
    corpus_fields = [_fields(rng, i) for i in range(n_corpus)]
    corpus = [
        {"id": f"case-{i:06d}", "text": _text_a(f), "status": rng.choice(["open", "closed", "pending"])}
        for i, f in enumerate(corpus_fields)
    ]

    # new cases: ~70% are re-worded copies of an existing case (expected neighbor), ~30% genuinely novel
    new = []
    paraphrase_sources = rng.sample(range(n_corpus), min(n_new, n_corpus))  # distinct, so no two new cases share text
    for i in range(n_new):
        if rng.random() < 0.7 and i < len(paraphrase_sources):
            j = paraphrase_sources[i]
            new.append({"id": f"new-{i:06d}", "text": _text_b(corpus_fields[j]), "expected_neighbor": corpus[j]["id"]})
        else:
            new.append({"id": f"new-{i:06d}", "text": _text_a(_fields(rng, n_corpus + i)), "expected_neighbor": None})

    # updates: an existing case gets materially different content (new product/component/symptom, resolution note)
    update_ids = rng.sample(range(n_corpus), min(n_updates, n_corpus))
    updates = []
    for u, j in enumerate(update_ids):
        f = _fields(rng, 10_000_000 + u)
        updates.append({"id": corpus[j]["id"], "text": _text_a(f) + " Resolution: fixed in patch release, customer confirmed."})

    # dataset A: the 'other' dataset for corpus-vs-dataset matching; ~80% have a true match in the corpus
    dataset_a = []
    for i in range(n_dataset_a):
        if rng.random() < 0.8:
            j = rng.randrange(n_corpus)
            dataset_a.append({"id": f"a-{i:06d}", "text": _text_b(corpus_fields[j]), "expected_match": corpus[j]["id"]})
        else:
            dataset_a.append(
                {"id": f"a-{i:06d}", "text": _text_a(_fields(rng, 20_000_000 + i)), "expected_match": None}
            )

    counts = {
        "corpus": write_jsonl(os.path.join(out_dir, "corpus.jsonl"), corpus),
        "new_cases": write_jsonl(os.path.join(out_dir, "new_cases.jsonl"), new),
        "updates": write_jsonl(os.path.join(out_dir, "updates.jsonl"), updates),
        "dataset_a": write_jsonl(os.path.join(out_dir, "dataset_a.jsonl"), dataset_a),
    }
    return counts
