# simtest: similarity-search test framework

Embeds text with **Amazon Bedrock Titan** and tests **similarity search** on an **OpenSearch Service domain** and an
**OpenSearch Serverless collection**, using the same data, queries and checks on both so the results are comparable.

It tests four kinds of cases, plus scale:

| Scenario | What it proves |
|---|---|
| `existing` | A case queried by its own text comes back at rank 1; a lightly re-worded version still comes back in the top-k |
| `new` | An incoming case finds its expected similar *existing* case **before** it is stored; once stored, it is searchable, and you get **time-to-searchable** |
| `updated` | After a case's text changes: the new content is found, the **old embedding is gone**, no stale or duplicate copies remain, plus time-to-searchable |
| `deleted` | Deleted cases stop appearing in results |
| `recall` | ANN recall@k against exact (brute-force) search, run **before and after** the changes above (updates and deletes can degrade HNSW graphs) |
| `load` (separate command) | Stepped concurrency ramp: QPS, p50/p95/p99, 429s and errors, optionally with concurrent writers, and the point where it degrades |
| `match` (separate command) | Searches every record of dataset A against dataset B (the indexed corpus), writes a CSV, and scores it if you supply expected matches |

## Setup

```bash
pip install -r requirements.txt
cp config.example.yaml config.yaml      # set your endpoints and regions
python -m simtest init-sample           # synthetic data to try it; replace with your own JSONL later
```

Dry run with no AWS at all (hash embedder, in-memory target):

```bash
python -m simtest run -c config.local.yaml -t local
```

### Permissions

* **Bedrock:** `bedrock:InvokeModel` on `arn:aws:bedrock:<region>::foundation-model/amazon.titan-embed-text-v2:0`, and the model must be available to your account in that region.
* **Managed domain:** the IAM identity must be allowed by the domain access policy (`es:ESHttp*`) and, with fine-grained access control, mapped to a backend role that can create/delete the index and index, search and delete documents. Or use `auth: basic` with `OS_USER` / `OS_PASS`.
* **Serverless:** `aoss:APIAccessAll` on the collection, a **data access policy** granting the index permissions (`aoss:CreateIndex`, `aoss:DeleteIndex`, `aoss:UpdateIndex`, `aoss:DescribeIndex`, `aoss:ReadDocument`, `aoss:WriteDocument`) and a **network policy** that allows the machine running the test.
* Run the load generator on EC2 in the same region as the targets, and give it enough CPU and network (see "Scaling up").

## Running against an existing index in a corporate network

1. Log in with your SAML tool and point at the profile it writes (`profile:` on both `embedder` and the target, or `AWS_PROFILE`). Temporary credentials are re-read every `credential_refresh_secs` (OpenSearch) and on `ExpiredToken` (Bedrock); a profile using `credential_process` refreshes most reliably.
2. Use the `corp` target in `config.example.yaml`: set `ca_certs` for an internal CA, `HTTPS_PROXY`/`NO_PROXY` if you go through a proxy, and the field names of the existing mapping (`vector_field`, `id_field`, ...).
3. Use the same Titan model, `dimensions` and `normalize` that built the index. Otherwise results are meaningless.
4. `read_only: true` makes the framework refuse to recreate the index or write/delete documents. `run` then skips `new`, `updated` and `deleted`. Run `check`, then `match`, `run --scenarios existing,recall`, and `load` only with agreement from the cluster owners.
5. `corpus.jsonl` must be (a subset of) the data in the index. The recall ground truth comes from re-embedding it locally.
6. For write scenarios, test against a **copy** of the index with `read_only: false`.

## Workflow

```bash
python -m simtest check        -t managed        # Bedrock + OpenSearch connectivity and credentials
python -m simtest embed-bench                    # Bedrock embeddings/s and throttling at 1/4/8/16 workers

python -m simtest ingest -t managed              # embed corpus (cached) + bulk index + time until countable
python -m simtest run    -t managed --reingest   # recall, existing, new, updated, deleted, recall again
python -m simtest load   -t managed --stages 8,16,32,64,128 --duration 120
python -m simtest match  -t managed --queries data/dataset_a.jsonl --k 5 --min-sim 0.8

# same again for Serverless, then compare
python -m simtest run    -t serverless --reingest
python -m simtest compare results/<managed run>.json results/<serverless run>.json
```

* `run` and `load` change the index (new/updated/deleted cases; writer threads), so pass `--reingest` to start each
  run from the known corpus. Without it, `run` refuses to start if the index doesn't match the corpus file.
* `run` exits non-zero if any check fails, so it works in CI. Thresholds live under `thresholds:` in the config.
* Every command writes JSON to `results/`; nothing is overwritten.
* `--scenarios existing,new` runs a subset; `--limit 5000` uses only the first 5,000 corpus cases; `--k` and `--n-queries` override the config.

## Your own data (JSONL, one object per line)

```jsonc
// corpus.jsonl: existing cases (any extra fields are stored as metadata)
{"id": "case-1", "text": "Payment gateway times out when applying a coupon ...", "status": "open"}
// new_cases.jsonl: expected_neighbor (optional) is an existing id that should show up as similar
{"id": "new-1", "text": "...", "expected_neighbor": "case-1"}
// updates.jsonl: an existing id with its new text
{"id": "case-1", "text": "... revised description ..."}
// dataset_a.jsonl: the second dataset for `match`; expected_match (optional) is a corpus id, or null for "no match"
{"id": "a-1", "text": "...", "expected_match": "case-1"}
```

Titan embeds the `text` field; put everything that should drive similarity in it.

## Reading the results

* **`recall_at_k`**: fraction of the true top-k (exact search) the engine returned. Low recall means tune `m`, `ef_construction`, `ef_search`, or use a larger `k`. A fast query with poor recall isn't a win. If `recall_after_changes` is lower than `recall_baseline`, updates and deletes are degrading the index.
* **`time_to_searchable_*`**: seconds from a write returning until a search finds it. Compare managed and Serverless here; Serverless is not immediate. `visibility_timeouts` counts writes that never appeared within `visibility_timeout`.
* **`old_vector_served` / `duplicates` / `stale_versions`** must be 0. Non-zero means an update left the old copy searchable. (Self-retrieval checks are tie-aware, so exact-duplicate cases don't count as misses.)
* **`load`**: `qps`, latency percentiles measured on the client, and `srv p50/p99` (the engine's own `took`). The verdict names the last healthy stage and the first degraded one, judged against `load_max_p99_ms` and `load_max_error_rate`. Rising 429s mean throttling (Serverless OCU limits, or write-queue rejection on a domain). Check CloudWatch alongside (`SearchOCU`/`IndexingOCU` for Serverless; CPU, JVM memory pressure, `KNNGraphMemoryUsage` and thread-pool rejections for a domain).
* Query vectors for `load` are embedded before the test, so Bedrock is not in the measured path.

## Scaling up

* **Embedding is usually the first bottleneck.** `embed-bench` shows your Bedrock quota ceiling in embeddings/s; divide your corpus size by it to estimate ingest time. Embeddings are cached on disk (`.cache/`), so each text is embedded once across all runs and both targets. For tens of millions of texts, use Bedrock batch inference outside this tool and feed the vectors in.
* **To test the engines at volume without paying for embeddings,** set `embedder.type: hash` with `dimensions: 1024`. It is lexical rather than semantic, so use it for throughput, memory and latency tests, and Titan for quality tests.
* **Memory:** the framework keeps an exact mirror of the corpus in RAM for ground truth (`corpus × dimensions × 4 bytes`, e.g. about 4 GB for 1M × 1024). Beyond that, use `--limit` or a sampled corpus for the quality scenarios and run `load` on its own.
* **Load generator limits:** this is a threaded Python client, good for roughly a few thousand QPS per process. Past that, run several instances in parallel and add their QPS, or use OpenSearch Benchmark in distributed mode.
* **Serverless:** raise the account's maximum OCU limits before a large test, or you measure the cap. Early stages run while it scales up, so use a longer `--warmup` and compare later stages. It bills per OCU-hour, so a long test costs real money.
* **Managed:** size shards for the data (aim for tens of GB per shard), and watch HNSW graph memory, since the graph lives off-heap. Consider `field_extra` (for example `data_type: byte` or on-disk mode) if it doesn't fit.

## What has and hasn't been verified

Verified offline (38 automated tests, `python -m pytest`): the Titan request format, retry on throttling, caching,
index body for managed vs Serverless, bulk/msearch/upsert/delete logic, 429 retries, the visibility-lag handling,
all scenarios against an in-memory emulator of the OpenSearch API (including a deliberately broken variant to confirm
the stale-copy check really fails), the load test and the CLI.

**Not yet run against real AWS** (no credentials were available when this was built). The Serverless behaviours below
are written from documented behaviour and are the first things to check on your first real run:

* `id_mode: auto` is the safe default for Serverless vector collections. If yours accepts custom ids, set `id_mode: custom` (simpler and faster updates).
* The index-level `knn.algo_param.ef_search` setting may be rejected on Serverless; set `ef_search: null` and use `query_ef_search` instead (needs OpenSearch 2.16+).
* Serverless has no `_refresh`; the framework polls instead, so `ingest` reports how long until all documents are countable.
* Index creation can take a few seconds to become usable on Serverless (`post_create_wait`, default 10s).
* `score_to_similarity` is a readability aid for `innerproduct` / `cosinesimil`. Every pass/fail check uses ids, never that value.

Start with `check`, then `run --limit 500 --reingest` on each target, before running anything large.
