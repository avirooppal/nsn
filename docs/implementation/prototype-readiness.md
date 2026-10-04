# NSN prototype readiness

NSN is a self-hosted memory layer: durable history, scoped retrieval, facts and bounded context for an existing model or agent. It does not add model intelligence or enlarge the model's actual context window. It stores history beyond that window and retrieves selected evidence.

## Candidate and supported pilot scope

Use the local, unpublished wheel `dist/memory-verification-20261004/nsn-0.3.0-py3-none-any.whl`; its SHA-256, tests and installed checks are in [the memory verification report](results/10-memory-verification-20261004/report.md). The earlier f02 and ce32 wheels and their benchmark artifacts are preserved. Resource measurements below belong to f02, not this new candidate. No publication or license selection occurred.

Recommended starting profile: **minimal**, one host, local SQLite directory, English-first retrieval, bounded packs, explicit namespace per agent/tenant. This candidate is a strengthened prototype for controlled use, not a claim that all production release gates are complete. The measured workload is 10,000 short events; large documents, 100k/1M records, multilingual recall and high-concurrency service throughput remain unqualified. Optional graph/consolidation/procedures are explicit and do not run automatically.

```python
import nsn
nsn.init("./agent-memory")
model = nsn.wrap(existing_model)
```

For a service, use a fixed absolute data path. Bind wrappers to the intended `namespace`, supply the real model's tokenizer/chat counter and context/output allowance where available, and leave semantic mode off until it demonstrates benefit on the application's own questions. A memory pack is evidence, not a model instruction. Escaping preserves its delimiter boundary; it cannot guarantee that a model will ignore hostile natural-language content.

## Operation and recovery

1. Install the exact wheel and verify its checksum. Run `python examples/restart_demo.py` before directing real workloads at the directory. Installed-wheel verification has also denied network access and checked persistence and namespace isolation.
2. Keep writers on the same host and local filesystem. Namespace separation is a logical boundary; a remote host must enforce its own authenticated identity mapping. The optional REST factory already requires host bearer-token mappings; do not expose a bare SDK directory remotely.
3. Inspect `nsn.get_default_runtime().stats(namespace=...)` and pending/failed jobs rather than assuming every derived index is up to date. Monitor disk growth and the application's retrieval latency. Storage failures now surface as errors; a missing/corrupt FTS index no longer masquerades as an empty successful recall.
4. Back up using the guide's namespace export or SQLite online backup procedure, including committed WAL data. Restore into an empty separate namespace/directory and verify records before switching the application. Retain the previous candidate and backup during upgrades. See [backup/migration instructions](../user-guide.md#migration-and-backuprecovery).
5. Finish or close model streams before `nsn.close()`. Closing is idempotent; public append/retrieval/fact operations reject a closed runtime. Reinitialize before using memory again. Existing cancelled-stream and durable-job recovery checks remain in the regression suite.

Optional CPU semantic profile: explicitly provision pinned ONNX assets, install `[semantic-onnx]`, pass the prepared encoder and use `defer_indexing=True` only when asynchronous indexing is acceptable. Durable events remain lexically searchable immediately; call `process_index_jobs(worker_id="indexer", limit=64)` explicitly to replay pending vectors and inspect failures/`flush()`. Deferred append measurements do not qualify default inline encoding. No cloud model, key or daemon is required for memory operation.

## Evidence and remaining gates

The earlier seven hardening regressions remain. Eighteen additional memory-contract tests check persistence, bounded recall beyond the model window, corrections, recovery, historical/session filtering and lifecycle guards. The unchanged three-line API has **300 functional tests passing**, two skips and one performance deselection. Separate installed Windows/Linux Python 3.9/3.10/3.12 runs each pass **100 tests**, with two skips and two explicit minimal-profile exclusions. The first concurrent Windows 3.9 optional-enrichment failure is retained; its root cause remains unresolved despite a passing separate confirmation. Fresh installed ONNX tests pass **23 tests**, including prepared real-model offline checks. An actual local Qwen 1.7B recalls a stored nonce before and after memory restart. These checks establish mechanics, not general answer accuracy. [Exact evidence](results/10-memory-verification-20261004/report.md).

Historical retrieval filters raw events by event_time (otherwise observed_at), before limits. Session filters apply to events; facts remain namespace-wide. Use namespaces for tenant isolation. Optional semantic filtering now applies session/time eligibility before top-k; large-scale scoped-filter performance is not qualified.

A checksum-verified memory-only before/after diagnostic reproduces the earlier selected evidence exactly before applying fixes. In that exploratory 15-question sample, minimal LoCoMo evidence recall improves 11.1% to 22.2%, and LongMemEval recall improves 8.3% to 33.3%. Semantic recall is unchanged. This is not a new answer-accuracy experiment or confirmatory quality result. [Raw diagnostic](results/10-prototype-hardening-20261004/memory-probe.json).

Historical f02 resource gates, including failures, are recorded in [readiness.json](results/10-prototype-hardening-20261004/readiness.json). That candidate's minimal fresh-query latency misses the original 25ms target. The separate unchanged synthetic hardware test failed its 1k-vector 10ms median assertion (10.34ms measured); its failure remains recorded. The latest candidate has not been requalified for performance. Default PyTorch/inline semantic qualification is not replaced by ONNX/deferred measurements. Full public evaluation, human grading calibration, actual competitor QA, public feature ablations, hosted CI and the owner's source-license choice remain open. See [Phase 09](handoffs/09.md), [Phase 10](handoffs/10.md) and [the completion plan](completion-plan.md). These remaining gates prohibit a full production-ready or superiority claim; they do not erase the verified functional prototype.
