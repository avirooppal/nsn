# NSN

Lightweight, self-hosted memory for small language models and AI agents. NSN stores history locally and retrieves relevant prior evidence into an existing model's context.

```python
import nsn
nsn.init("./agent-memory")
model = nsn.wrap(existing_model)
```

`existing_model` is your model callable or supported client. NSN retrieves prior memory before recording the current input, calls the model, and stores the turn locally in SQLite. Memory survives process restarts. History can exceed the model's context window; only a bounded evidence pack enters each call. NSN does not change the model's weights, reasoning ability or actual context-window size.

The default memory layer needs no cloud key, model download, GPU or daemon. Your model runs wherever you configure it. `init("")` uses `./agent-memory`; the argument is a storage directory, not an API key.

## Install and verify

Install the verified local SDK candidate from the repository root:

```powershell
python -m pip install dist/memory-verification-20261004/nsn-0.3.0-py3-none-any.whl
python examples/restart_demo.py
```

Wheel SHA-256: `423370a5b09e7104627dbad911e39c6a8b3f8c0c974d2dc6799ed6e726af6f3b`. This artifact has been installed and tested outside the checkout. Earlier wheels remain available with their original reports.

For development from source, use `python -m pip install -e ".[dev]"`. Building a new wheel with `python -m pip install build` and `python -m build --wheel` creates a new artifact; verify it separately rather than applying the candidate's checksum to it.

The restart demo checks evidence injection deterministically. A real local Qwen model also recalled a stored token before and after restarting NSN in the [installed verification run](docs/implementation/results/10-memory-verification-20261004/report.md); this was a single integration smoke, not an accuracy benchmark.

Use a fixed data path in services and call `nsn.close()` at shutdown or before changing directories. Closed runtimes reject memory and maintenance operations. `nsn.flush()` reports pending indexing consistency; inspect failures instead of assuming success.

[Quickstart, adapters, profiles and migration guide](docs/user-guide.md) includes a real local-model example, offline assets, advanced commands and backup/restore. [Phase status](docs/implementation/README.md) records verified behavior and limitations.

## Supported profiles

| Profile | Setup | Scope |
| --- | --- | --- |
| Minimal | `pip install` the wheel | Scoped durable events, BM25, typed temporal facts, bounded evidence packs |
| Semantic | `nsn[semantic]`, explicitly prepared encoder assets | One pooled vector per item, exact search and rank fusion; no FAISS/spaCy dependency |
| Semantic ONNX | `nsn[semantic-onnx]`, explicitly prepared local ONNX assets and encoder | Optional CPU mean pooling without PyTorch; separate asset fingerprint |
| Legacy ML | `nsn[legacy]` | Compatibility dependencies for older token-vector `Memory` interfaces |
| Relationships | `nsn.init(path, graph=True)` | Explicit source-backed bounded paths; no graph server |
| Consolidation | Explicit `nsn.sleep()` | Recoverable bounded extractive jobs; originals retained, summaries off by default |
| Procedures | Explicit runtime commands | Source-backed host-validated outcomes and retrieval; steps never auto-execute |
| REST / LangChain | `nsn[api]` / `nsn[langchain]` | Optional host-authenticated API or supported text Runnable/history adapters |

Namespaces isolate records within a trusted host process. Bind separate agents or tenants with `nsn.wrap(existing_model, namespace="agent-name")`. Session filters constrain events; structured facts remain namespace-wide. Remote API use needs host-configured credentials and TLS. Stored content is evidence, not authenticated instructions.

Semantic writes index inline by default. Advanced callers can explicitly choose
`defer_indexing=True`: events and indexing intent are durable immediately, while
semantic recall waits for `runtime.process_index_jobs()` or successful `flush()`.
Lexical recall remains available before vector replay. No worker starts automatically.

## Local container fixture

```powershell
docker compose run --rm nsn
```

This builds the minimal installed package and runs the offline restart fixture with a named persistent `/data` volume and runtime networking disabled. No model is baked into the image. The embedded library remains the primary interface. Semantic users prepare assets separately and mount them read-only; no automatic startup download is enabled.

## Evidence and release status

**Core memory behavior is verified as a working prototype.** The latest functional run passed **300 tests**, with two skips and one hardware performance test excluded. Eighteen new memory-contract tests cover cross-process persistence, namespace separation, recall from history larger than the input window, budgets, corrections, provenance deletion, historical/session filtering, backup/restore and indexing recovery. Fresh minimal and ONNX installations passed offline checks. Separate installed Windows/Linux Python 3.9/3.10/3.12 checks each passed 100 core tests; an initial Windows optional-enrichment failure remains recorded alongside its passing separate confirmation.

Run functional checks from the repository root after installing development dependencies:

```powershell
python -m pytest tests/ benchmarks/tests/ -m "not performance" -q -p no:cacheprovider
```

Hardware qualification is separate: excluding it from functional checks does not mark performance targets passed. [Verification report and exact commands](docs/implementation/results/10-memory-verification-20261004/report.md) retain skips, exclusions, the compatibility failure, raw results and artifact checksums.

Phases 00–08 are implemented. Phase 09 is partial: checksum-pinned full LongMemEval S/M and LoCoMo data are prepared, frozen holdout generation/resume works, and real local roughly 2B/4B/8B development evidence exists. Independent local grading and frozen human-review support exist, but full grading, public-model ablations and comparable competitor runs remain unfinished. [Evaluation handoff](docs/implementation/handoffs/09.md) records exact coverage and excludes historical self-judge scores. Memory does not transfer frontier model weights or general reasoning, and competitor leadership has not been established.

Phase 10 prepares a local release candidate: installed-wheel checks on Windows/Linux Python 3.9/3.10/3.12, fresh semantic dependency resolution, offline operation, resources, migration/recovery documentation and checksums. Hosted GitHub Actions remains unexecuted. [Release handoff](docs/implementation/handoffs/10.md) distinguishes repeated-query and uncached timings and records remaining resource gates. No package was published.

Full production qualification remains incomplete: previous latency/resource misses, full public and competitor evaluations, human grading, hosted CI and the owner's license decision remain open. Latest candidate performance has not been requalified. See the [operational prototype guide](docs/implementation/prototype-readiness.md) and [completion plan](docs/implementation/completion-plan.md).

Historical prototype charts are retained in [the archived README](docs/historical/README-prototype.md) solely for audit. They are not current accuracy or performance evidence. Plan resource/quality numbers remain goals unless a linked measured run establishes them.

## Repository layout

| Path | Purpose |
| --- | --- |
| `nsn/` | Public API, runtime and model adapters |
| `neurosleepnet/` | Storage, facts, retrieval and optional memory components |
| `tests/`, `benchmarks/tests/` | Functional and evaluation-integrity tests |
| `examples/`, `scripts/` | Usage examples and installation/compatibility checks |
| `benchmarks/` | Evaluation runners, datasets and competitor support |
| `docs/implementation/` | Phase briefs, handoffs and reproducible evidence |
| `dist/` | Preserved local wheel candidates |

`build/`, `__pycache__/` and `.pytest_cache/` are disposable generated output. Memory databases, model assets and evaluation artifacts contain state or evidence; do not treat them as caches. `nsn.egg-info/` may support an active editable installation.

Retired dashboards, the old standalone cloud server, paper drafts/figures and superseded root benchmark scripts have been removed from the repository. The supported optional REST factory remains in `neurosleepnet/integrations/api.py`. Dependencies are defined in `pyproject.toml`; the obsolete root ML requirements file has been removed. [Cleanup inventory and validation](docs/implementation/results/10-repository-cleanup-20261004/retired-report.md) records the recovery copy and preserved functionality.

## Compatibility and distribution

The minimal runtime uses schema 7. Original evidence, history and namespace rules are retained; migrations reject newer unsupported schemas. See the migration guide before moving legacy databases. The older `neurosleepnet.Memory` interface requires optional legacy ML dependencies and does not automatically share the new facade's event history.

The repository has no declared NSN redistribution license. [Distribution notice](NOTICE.md) records this open release decision and third-party/data attribution boundaries. Choose a license before public distribution. Evaluation datasets and model weights are not bundled in the wheel.
