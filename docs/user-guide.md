# NSN local release candidate guide

## Three-line integration and persistence

Install the built wheel, create your model as usual, then:

```python
import nsn
runtime = nsn.init('./agent-memory')
model = nsn.wrap(your_model, namespace='agent')
```

The directory contains `memory.db` and SQLite sidecars. `init('')` resolves to `./agent-memory` relative to the process working directory. Use a fixed absolute path for applications/services. Close with `nsn.close()` before changing the active directory; repeated initialization of the same active directory returns the existing runtime. Explicit `Runtime(path)` instances can be passed to `wrap(..., runtime=runtime)`.

Supported calls retrieve prior memory before recording current input, call your model, then record assistant output. Assistant text is not confirmed truth. Failed and cancelled outputs have explicit lifecycle statuses. `nsn.flush()` is the consistency barrier for pending index work; `False` means unresolved work, not success. Closing is idempotent.

Run `python examples/restart_demo.py` for an offline deterministic fixture that closes/reopens the runtime and checks prior recall. It does not claim language-model intelligence. A second application process can reopen the same persistent directory.

## Adapter contract

| Input model | Intercepted calls | Limits |
| --- | --- | --- |
| Text callable | `model(text)`, `invoke(text)`, `ainvoke(text)` | Plain text prompts/results; rich framework objects need a dedicated adapter |
| Async text callable | Awaited call / `ainvoke` | Async errors remain visible |
| OpenAI-compatible sync client | `chat.completions.create`, including supported chunk streaming | Completed versus cancelled streams recorded; other SDK APIs are forwarded without memory interception |
| Text LangChain Runnable | Explicit `wrap_runnable` helper | Install `nsn[langchain]`; multimodal/tool-call history needs another adapter |
| Custom client | Explicit `adapter=` | Host implements and verifies the contract |

An asynchronous OpenAI client, Responses API, arbitrary tool-message formats and every method of every model SDK are not universally supported. Verify the invocation path you use. Local tests cover sync/async text errors, supported streaming cancellation, caller-message immutability, no self-recall and namespace isolation.

Set `max_tokens=512`, `input_window_tokens=...`, `reserved_output_tokens=...` and optionally `tokenizer=...` on supported wrappers to bound injected memory. Counts without a target tokenizer/chat template are estimates. Crowded prompts may receive an empty evidence pack with a diagnostic; memory does not extend the model's actual window.

## Real local model example

Prepare a model in your own local Ollama installation first; NSN never chooses/downloads the answering model. Install the separate `openai` client if using this example:

```python
from openai import OpenAI
import nsn

client = OpenAI(base_url='http://127.0.0.1:11434/v1', api_key='local-placeholder')
nsn.init('./agent-memory')
agent = nsn.wrap(client, namespace='local', input_window_tokens=2048)
agent.chat.completions.create(model='qwen3:4b', messages=[
    {'role': 'user', 'content': 'The gateway production port is 8080.'}
], max_tokens=32)
response = agent.chat.completions.create(model='qwen3:4b', messages=[
    {'role': 'user', 'content': 'What is the gateway production port?'}
], max_tokens=32)
print(response.choices[0].message.content)
nsn.close()
```

This endpoint stays local. Availability and answer quality depend on the installed model. This usage example is distinct from the Phase 09 matched retrieval experiment.

## Offline semantic assets

The compact semantic extra contains sentence-transformers and NumPy. FAISS and spaCy belong to the `legacy` extra. Optional dependencies are not loaded by `import nsn`.

```python
import nsn
# Explicit preparation, with network access, before deployment:
path = nsn.prepare_assets('./encoder-assets')
# Later, with those assets available locally:
runtime = nsn.init('./agent-memory', semantic=True,
                   local_asset_path=path, offline=True)
```

Install `nsn[semantic]` before preparation. Preserve the encoder's license and configuration. Model assets are not included in the wheel/container. Use the same encoder fingerprint/dimension when reopening an index; mismatch raises an actionable error. `runtime.rebuild_vectors(namespace='agent')` rebuilds derived vectors from authoritative records. Keep prepared assets read-only when mounting into a semantic container; create a separate image with the semantic extra and explicitly configure `local_asset_path`/`offline=True` in your application.

## Optional ONNX and deferred indexing

Install the wheel with `[semantic-onnx]` for CPU inference without PyTorch.
Prepare the checksum-pinned MiniLM assets explicitly, before offline operation:

```powershell
python scripts/prepare_onnx_assets.py --target C:/models/nsn-minilm-onnx
```

Then configure the prepared encoder explicitly:

```python
from neurosleepnet.embeddings.onnx_pooled import ONNXPooledEncoder
encoder = ONNXPooledEncoder('C:/models/nsn-minilm-onnx')
runtime = nsn.init('./agent-memory', encoder=encoder, defer_indexing=True)
# Foreground writes persist evidence and vector jobs; lexical recall is immediate.
runtime.process_index_jobs(worker_id='my-worker', limit=100)
assert runtime.flush()  # Check success: failed jobs may still need investigation.
```

`defer_indexing` defaults to `False`. With it enabled, semantic vectors may lag
until explicit replay; a query does not start a worker. Events, fact updates and
index intent remain transactional, and replay reads current authoritative records
so deletion/supersession cannot restore stale vectors. Closing retains pending jobs
for restart. ONNX and sentence-transformers indexes have different fingerprints:
do not switch an existing index in place without an explicit rebuild/migration.
Inference uses local files only and bounds neural batches to 64 turns. Asset
preparation downloads third-party weights explicitly; assets are outside the wheel.

## Advanced commands and optional integrations

Explicit runtime commands include `record_relation`, `timeline`, `explain`, `stats`, `sleep`, `record_procedure`, `record_outcome`, `retrieve_procedures`, `export_memory`, `import_memory`, and `delete_namespace`. See phase handoffs 06–08 for signatures and provenance rules. Relationship mode requires `graph=True`. Summary retrieval is off by default; procedure steps never execute automatically. Trusted host observations must reference completed source evidence to confirm success.

`MemoryTool(runtime=runtime, namespace='agent')` shares the lightweight runtime. LangChain text history and Runnables use the optional integration helpers. REST requires `nsn[api]`, the `create_app` factory and host bearer-token-to-namespace mapping. The localhost example is `python -m examples.local_api` with `NSN_API_TOKEN` configured. No API server starts on import. Logical namespaces are not remote authentication.

## Migration and backup/recovery

Current schema: 7. Migrations are transactional, repeatable and reject newer unsupported schema versions. On an unversioned legacy `memories` database, initialization makes a `.bak` with SQLite's backup API, including committed WAL content. Existing legacy rows remain in their original table; they are not automatically converted into new facade events or trusted facts. The legacy ML interface and new event runtime are different retrieval paths. Inspect a copy before adapting legacy content through explicit runtime writes.

For a namespace backup:

```python
import json
from pathlib import Path
snapshot = runtime.export_memory(namespace='agent')
Path('agent-backup.json').write_text(json.dumps(snapshot), encoding='utf-8')
# Later: same schema, same namespace, EMPTY destination only:
fresh = nsn.Runtime('./restored-memory')
fresh.import_memory(snapshot, namespace='agent')
```

Treat imported backups as trusted host input; they include state/provenance/jobs. Import validates references, rolls back malformed input and never overwrites a populated namespace. Derived vectors are rebuilt separately when semantic assets are configured. Do not rename namespaces by editing JSON blindly.

For a whole database, stop/close writers and use SQLite's online backup API or copy the entire closed directory after checkpointing. Never copy a live `.db` alone while ignoring its WAL. Before upgrading, keep a recoverable database copy. To recover a legacy backup, close all runtimes, preserve the failed directory, restore the backup into a separate data directory and verify records before directing applications there. Do not delete original evidence during rollback. There is no distributed replication or automatic downgrader.

## Compatibility and release gates

Local installed-wheel checks exercise Windows and Linux-container Python 3.9/3.10/3.12; the Phase 10 handoff records each result, including a retained failure under concurrent load and its serial rerun. Fresh Python 3.12 semantic dependencies were independently resolved and tested with prepared assets and network access denied. Hosted CI defines the same platform matrix and optional jobs but has not executed in this session. Separate optional API/LangChain checks avoid forcing extras into the default installation. Full repository model/performance tests need explicitly prepared assets; keep them separate from fast offline checks.

Run functional regressions with `python -m pytest tests/ benchmarks/tests/ -m 'not performance' -q -p no:cacheprovider`. Run explicit hardware qualification with `-m performance -v -s`; its timing assertions are retained and may fail on a slower/busy machine. Such a failure remains a performance miss even when functional tests pass. Current measured results and failed qualification are linked from the Phase 10 handoff.

Release artifacts are local. Phase 09 remains incomplete and does not establish frontier equivalence or competitor superiority. Resource targets in `plan.md` are goals unless supported by the named measured workload. The NSN source redistribution license is not declared; public distribution awaits that decision and completion of applicable checks. See the Phase 10 handoff for measured gates, checksums and remaining work.

Repeated-query retrieval uses a bounded SQL row cache, invalidated by own-runtime or external SQLite writes; caller-mutated evidence packs do not modify cached rows. Fresh-query measurements are reported separately. The isolated 10k real semantic profile exceeds the 400 MiB incremental RSS goal, despite passing the repeated-query latency goal. Semantic ingestion in that profile uses explicit batch indexing, so it does not qualify foreground inline semantic append latency. These limitations remain release gates.


## Hardened prototype candidate

The [prototype readiness guide](implementation/prototype-readiness.md) identifies the exact new wheel, deployment scope, verification artifacts and remaining gates. Lexical ranking now preserves BM25 order rather than promoting less relevant newer events. Every rendered event/fact field is escaped before budgeting so stored markup cannot close the evidence container. This maintains a delimiter boundary, not immunity to natural-language prompt injection. SQLite retrieval failures are raised; callers must distinguish operational errors from an empty successful pack. Public append/retrieval/fact operations reject a closed runtime.

Start controlled deployments with the minimal profile. Optional ONNX/deferred indexing has its own installed, network-denied and resource checks; it does not qualify default PyTorch or inline encoding. Persistent storage can exceed a model's context, but retrieval remains bounded and current large-scale/quality limitations stay explicit.
