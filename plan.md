# NSN: implementation and validation plan

Date: 2026-10-03 (Asia/Calcutta)

Status: architecture reference plus current phase status. Phases 00–08 are implemented; Phase 09 remains partially evaluated. Phase 10 local release preparation has proceeded under its explicitly documented incomplete-evaluation prerequisite. No public release or superiority claim. See numbered handoffs for exact checks and open gates.

Execution order is now specified in [completion-plan.md](docs/implementation/completion-plan.md), using existing assets and work packages inside Phases 09/10. First repair the reproduced Unicode JSONL reader defect; then measured resource misses and native competitor adapters, frozen evaluations, human/reference grading and public ablations. Final installed/recovery/CI and owner-license gates follow. Production qualification and task-specific comparative capability claims remain separately evidenced; planning itself completes no acceptance gate.

Historical 2026-10-04 07:41 UTC checkpoint: execute [completion-plan.md](docs/implementation/completion-plan.md). Full checksum-pinned LongMemEval S/M and native LoCoMo splits remain verified. The then-active full-S local 2B answering run reached 126/4,500 rows (14 complete QA); that historical checkpoint does not establish a currently active writer; all 70,767 baseline rows, competitors and public ablations remain incomplete. Local 2B/4B/8B answering is retained; Cloud is authorized for independent judging and separate larger-model references. Development Cloud reference/judging completed 27 rows, with resumable grading and human-agreement support; human labels are pending. Phase 10's current wheel is ce32c7573366aa9d706889b042224f2d974edf2197e17aed34dac8fcc79e1431; all six local Windows/Linux Python 3.9/3.10/3.12 cells pass. Optional ONNX plus explicit deferred indexing meets RSS/append/repeated-query targets, but fresh-query latency and default inline semantic append/PyTorch RSS misses remain. Hosted CI, owner license and evaluation gates stay open; both phases remain PARTIAL and nothing is published. Current numbered handoffs supersede historical candidate evidence.

Remaining execution order, concrete exit criteria, compute/grading inputs and owner decisions are in [the completion execution plan](docs/implementation/completion-plan.md). It closes existing Phases 09 and 10 without adding phases or marking any outstanding gate passed.

Implementation handoff pack: [docs/implementation/README.md](docs/implementation/README.md). It breaks this architecture into 11 focused phase briefs with copy-paste prompts, acceptance gates, handoff requirements, and explicitly labeled recommendations. Its finer build order moves basic retrieval before the wrapper and splits optional graph, consolidation, and procedures into separate steps. Repository observations below describe the original planning audit; current implementation evidence lives in the numbered handoffs. Follow that order and preserve incomplete evidence gates when moving to local packaging.

User clarification (2026-10-04): NSN is memory infrastructure for extensive persistent history and useful bounded context. Intelligence transfer is not its product objective. The targeted prototype-hardening evidence and operational limits are in [prototype readiness](docs/implementation/prototype-readiness.md); existing Phase 09/10 acceptance targets remain unchanged.

## 1. Product objective

Build a lightweight, self-hosted memory library that gives small language models and agents persistent, useful context across sessions. Integration should require three lines after the application's model already exists:

```python
import nsn
nsn.init("./agent-memory")
model = nsn.wrap(model)
```

The string passed to `init` is a local data directory, not an API key or a model identifier. `init("")` resolves to a documented local default directory. `wrap()` requires an explicit model/client; it must not silently select a provider or patch unrelated objects globally.

The intended advantage is accurate, compact, evolving context at low operational cost. Memory can supply facts, history, preferences, and validated procedures that an SLM would otherwise lack. It does not transfer a frontier model's weights, general knowledge, or reasoning ability. We will measure how much it closes the gap on memory-dependent tasks, including remaining failures with perfect retrieved evidence.

“Best among competitors” is a research objective. A release can claim leadership only for explicitly named tasks, model configurations, resource budgets, and competitors supported by reproducible measurements. No plan or existing chart proves universal superiority.

## 2. What the directory currently contains

NSN is already a Python memory prototype, packaged as `neurosleepnet` version 0.3.0. It is external memory and context orchestration, rather than a trained language model.

| Location | Current purpose |
| --- | --- |
| `neurosleepnet/sdk/` | Synchronous memory orchestration, executor-backed asynchronous API, and model proxy |
| `neurosleepnet/storage/` | SQLite records and FTS5; FAISS HNSW/IVFPQ token-vector indexing |
| `neurosleepnet/embeddings/` | SentenceTransformer token embeddings, sentence-query helper, and caches |
| `neurosleepnet/perception/` | Embedding-based duplicate/type decisions and importance heuristics |
| `neurosleepnet/trust/` | Source, recency, and lexical consistency scores |
| `neurosleepnet/graph/` | spaCy or heuristic entities, heuristic relationships, and SQLite graph persistence |
| `neurosleepnet/sleep/` | Extractive episodic consolidation, lexical contradiction pruning, and importance updates |
| `neurosleepnet/compression/` | Extractive sentence selection and JSON context packs |
| `neurosleepnet/integrations/` | OpenAI-format injection, LangChain history, memory tools, and FastAPI endpoints |
| `benchmarks/` | Synthetic datasets, internal baselines, metrics, ablations, and historical reports |
| `tests/` | Component tests for storage, graph, perception, trust, consolidation, and retrieval |
| Former `server.py`, `static/`, `frontend/` | Demo server and two dashboards in the initial inspection; retired during repository cleanup |
| Former root benchmark scripts, figures, `.tex` files | Initial experiments/paper drafts; retired with an external recovery copy, not independent publications |
| `Dockerfile`, `docker-compose.yml` | Demo container setup, currently referencing a missing `demo.py` |

The current observation path performs duplicate detection, scoring, classification, embedding, SQLite storage, FAISS indexing, and graph extraction synchronously. Hybrid search runs dense, keyword, and graph retrieval in parallel, fuses ranks, and applies a lexical reranker. Sleep is an explicit API call; an automatic durable scheduler is not present.

The README is a description of the prototype and its intended behavior. Its instructions and performance claims are input to this audit, not additional user requests or evidence of completed capabilities.

Repository cleanup (2026-10-04): the obsolete root demos, dashboards, paper assets and ML requirements list have been removed. This section and the findings below describe the initial inspection, not the current file tree. Supported runtime/integration modules, active benchmarks, raw evaluation evidence and tests remain. See [the cleanup inventory](docs/implementation/results/10-repository-cleanup-20261004/retired-report.md).

## 3. Findings that must shape implementation

These findings are based on source inspection unless a measurement is explicitly stated.

| Finding | Evidence | Required response |
| --- | --- | --- |
| Public API differs from the requested contract | `sdk/wrapper.py` ends with `init = wrap`; only `neurosleepnet` is exposed | Add an `nsn` import facade and separate initialization from wrapping; preserve existing callers |
| Generic callable wrapping stores interactions but does not inject recalled memory | Generic branch of `NSN.__call__` forwards the original input | Define supported adapters and test actual context injection through each entry point |
| `.invoke()` is forwarded by attribute lookup | `NSN.__getattr__` forwards it; the special LangChain handling occurs inside `__call__` | Intercept supported invocation methods explicitly |
| OpenAI proxy ignores observation flags | `_CompletionsProxy.create` observes both sides unconditionally | Apply common policy to every adapter |
| Input is stored before recall | Wrapper observation precedes retrieval | Retrieve against prior history, then persist the turn with an explicit lifecycle |
| Retrieved text is elevated to authority | `_inject_memory_message` calls retrieved content verified and authoritative; source labels derive from importance | Preserve real provenance; treat retrieved text as evidence and distinguish authority from relevance |
| Namespace isolation is incomplete | Dense store searches all stored records; graph tables/query methods lack namespace fields; `get`/`forget` use IDs without scope checks | Enforce scope on every read, write, graph hop, duplicate check, deletion, cache, and index |
| Similarity can discard changed facts | `DuplicateDetector` rejects high-similarity content without checking changed values, dates, or negation | Restrict deduplication to equivalent observations; preserve corrections and history |
| Keyword baseline is weaker than its BM25 label | `SQLiteAdapter.search_keyword` matches the complete query as one phrase and does not order by `bm25()` | Implement real term-based FTS5 BM25 with explicit phrase support |
| Reranker is lexical, not a neural cross-encoder | `_rerank_score` uses substrings, bigrams, and token overlap | Rename accurately; evaluate an optional genuine local reranker separately |
| Token-vector retrieval is expensive | 384-dimensional vectors per token; JSON embedding storage; full index persistence on normal additions | Default to compact pooled vectors and incremental indexing; retain token interaction only if justified |
| Model/config wiring is incomplete | `Memory` creates the default embedder and detector rather than applying all settings | Centralize validated configuration and report effective settings |
| Graph extraction can invent weak links | Relationship extractor connects entity pairs with generic verbs or `RELATED_TO` | Separate association from factual relations and require source spans for facts |
| Consolidation can lose or hide evidence | REM deletes lower-scored records; NREM stores a summary without its embedding in SQLite, then adds an index entry whose search requires the record embedding | Preserve original evidence; version and index derived records through the normal transactional pipeline |
| Consolidation/concurrency need stronger guarantees | Pairwise REM scan; per-instance sleep flag; mutable FAISS maps and caches; executor-based async access | Bound work, introduce durable job ownership, and serialize index mutation |
| Token budget is not model-token accurate | Compressor counts words; reasoning pack emits full contexts and does not use its compressor | Budget the entire rendered pack using the target tokenizer where available |
| Local-memory claim differs from demo behavior | `server.py` defaults to `https://ollama.com/v1`; embedding assets download on first use | Make the self-hosted path explicit, local, and testable without outbound traffic |
| Install/demo packaging needs verification | README names missing root demo/test scripts; Docker copies missing `demo.py`; YAML package data is not explicitly declared | Validate built wheels, docs, examples, and container from a clean environment |

### Existing benchmark evidence is insufficient

The committed `benchmarks/results/FINAL_REPORT.md` reports:

- Update: NSN evaluates 35 of 40 questions; five are skipped.
- Contradiction: NSN evaluates four of 20 questions, reports 100% recall on those four, and reports 0% answer exact match.
- Multi-hop: NSN evaluates five of 50 questions; 45 are skipped.

`runners/evaluator.py` skips queries with no translated gold IDs and removes unavailable IDs from partially translated gold sets. This changes both question coverage and evidence denominators between systems. Rejected necessary facts must count as failures unless an equivalent retained fact demonstrably supplies the same evidence.

Additional problems: most adapters return the first retrieved record or use deterministic extraction instead of running the same answering model; the “vanilla LLM” baseline is a history scan; dense baselines reuse NSN's token-vector machinery; head-to-head evaluation does not call sleep even though reports attribute gains to REM; evaluators hardcode seed 42; cleanup misses actual `_hnsw.faiss`/`_ivfpq.faiss` filenames; report code marks NSN “BEST” unconditionally. Public dataset adapters mostly load JSON without official schema conversion or scoring.

README/report latencies and Hit@5 values also differ. Keep historical results as historical artifacts, and generate future claims from a single run manifest and raw logs.

**Verification during this planning pass:** `python -m pytest benchmarks/tests/ -q -p no:cacheprovider` completed with **68 passed**. This establishes that the existing framework tests pass; it does not validate benchmark fairness, runtime behavior, public dataset performance, or competitor superiority. The full runtime suite and expensive benchmarks were not run for this document.

## 4. Research foundation and limits

The proposed combination is an engineering hypothesis. Published results support studying particular mechanisms; their gains do not automatically transfer to NSN or small local models. Primary sources below were checked on 2026-10-03.

| Research | Mechanism informing NSN | Adaptation and evidence limit |
| --- | --- | --- |
| [MemGPT: Towards LLMs as Operating Systems, Packer et al., 2023/2024](https://arxiv.org/abs/2310.08560) | Manage working context separately from persistent external memory | Adopt explicit context budgets and tiering; do not require an autonomous paging agent in the default path |
| [LongMemEval, Wu et al., ICLR 2025](https://proceedings.iclr.cc/paper_files/paper/2025/hash/d813d324dbf0598bbdc9c8e79740ed01-Abstract-Conference.html), [full text](https://arxiv.org/html/2410.10813v2) | Decomposed storage, expanded retrieval keys, temporal query handling | Evaluate turn/fact granularity, aliases, timestamps, and compact reading packs independently |
| [Zep: A Temporal Knowledge Graph Architecture for Agent Memory, Rasmussen et al., 2025](https://arxiv.org/html/2501.13956v1) | Preserve episodes, derive facts, retain temporal validity and provenance | Implement temporal relations in SQLite; optional local extraction replaces reliance on hosted models. This is a vendor-authored preprint |
| [A-MEM: Agentic Memory for LLM Agents, Xu et al., NeurIPS 2025](https://arxiv.org/abs/2502.12110), [full text](https://arxiv.org/html/2502.12110v11) | Structured notes, memory links, and evolving derived representations | Use bounded neighborhood enrichment; evaluate local-model quality and cost before enabling automatic enrichment |
| [Mem0: Building Production-Ready AI Agents with Scalable Long-Term Memory, Chhikara et al., 2025](https://arxiv.org/html/2504.19413v1) | Incremental extraction and update decisions | Separate event capture from derived fact updates; compare equivalent local configurations. Vendor-reported outcomes require independent reproduction |
| [Reciprocal Rank Fusion, Cormack, Clarke, and Buettcher, SIGIR 2009](https://doi.org/10.1145/1571941.1572114) | Fuse heterogeneous retrieval ranks | Keep RRF as an initial fusion method; tune weights on development data only |
| [MemoryBank, Zhong et al., AAAI 2024](https://arxiv.org/abs/2305.10250) | Retention and reinforcement of useful memories | Treat decay as a cache/ranking policy; never equate non-use with falsity |
| [Complementary Learning Systems, McClelland, McNaughton, and O'Reilly, 1995](https://www.cnbc.cmu.edu/~plaut/IntroPDP/papers/McClellandMcNaughtonOReilly95PR.comp-learn-sys.pdf) | Rapid episodic capture and slower integration | Inspiration for foreground capture/background consolidation; not proof that NREM/REM labels improve LLM memory |
| [ColBERT, Khattab and Zaharia, SIGIR 2020](https://arxiv.org/abs/2004.12832) | Trained token-level late interaction | Optional research comparison. MiniLM token embeddings plus MaxSim are not a reproduced ColBERT model |
| [LoCoMo, Maharana et al., ACL 2024](https://aclanthology.org/2024.acl-long.747/) | Long conversational memory evaluation | Use official QA categories and scoring; keep retrieval-only diagnostics distinct |
| [MemX: A Local-First Long-Term Memory System for AI Assistants, 2026](https://arxiv.org/abs/2603.16171) | Recent local-first retrieval reference | Include in the research/competitor refresh; a preprint and availability of code are separate evidence questions |

New NSN hypotheses requiring ablation: typed temporal facts reduce stale answers; provenance-aware conflict handling reduces feedback contamination; small evidence packs improve SLM answers; asynchronous bounded consolidation improves useful recall per byte without harming history.

## 5. Target architecture

Use one embedded engine with a transactional SQLite source of truth and replaceable derived indexes. No mandatory graph server, vector database service, cloud key, GPU, or always-running daemon.

| Component | Responsibility | Default behavior |
| --- | --- | --- |
| Public SDK and adapters | Initialize, wrap supported invocation paths, preserve response contracts | Explicit local runtime; no global provider patching |
| Event store | Durable interactions, documents, tool observations, provenance | SQLite transactions, scoped IDs, UTC timestamps |
| Fact/version manager | Typed facts, temporal validity, conflict groups, lifecycle | Conservative extraction; retain uncertainty and original events |
| Retrieval planner | Interpret scope, time, entities, and query intent | Cheap deterministic rules; optional bounded local enrichment |
| Hybrid retriever | BM25, optional pooled-vector search, bounded relationship expansion | Capability-aware RRF and deterministic tie-breaking |
| Evidence packer | Select useful evidence under the model's context budget | Facts, dates, sources, unresolved conflicts, and supported paths |
| Consolidation worker | Build derived facts/summaries/procedures; index updates | Durable SQLite job queue, bounded batches, local execution |
| Lifecycle/diagnostics | Close, flush, delete, export, backup, inspect | Visible errors and effective configuration |

### 5.1 Storage and memory semantics

Proposed logical tables: `events`, `facts`, `fact_support`, `entities`, `entity_aliases`, `relations`, `procedures`, `summaries`, `jobs`, `index_state`, `schema_migrations`, plus FTS5 tables.

Every record and lookup carries namespace scope. Sessions live inside namespaces; cross-agent sharing requires an explicit shared namespace or policy. A namespace is logical isolation inside a trusted process, not an authentication boundary for an exposed service.

An event stores ID, namespace, session, role, raw content, source identity, observed time, optional event time, metadata, retention policy, and idempotency key. Events remain unchanged except explicit retention/deletion operations. Store assertions, questions, and assistant responses as different event roles; a question is not automatically a fact.

A fact stores subject, predicate, typed value, qualifiers, validity interval, recorded/invalidation times, status, extraction confidence, conflict group, and links to supporting event spans. Distinguish asserted, derived, confirmed, superseded, disputed, and deleted states. Confidence in extraction, source authority, factual support, and retrieval relevance are separate signals.

Resolve fact identity using subject + predicate + qualifiers + namespace. Predicate schemas specify whether values are single-valued or set-valued. A new value does not automatically contradict a set-valued fact. Missing event dates remain unknown; ingestion order must not masquerade as real-world chronology.

For a clear single-valued update, retain the previous version and close its validity interval. For incompatible claims with overlapping validity, retain both and apply a documented source policy; unresolved conflicts appear in the pack. Explicit user corrections can supersede that user's earlier preferences. Repetition and assistant agreement are not independent corroboration. Caller-supplied `source="system"` alone does not authenticate a source.

Deduplicate exact repeat events with scoped hashes/idempotency keys. Link semantically equivalent facts only after values, units, dates, qualifiers, and negation agree. Never discard a changed port, identifier, preference, or date solely because the sentences have similar embeddings.

### 5.2 Foreground lifecycle

1. Validate input and resolve runtime, namespace, session, and context limits.
2. Retrieve prior evidence using a database snapshot before the current input is searchable as memory.
3. Build a bounded evidence block without changing the caller's original messages.
4. Durably record the input event and turn state, keeping it excluded from the earlier retrieval snapshot.
5. Invoke the original model once, preserving parameters, types, exceptions, tool calls, and stream behavior.
6. Record the completed response as an assistant event; do not promote its claims to confirmed facts by default.
7. Commit enrichment/index jobs. `flush()` provides an explicit durability/index-consistency barrier.

Failed, cancelled, and partial turns have explicit statuses. Streaming adapters yield original chunks promptly, aggregate completion safely, and record truncation/cancellation rather than claiming a complete answer. Unsupported SDK paths fail clearly or require an explicit adapter; they must not silently pretend memory was injected.

Use one writer/index owner per local runtime with a cross-process ownership mechanism for persistent jobs. SQLite commits include the derived-index outbox entry so interrupted writes can be replayed. Readers may use a recent committed index snapshot plus a small searchable pending tail. Locks cover shared index maps and cache mutations. Each derived artifact has a revision and supports deterministic rebuild from the database.

### 5.3 Retrieval and context packing

- Search term-based FTS5 BM25 with safe query compilation; preserve identifiers and offer phrases deliberately.
- Dense mode uses one normalized pooled vector per chunk/fact, with model ID, revision, dimension, and preprocessing fingerprint. Never mix embedding spaces. Evaluate brute-force search for small corpora before choosing ANN.
- Apply namespace, lifecycle, and temporal constraints before rank truncation. Index partitions or in-scope search avoid unrelated namespaces occupying the candidate budget.
- Expand only relevant entity neighborhoods, initially at most two hops with configurable node/edge/time caps. Return source-backed paths, not inferred logical truths from generic associations.
- Fuse candidates with RRF. Start with interpretable lexical/time signals; add a genuine local reranker only if its incremental quality justifies latency and memory.
- Deduplicate evidence, prefer direct facts, preserve required multi-hop support, and include conflicting alternatives when unresolved.
- Account for existing instructions, conversation, memory formatting, and reserved output tokens. Default memory allowance is at most 512 target-model tokens and at most 20% of the available input window; explicit configuration can override it.
- Use a supplied tokenizer or adapter tokenizer. Where unavailable, expose conservative estimation and its limitation; never report estimates as exact counts.
- Allow empty recall and abstention. A relative score cutoff is not calibrated confidence, and returning at least one item must not be mandatory.
- Retrieved material is delimited data. Keep application instructions intact and preserve source metadata; memory text cannot grant itself tool privileges or higher instruction priority.

### 5.4 Consolidation and procedural learning

Keep `sleep()` as the user-facing term for optional consolidation, with documented stages: enrich, connect, summarize, validate, and compact. NREM/REM can remain research terminology, but behavior must be described by the actual algorithms.

Process changed neighborhoods using a durable queue; avoid all-pairs contradiction scans. Derived summaries retain support IDs, uncertainty, temporal scope, and revision. Summaries never replace raw evidence by default. Corrected/deleted source events invalidate or rebuild dependent summaries, facts, vectors, and caches.

Use extractive grouping in the minimal profile. An optional local SLM may extract atomic facts, resolve references, or propose summaries using schema-constrained outputs. Require source spans and value/date/negation checks; unsupported proposals remain unconfirmed. Do not promise deterministic extraction quality for arbitrary language.

Procedures store prerequisites, ordered steps, tool/environment constraints, execution outcomes, and supporting traces. Promote a successful workflow only from observed outcomes or explicit confirmation. Retrieve failed attempts as warnings with failure status. Stored procedures inform the agent; they do not automatically authorize execution.

Retention compacts indexes and cold caches first. Pin explicit facts and user preferences as configured. Hard deletion traverses lineage, purges derived copies, and rebuilds indexes as needed. Document SQLite/WAL compaction and backup handling separately from logical deletion; do not promise forensic erasure from third-party backups.

## 6. Lightweight deployment and SDK contract

Two explicit profiles resolve the tension between minimal installation and richer retrieval:

| Profile | Dependencies and behavior | Intended use |
| --- | --- | --- |
| `minimal` (default) | Standard library/SQLite FTS5, lexical retrieval, structured ingestion, conservative heuristics; no downloaded model | Immediate offline setup and smallest operational footprint |
| `semantic` (optional extra) | Small local encoder and pooled-vector backend; evaluate ONNX/quantization; no mandatory Torch/spaCy stack | Better paraphrase retrieval while remaining self-hosted |
| Advanced local enrichment (explicit option) | User-configured local SLM for structured extraction/consolidation and optional reranker | Higher memory quality with separately measured resource cost |

The default's semantic limitations must be visible. If minimal cannot meet quality goals, publish separate profile results and choose the recommended profile from measurements; do not attribute semantic-mode results to the dependency-free default.

No model downloads at import or initialization. Optional assets are prepared explicitly, versioned, and reusable offline. An offline mode rejects remote memory-processing endpoints. The application's own inference provider remains an explicit user choice; a cloud model receives any injected evidence, so completely local deployment requires a local model too.

Target SDK behavior:

```python
import nsn

# Simple integration; model is an existing application model/client.
nsn.init("./agent-memory")
model = nsn.wrap(model)

# Explicit runtime for advanced use and multiple independent agents.
memory = nsn.init("./team-memory", namespace="research", profile="semantic")
agent = nsn.wrap(client, memory=memory, session_id="session-1", token_budget=512)
memory.remember("Service Alpha uses port 8080", source="user")
hits = memory.recall("Which port does Service Alpha use?")
pack = memory.context("Which port does Service Alpha use?", token_budget=256)
memory.sleep()                 # Bounded consolidation; returns a report.
memory.flush()                 # Wait for queued work to become consistent.
memory.close()                 # Idempotent lifecycle cleanup.
```

Additional advanced APIs: structured `remember` with event time/metadata, `forget`, `timeline`, `explain`, `stats`, `export`, and import/backup tools. Freeze names and return schemas during the SDK phase. Reinitializing the same directory/config reuses a runtime; incompatible settings require an explicit new handle. Calls before initialization raise an actionable error. Double wrapping is idempotent or clearly rejected.

Keep the `neurosleepnet` distribution/import working while adding `nsn` as an import package. Check distribution-name availability before any publication; `import nsn` does not require renaming the PyPI distribution. Keep legacy `neurosleepnet.init(model, ...)` through a deprecated compatibility shim, with explicit routing that distinguishes a model object from a directory path. Back up old stores, migrate schema transactionally, and rebuild old derived indexes without deleting raw memories.

## 7. Implementation phases and exit criteria

Work proceeds in small reviewable changes. Later phases depend on the earlier correctness gates; preserve useful existing components rather than rewriting the repository wholesale.

| Phase | Work and main files | Exit criterion |
| --- | --- | --- |
| P0: Freeze evidence and repair evaluation | `benchmarks/runners/evaluator.py`, baseline adapters, metrics, report generation, public dataset conversion | Same question set for all systems; rejected evidence counts; no synthetic “LLM” claims; actual artifact cleanup; neutral reports and run manifests |
| P1: Establish durable scoped storage | `storage/sqlite.py`, schemas, migrations, repository methods, new event/fact tables and outbox | Cross-namespace tests pass for every operation; corrections/history survive restart; interrupted index updates replay correctly |
| P2: Deliver the three-line SDK | New `nsn/` facade, `sdk/wrapper.py`, settings/runtime lifecycle, package metadata | Clean wheel installation; documented supported paths inject prior context; flags, streaming, async, failures, and double-wrap behavior verified |
| P3: Deliver useful minimal memory | FTS5 retrieval, temporal fact manager, safe deduplication, evidence packer | Changed values remain available; current/as-of answers use correct evidence; conflicts persist; packets respect token budgets; empty recall supported |
| P4: Add compact semantic retrieval | `embeddings/`, pooled-vector storage/index backend, asset preparation and diagnostics | Better dev-set paraphrase recall within declared CPU/RAM budget; dimensions/models validated; no outbound calls offline |
| P5: Add bounded graph and consolidation | `graph/`, `sleep/`, `compression/`, local enrichment interface | Source-backed bounded paths; idempotent jobs; summaries preserve facts; no regression on updates/history/deletion; measurable quality gain or leave feature optional |
| P6: Evaluate real SLMs and competitors | Official dataset loaders, competitor adapters, generator protocol, scoring, resource telemetry | Complete comparable runs, paired confidence intervals, per-category errors, and reproducible artifacts |
| P7: Package and release | README/examples, Docker/service extras, CI, migration notes, one dashboard | Minimal install/offline/restart demo works on supported platforms; claims trace to valid runs; release gates pass |

P0 and P1 are the first implementation priority. Benchmark bias and lost updates can make subsequent optimizations look successful while hiding failures.

The first usable milestone is P1-P3: reliable offline memory and the requested API. Research-heavy graph/consolidation features follow after that foundation works.

## 8. Validation against real alternatives

### 8.1 Comparison set

Use actual implementations, pinned to released versions or commits:

| Comparator | Why include it | Fair configuration |
| --- | --- | --- |
| Stateless SLM, rolling conversation, full context | Isolate the value of persistent memory and context reduction | Same real answering model; enforce context limits and document truncation |
| Correct BM25, pooled-vector RAG, hybrid RAG | Strong simple baselines | Same corpus, encoder where applicable, chunking search budget, and generator |
| [Mem0 OSS](https://github.com/mem0ai/mem0) | Incremental memory extraction/update competitor | Run supported local configuration; record extraction and generation costs |
| [Graphiti](https://github.com/getzep/graphiti) | Temporal graph competitor | Pin graph backend and local model settings; Graphiti's official README includes an Ollama configuration |
| [Letta](https://github.com/letta-ai/letta) / MemGPT | Stateful memory management alternative | Document platform/agent differences and count all management-model work |
| [A-MEM](https://github.com/agiresearch/A-mem) | Research baseline for linked/evolving notes | Official implementation and bounded local-model setup where supported |

Refresh the landscape before final evaluation, including newer local-first systems such as MemX where reproducible code exists. Graphiti OSS results must not be labelled hosted Zep results. Unavailable/incompatible systems are reported as unevaluated, not inferior.

Run two tracks: (A) matched local models/context/resources to compare memory mechanisms; (B) each system's documented recommended setup to compare complete products. Label them separately. No cross-paper percentage comparison counts as a head-to-head result.

### 8.2 Datasets and task coverage

Use [official LongMemEval](https://github.com/xiaowu-jiang/LongMemEval), including update, temporal, multi-session, assistant-side information, and abstention tasks; and [official LoCoMo](https://github.com/snap-research/locomo) QA with its published category/scoring conventions. Convert native schemas and preserve sessions, speakers, dates, and evidence labels. Existing JSON loader classes are not sufficient integrations.

Start with a small frozen development slice for debugging; run complete designated evaluation sets for published results. Record dataset version/checksum, license, any exclusions, and the exact variant. Avoid random question-level splitting when questions share conversations: tune on disjoint conversation clusters and freeze all test settings. Respect official splits where supplied; label any custom partition clearly.

Add synthetic regressions with unique scoped entities and explicit timestamps: changed numeric values, negations, equal-authority conflicts, delayed/out-of-order events, multilingual queries, identifier retrieval, pronoun references, complete multi-hop evidence, malicious memory instructions, namespace isolation, deletion, and restart recovery. Synthetic diagnostics supplement public evaluation rather than replace it.

Add tool-task traces to test reusable procedures and demonstrated success/failure. Context-only QA cannot prove improved agent task success.

### 8.3 Answering models and the frontier-context objective

Pin at least three locally available model sizes, approximately 1-2B, 3-4B, and 7-8B, subject to hardware feasibility. Record exact checkpoint, quantization, tokenizer, context window, prompt, and decoding settings. Choose actual models when evaluation begins rather than assuming today's names remain best.

For each model compare no memory, NSN, matched baselines, and an oracle-evidence diagnostic. The oracle shows the model's remaining reasoning limit even when retrieval succeeds.

A frontier-model reference is optional and separately identified: use consented non-private evaluation data and an explicitly configured provider, or a suitable stronger self-hosted model. It is an evaluation reference, not a runtime dependency. If no frontier reference is run, do not claim frontier parity.

Measure gap closure only when the frontier reference outperforms the baseline: `(SLM+NSN score - SLM baseline score) / (frontier reference score - SLM baseline score)`. Publish raw scores, confidence intervals, conditions, and negative or greater-than-one values without misleading clipping. This measures the specified task gap, not universal intelligence.

### 8.4 Metrics and experiment integrity

Report answer accuracy under official scoring, supported-answer accuracy, abstention quality, stale-answer rate, conflict handling, complete-path success, and agent task success where applicable. Retrieval diagnostics include Recall/Hit@K, MRR, nDCG, and all-required-evidence coverage. A single retrieved edge is insufficient for a multi-hop success.

Keep immutable question/evidence identities across systems. Derived memories receive lineage-based evidence coverage only when they preserve the required claim; merely listing a source ID is insufficient. Deduplication failure, capacity loss, parsing failure, and timeout remain in the fixed denominator. Use retrieval N/A only for systems/tasks where it is genuinely undefined.

Separate retrieval, packing, ingestion, consolidation, generation, and end-to-end latency; report cold/warm p50/p95, unique versus repeated queries, throughput, initialization time, peak RSS, disk/index size, and actual prompt/output tokens. Include extraction/reranking/consolidation model calls and token cost, not only the final answer call.

Use paired comparisons on identical questions, conversation-cluster bootstrap 95% confidence intervals, multiple seeds where randomness exists, and an explicit multiple-comparison policy. Existing approximate statistics helpers need review before publication. Judges use frozen prompts/settings, recorded rationales/scores, and a human-reviewed sample; deterministic containment is not sufficient for contradictory answers.

Log every query, selected evidence, packed context, answer, failure, model setting, artifact revision, hardware specification, and timing. Test index reset/rebuild and randomize run order to avoid stale indexes/cache bias. Tune only on development data with comparable tuning budgets.

### 8.5 Ablations

Measure minimal versus semantic mode; pooled versus existing token vectors; lexical/dense/hybrid; temporal versions on/off; provenance policy on/off; graph on/off; reranker on/off; consolidation on/off; raw-only versus facts+raw; and fixed versus budgeted context packs. Keep input corpus, generator, questions, and budget constant. “Off” must actually disable the component, including its background work.

Accept additional complexity only when it improves the preregistered quality/resource tradeoff. A simpler system that matches accuracy with less RAM is a successful result.

## 9. Proposed release targets

These are engineering goals to validate, not measured capabilities. Establish a reference machine and frozen workloads in P0; publish separate profile results.

| Dimension | Initial target/gate |
| --- | --- |
| Setup | Three integration lines after the model exists; default works with no cloud key or model download |
| Import | No model loading, outbound requests, database creation, or background thread creation on import |
| Minimal footprint | NSN wheel under 5 MB; incremental memory runtime RSS under 100 MB at 10k short events, excluding the application's model |
| Semantic footprint | Encoder assets at most 100 MB; incremental runtime RSS under 400 MB at 10k short events; verify actual backend behavior |
| Warm retrieval | p95 under 25 ms minimal / 100 ms semantic at 10k short events on the declared CPU, including query embedding and packet building, excluding generation |
| Foreground storage | p95 durable append under 10 ms on the reference disk; expensive enrichment stays off the immediate path |
| Context size | Default memory pack fits its 512-token/available-window allowance; exact checks when a tokenizer is provided |
| Isolation/correctness | Zero leakage in namespace regression suite; changed facts retained; deletion propagates through derived stores |
| Recovery | Crash/restart tests preserve committed events and replay pending index work; durable jobs are idempotent |
| Local operation | Network-denied core test succeeds; semantic mode succeeds with pre-provisioned assets |
| Quality milestone | Aim for at least +5 absolute accuracy points over matched hybrid RAG on the frozen primary evaluation; paired CI must exclude zero |
| Competitor claim | Every named superiority claim has a comparable run; include losses and tradeoffs rather than a universal “best” label |

Scale separately at 1k, 10k, and 100k events, and investigate 1M only after the small deployment meets targets. Report corpus lengths because event counts alone are insufficient. If a target fails, record the result and adjust design or declared scope rather than silently changing the test.

## 10. Completion checklist and boundaries

- [x] Inspect README, source paths, integrations, dependency declarations, benchmark code, and committed reports.
- [x] Verify relevant primary research and identify reproducible competitors.
- [x] Run existing benchmark-framework tests: 68 passed.
- [x] Create this plan before architecture implementation.
- [x] Repair evaluation coverage, baselines, and run provenance.
- [x] Implement scoped durable events, fact versions, and recovery.
- [x] Implement and verify the `nsn.init` / `nsn.wrap` contract.
- [x] Measure isolated minimal and real semantic installed-wheel profiles, including child processes and excluding the answering model (2026-10-04).
- [ ] Qualify all profile resource gates: semantic RSS and fresh-query latency still miss targets; inline semantic append remains unqualified.
- [ ] Validate optional graph/consolidation with ablations.
- [ ] Run public benchmarks with real SLMs and pinned competitors.
- [ ] Release only with accurate, reproducible claims.

Defaults assumed for planning: Python-first, embedded SQLite, single-host deployment, English-first extraction with explicit multilingual evaluation, and optional HTTP/agent-framework adapters. Distributed replication, universal framework interception, model training, and a mandatory dashboard are outside the initial milestone.

Open implementation decisions are resolved through measured spikes: encoder/backend choice, which SLMs fit the available hardware, extraction quality thresholds, and which advanced features merit default activation. Start with correctness and a usable API; let the experiments determine the final architecture.

Repository note: `plan.md` is now eligible for versioning so future phase handoffs retain this architecture and its outstanding evidence gates. No commit is implied.

Memory verification checkpoint (2026-10-04): 18 additional contract tests verify durable memory, bounded recall beyond the model window, namespace separation, corrections, recovery and installed semantic filtering. Reproduced historical-event/session top-k and closed-maintenance defects repaired with narrow compatible changes. Full functional suite: 300 passed, 2 skipped, 1 performance deselected. See [actual installed evidence](docs/implementation/results/10-memory-verification-20261004/report.md). This does not complete performance, public comparison, hosted CI or owner-license gates in Phases 09/10.
