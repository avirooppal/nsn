# NSN completion execution plan

Date: 2026-10-04. Status: ACTIVE; Phase 09 and Phase 10 remain PARTIAL.

This plan closes the existing Phase 09 and Phase 10 requirements in `plan.md`; it does not introduce new phases. The runtime is usable, but research qualification and release readiness are incomplete. Current evidence is linked in `handoffs/09.md` and `handoffs/10.md`; the `10-completion-*` artifacts identify the current wheel. Earlier `10-final-*` artifacts identify an older candidate and remain historical evidence. Preserve historical failures and existing user changes.

Latest targeted prototype checkpoint: [readiness](prototype-readiness.md) records four reproduced defect repairs, seven new tests, a separately built f02 candidate, 282 functional passes, six fresh installed matrix passes and improved minimal evidence recall. Fresh-query latency and full Phase 09/10 gates remain open. User scope is extensive persistent memory/context, not intelligence transfer. Continue work packages 09-B/F and 10-A/B against this candidate; preserve old wheel/results.

## Execution plan using the assets already available

Updated 2026-10-04 (Asia/Calcutta). This is the actionable order within the existing **Phase 09**, followed by **Phase 10**. The numbered work packages below are assignments, not new implementation phases. Planning does not mark any remaining gate passed.

### What success means

1. **Production qualification:** the declared installation, durability, isolation, adapter, recovery and resource contracts pass on the supported configurations, with accurate operational documentation and release metadata.
2. **Measured memory improvement:** complete, independently checked experiments show whether NSN improves the selected SLMs' answers, evidence use and agent tasks. A negative result is a valid outcome, followed by a development iteration if needed.
3. **Competitive standing:** actual pinned competitors are compared on the same frozen questions under separate matched-resource and vendor-recommended tracks. Claim a win only for the configurations, datasets and metrics whose comparable evidence supports it.
4. **Larger-model gap:** use the available Cloud models as separately identified references and measure the remaining task-specific gap. A 31B reference or a 120B judge is not automatically a verified frontier reference. General frontier reasoning/intelligence equivalence is not an acceptance criterion that memory alone can guarantee.

Completing software and validation is achievable work. Best-among-all-competitors and frontier equivalence are research outcomes to test, not labels to promise in advance. Preserve existing targets; a limited internal deployment does not silently close the full Phase 09/10 gates.

### Reuse inventory; avoid a rebuild

| Already available | Use in this plan |
| --- | --- |
| Three-line API, SQLite/WAL events/facts, scoped retrieval, durable indexing jobs | Keep the default product surface and source-of-truth contracts |
| Optional semantic, bounded graph, consolidation, procedures and existing integrations | Qualify supported behavior; keep optional features explicit until their own evidence exists |
| Current 104,013-byte wheel, SHA `ce32c7573366aa9d706889b042224f2d974edf2197e17aed34dac8fcc79e1431` | Baseline candidate; preserve its artifacts and source snapshots |
| Local pinned Qwen3 1.7B (approximately 2B), 4B and 8B | Required answering models; run one heavy model workload at a time on the current host |
| Authorized Ollama Cloud credential, Gemma4 31B reference and GPT-OSS 120B judge | Independent judging and separate larger-model reference; never substitute for local SLM coverage |
| Prepared MiniLM PyTorch/ONNX assets, exact Qwen chat counter | Neural comparisons, resource profiles and honest native management-window checks |
| Full checksum-pinned LongMemEval S/M, LoCoMo, frozen splits/scorers | Reuse existing corpus preparation; no new broad dataset collection is needed to start |
| Immutable corpus cache, raw row ledgers, grader/reporter, frozen source snapshots | Resume compatible work, share only native content, independently regenerate reports |
| Actual pinned Mem0/Graphiti/A-MEM environments, current Letta source, pulled Neo4j image | Complete missing native adapters and qualify service setup sequentially |
| Installed matrix scripts, offline smoke/recovery tests, hosted CI definition | Requalify the final candidate rather than design another testing platform |

The current host has 12 logical CPUs and about 16GB RAM. Use sequential heavyweight jobs, bounded embedding batches and explicit process cleanup. Do not buy hardware, add a hosted memory dependency, build a distributed evaluator, or run resource profiles alongside answering jobs. Measure remaining runtime from actual artifacts before estimating a completion date.

### Phase 09 work packages, in execution order

**09-A — Restore trustworthy reporting and continuation first.**

Implementation checkpoint (2026-10-04): current ledger/grading/report/audit readers now share a small strict physical-LF parser; 138 benchmark tests pass. Real-run coverage reads correctly and Cloud report regeneration agrees. SDK/raw artifacts/frozen snapshots are unchanged. Legacy frozen-resume and truncated-record recovery boundaries remain open; see [handoff](handoffs/09.md). This does not complete the entire 09-A continuation gate or either phase.

New reproduced defect: native conversation content contains U+2028 inside valid JSON strings. `str.splitlines()` treats that content as record separators. A read found 261 valid physical JSONL rows, 261 unique pairs and 29 complete QA, while text splitting produced 277 fragments. This is a parsing defect, not evidence that those native rows are corrupt. See [diagnosis](results/09-jsonl-unicode-planning-diagnosis-20261004.json). Earlier live-audit errors must not be called completed reports.

- Fix record iteration in `ledger.py`, both grading readers, report CLI and `scripts/audit_completion.py` using physical newline-delimited records. Preserve embedded Unicode content exactly.
- Add meaningful tests for U+2028/U+2029 inside strings, malformed complete records, a truncated final record after process interruption, duplicate/unknown identities and unchanged full denominators. A live checkpoint may explicitly defer a still-in-flight final record; it must not silently skip malformed committed records.
- Regenerate coverage without touching raw rows. Verify current process ownership before resuming; do not create a second writer.
- Existing source snapshots contain the old parser. Do not patch them in place or bypass checksum validation. Current fixed readers can regenerate reports from old immutable raw artifacts with the reader version recorded. If frozen resume needs a changed reader, choose a new explicit run/provenance boundary and preserve the parent artifacts; do not silently label changed sources as the same experiment.

**Done when:** Unicode-containing native rows report correctly, crash/resume is tested, raw hashes remain unchanged by reporting, and compatible continuation has explicit provenance.

**09-B — Repair the remaining measured resource misses before a final SDK freeze.**

- Reproduce unique-query p95 on the unchanged isolated 10k workload: minimal 85.88ms against 25ms; optional ONNX 143.88ms against 100ms in the retained measurements.
- Profile SQL ranking/selection, candidate resolution, encoding and packing separately. The previous SQLite cache-size experiments showed no useful gain; do not repeat them without new evidence. Change only a measured bottleneck and verify ranked evidence equivalence, identifiers, temporal/status filtering, namespace/session isolation and cache invalidation.
- Retain the explicit ONNX/deferred configuration: its measured memory, repeated-query and deferred-append targets pass. It does not fix default inline encoding or default PyTorch memory. Investigate those independently; batching/replay must not be counted as inline append.
- Test foreground durability, pending-vector visibility, flush/replay, stale target validation and process-tree memory after every relevant change. Do not weaken workloads, deadlines, assertions or resource targets.

**Done when:** the unchanged declared performance gates pass for every configuration being qualified; any remaining miss stays an open gate. If the SDK changes, build a new wheel and start new evaluation source snapshots. Old candidate evaluation stays separate; it can continue for diagnostic coverage while coding, but performance qualification needs a quiet machine.

**09-C — Complete native competitor adapters with the installed sources.**

| System | Next concrete work | Qualification evidence |
| --- | --- | --- |
| Mem0 | Diagnose the dispatched 8k extraction timeout; retain the matched 2k overflow. Use the verified exact counter and preserve native prompts/roles/dates. Verify actual returned input counts before accepting a native extraction result | Full conversation ingestion/query/restart, recorded management calls/failures, native result lineage |
| Graphiti | Configure an isolated local backend from the already pinned image; implement pinned SDK ingestion/search and management-provider instrumentation. Run only when sufficient memory is available | Real service calls, isolated graph namespace, complete native ingestion, path/source evidence and recovery |
| A-MEM | Diagnose native import/startup, then use its actual note creation/evolution/retrieval APIs in an isolated process; inspect native persistence/reset semantics rather than inventing restart support | Real native calls, management logs, roles/date mapping, isolation and supported restart behavior or explicit limitation |
| Current Letta | Build the already pinned current code with its declared runtime; qualify local provider and native agent/memory interaction. Do not substitute the archived Python implementation | Native setup/ingestion/query, complete agent/management logs, supported state persistence |

Keep matched-resource and documented vendor-recommended settings separate. Cloud JSON-schema support cannot be assumed; do not replace native extraction with a hand-written RAG or fabricated response. Missing backends, incompatible windows and failed startup/extraction remain recorded failures/exclusions, not invented competitor accuracy zeros.

**Done when:** each required actual system has a working verified adapter and documented configuration, or an explicit reproducible incompatibility preventing a required comparison. The latter does not close the all-competitor acceptance gate. Add adapter integration tests using real local services where feasible; mocks test failure handling only.

**09-D — Freeze and execute the final answering campaign.**

- Freeze the final SDK/evaluator, encoder assets, scorer, prompts, budgets, model digests, source checksums, seed, hardware and analysis policy after relevant fixes. Preserve historical exposure and zero holdout tuning.
- Order: finish compatible current 2B full-S diagnostic work; prioritize the declared LoCoMo 4B primary comparison in the final candidate, then complete the remaining model/dataset tracks. A new candidate requires its own complete coverage; old diagnostic rows cannot fill its missing identities.
- Run all nine existing profiles, three local sizes, LoCoMo 1,621 questions and LongMemEval S/M 500 each: **70,767 baseline rows**, before additional competitors, ablations and references.
- Run sequentially on this machine; reuse only checksum-bound native corpus builds and restore private mutable copies. Log one-time ingestion/index costs, restore costs, per-question retrieval/generation and every failure. Maintain one writer per run.
- Measure remaining native build times plus answer/management/judge call counts and observed latency. Estimate runtime and Cloud cost from those measurements; no arbitrary deadline or unapproved new provider is needed.

**Done when:** every declared question/profile/model identity has exactly one row, manifests agree, failure denominators are complete and all missing grades are explicitly visible.

**09-E — Calibrate independent grading and measure the larger-model gap.**

- Keep upstream deterministic LoCoMo scoring. Use GPT-OSS 120B Cloud with the pinned upstream LongMemEval rubric; record provider revision, full requests/responses, failures and output budget. Provider weight/hardware opacity remains a limitation.
- Start with the existing frozen [10-item human review queue](results/09-cloud-resumable-judge-human-review-20261004.json). A human fills labels/rationales; validate agreement, disagreement, uncertainty and missing judge outcomes. Expand the predeclared development calibration sample if the existing sample does not cover relevant categories or is too small; do not select only favourable cases.
- Predeclare the calibration sample/adequacy rule before looking at new labels. Use uncertain-label/adjudication policy already declared; no AI-generated human labels. Same-model self-judge scores stay excluded permanently.
- Grade full compatible holdout runs resumably. Failed grades are retained, with full-denominator bounds; no selective retry or deletion.
- Evaluate the available larger-model reference on the same frozen QA. For a matched comparison, keep evidence and generation budgets fixed. Report a larger-window/reference configuration separately, with its actual cost. Quantify per-category/task gap instead of claiming general intelligence transfer.

**Done when:** full answering coverage has validated independent scoring and completed human calibration; reference comparisons have identical declared question coverage and explicit limitations. If human validation is inadequate, judge-based accuracy remains unqualified.

**09-F — Run public ablations, then audit the evidence.**

- Existing baseline contrasts cover lexical/dense/hybrid and minimal/semantic. Add public feature contrasts for temporal state, provenance, bounded graph, consolidation, facts-plus-raw and context budget using actual toggles; paired runs keep model/questions/other budgets fixed.
- Declare pooled/token-vector and reranker status honestly. If the plan requires an implementation that does not exist, make it a separately bounded optional experiment; absent functionality is not a successful ablation. Do not expand the default SDK to run an experiment.
- Prepare source-backed stale/conflict/relationship/task annotations on development data with independent validation. Freeze annotations before holdout scoring. Synthetic fixtures support functional testing and cannot supply public ablation/task success claims.
- Respect [analysis policy](phase09-analysis-policy.json): LoCoMo 4B hybrid versus NSN semantic is primary; other contrasts exploratory. Preserve paired cluster intervals and the eight-cluster limitation. LongMemEval's one connected cluster cannot support a meaningful cluster interval; resolving that needs an explicit justified independent-data/design decision, not treating QA as independent.
- Regenerate every report from raw artifacts using a separate invocation. Audit provenance, costs, failure coverage, scoring, human review and matched/recommended labels. Test the +5-point hypothesis and report a miss if that is the result.

**Done when:** the planned public comparisons/ablations and uncertainty gates are evidenced; there is a reproducible neutral report. If comparable results do not favour NSN, development iterations use development data and a new candidate, without rewriting historical evidence or tuning on holdout.

### Phase 10 work packages after Phase 09 qualification

**10-A — Qualify the final installed candidate.**

Reuse the existing Windows/Linux Python 3.9/3.10/3.12 scripts and fresh minimal/PyTorch/ONNX environments. Build a wheel from the evaluated source; verify its bytes/checksum and install outside checkout. Recheck offline operation, concurrent owners, namespace isolation, migration/WAL backup, malformed recovery, cancelled streams, sync/async/invoke adapters, context budgets, delete/supersede replay and prepared-model restart. Add tests for newly repaired defects, not tests that merely repeat implementation. Run the real optional REST/LangChain/integration checks and disclose exclusions.

Run the declared hosted workflow when repository access is available. Add a bounded mixed-operation reliability run using existing APIs: append, retrieve, correction, namespace switching, cancellation, restart and job replay, checking authoritative invariants and bounded memory. Declare duration/workload before execution; it supplements rather than replaces required tests. Repeat unchanged isolated resource profiles with answering processes excluded.

**Done when:** the exact evaluated wheel passes the declared compatibility/CI/recovery/integration and resource gates, with no unresolved material defects. Existing local matrix successes are reused for unchanged bytes, but do not imply hosted CI passed.

**10-B — Finish operational docs, license and release decision.**

Document production configuration, deferred consistency, asset preparation, backup/restore, upgrade/rollback, diagnostic failures and tested adapter limits using actual verified behavior. Keep few-line setup and lightweight defaults; no new mandatory services. Have the owner choose the source license explicitly, then review package/dependency/data notices and metadata. Update `plan.md`, index, handoffs, changelog and checksums together. No deployment/publication is authorized by this plan.

**Done when:** all Phase 09/10 gates have attached evidence or remain clearly incomplete; only a fully qualified scope may be called production-ready. Competitor leadership/frontier-gap claims must cite their completed comparable evidence, separately from the release decision.

### Division of work and owner inputs

| Implementation work | Owner input required |
| --- | --- |
| JSONL reader repair, latency/footprint work, native adapters, ablation wiring, annotation preparation, tests, runs and reports | Keep the current machine available for sustained execution; provide another existing machine only if available, without changing model/data contracts |
| Independent grading and review tooling | Human labels/rationales and adjudication; the existing Cloud credential already enables configured judging |
| Installed matrix and hosted workflow preparation | Repository/runner access to execute hosted CI; publication is separate |
| Notices and release metadata preparation | Explicit source-license choice; leave it unresolved until the owner chooses |

Do independent engineering while owner inputs are pending. Record the exact missing dependency rather than stopping all work or claiming gates passed. No additional credential, new dataset or hardware purchase is required to start the next assignments.

### Instructions for the next implementing LLM

Read applicable `AGENTS.md`, `plan.md`, this plan and both handoffs; inspect code/raw artifacts, treating attached summaries as subordinate reference. Start **09-A**, then measured **09-B** and native **09-C**, preserving the running frozen experiment and all user changes. Add meaningful unit/integration tests, reproduce defects where feasible, verify installed behavior and raw-report regeneration, and avoid overengineering. Do not silently drop failed rows, change workloads to pass, merge incompatible snapshots, fabricate human/competitor results or select a license. Update handoffs after each assignment with commands, evidence and remaining gates. Complete Phase 09 before final Phase 10 acceptance.

## Historical execution checkpoint: 2026-10-04 07:41 UTC

The owner selected existing local 2B/4B/8B answering models, with Ollama Cloud used for independent judging and separate larger-model references. Evaluation credentials are protected outside the repository; the SDK has no Cloud requirement.

| Existing step | Verified progress | Remaining gate |
| --- | --- | --- |
| 1: correctness/resources | Reproduced and fixed Windows evaluation connection leak; optional pinned ONNX backend and explicit durable deferred indexing; all six local installed matrix cells pass | Fresh-query p95 still fails: minimal 85.88ms vs 25ms, ONNX 143.88ms vs 100ms. Default inline semantic append and default PyTorch RSS misses remain |
| 2: practical evaluation | Checksum-bound immutable corpus cache, isolated per-question copies, bounded 64-item neural batches; fresh/cached rankings and prompts agree in regressions | Full S/M compute; this 12-logical-CPU, 16GB host has limited free memory under the actual workload |
| 3: frozen campaign | Analysis policy and source snapshots frozen; all 70,767 baseline identities retained | Full-S 2B run is active: checkpoint 126/4,500 rows, 14 complete cases. Remaining model/dataset tracks have not completed |
| 4: competitors | Actual pinned packages/source prepared; exact Qwen native template counter validated against Ollama; Mem0 expanded request dispatched | Mem0 native extraction timed out at 120s (8,062 input + 128 reserved tokens in 8,192); no complete ingestion. Graphiti/A-MEM/current Letta native adapters and full comparisons remain software work, not just missing credentials |
| 5: grading/review | Separate GPT-OSS 120B Cloud judge; complete 27-row development grading, partial-to-complete resume verified, reports independently regenerated; human agreement validator implemented | Human labels/rationales and adjudication; full holdout answers and grading. Cloud revisions are provider tags, not full weight hashes |
| 6: public ablations | Existing nine baseline profiles and synthetic feature fixtures retained | Planned public feature ablations/validated annotations and meaningful intervals. LongMemEval's one connected cluster cannot qualify a cluster CI |
| 7: release checks | Same 104,013-byte wheel across Windows/Linux Python 3.9/3.10/3.12; fresh isolated PyTorch and ONNX offline/recovery checks | Hosted CI unexecuted; failed performance gates remain visible; rerun qualification if SDK changes |
| 8: owner/docs | Handoffs, raw evidence and continuation commands updated; no publication | Owner license choice, completed evaluation, remaining resource/CI gates |

The 31B Cloud reference and Cloud judging experiments are development diagnostics, not substitutes for the complete frozen holdout. Do not merge different SDK/evaluator snapshots or interpret the 27-row development result as superiority evidence.

Keep the product self-hosted and the default integration unchanged:

```python
import nsn
nsn.init('./agent-memory')
model = nsn.wrap(existing_model)
```

## Required inputs and decisions

| Input | Who provides it | What it enables |
| --- | --- | --- |
| An available machine and sustained execution window for answering models, management models and an independent judge | Owner/operator | Complete frozen runs rather than repeated smoke qualification |
| Independent judge checkpoint/configuration and human reviewer(s) | Owner/operator plus evaluator | Reference grading, calibration and completed review labels/rationales |
| Actual competitor services/configurations | Implementer; operator provides any unavailable service access | Graphiti graph service, current Letta service and A-MEM provider; qualified Mem0 extraction windows |
| Access to execute the declared hosted CI workflow | Repository owner/operator | Hosted matrix evidence; local checks do not substitute for hosted execution |
| Explicit NSN source-license choice | Owner | Redistribution metadata and public release readiness; no inferred license |

Selected policy: keep local answering/management models and use the authorized Cloud credential for independent judging and larger-model references. Measure the pilot before selecting additional compute; do not buy hardware from an assumed runtime estimate. The library must never acquire a cloud dependency because evaluation uses one. Human review remains required with a separate judge.

## Phase 09: close prerequisites before freezing expensive runs

### 1. Reproduce and repair the remaining correctness/resource issues

**Windows Python 3.12 consolidation:** the current installed full matrix passes; historical one-second deadline failures remain preserved and their cause was not confirmed. Do not invent a repair or redo passing checks without a new change/concern. If the failure recurs, instrument enqueue, lease claim, source loading, validation and commit separately, including contention/SQLite waits. Retain deadline semantics and add a regression for a confirmed defect.

**Unique-query retrieval:** profile query compilation, FTS selection/ranking, facts scan, dense search, candidate resolution and packing on the existing 10k workload. Latest retained p95 is 85.88ms minimal and 143.88ms optional ONNX, versus 25/100ms goals. Investigate measured SQL/query-plan and packing costs before introducing another index; previous cache-size experiments did not establish an improvement. The repeated-query cache must not hide this workload. Preserve ranking, status filters, exact identifiers, namespace/session boundaries and budgets. Add equivalence/invalidation regressions if changing candidate selection.

**Semantic footprint:** separate interpreter/import overhead, encoder weights, inference allocations and vector/cache memory. Current incremental runtime is about 533 MiB versus the 400 MiB goal. First remove measured redundant allocations/imports and bound caches. If the encoder backend itself prevents the target, run one bounded development experiment with a leaner local CPU inference backend or smaller/quantized encoder. Keep it optional, offline and fingerprinted; compare real paraphrase/identifier quality before adopting it. A smaller synthetic hashing encoder is not an acceptable substitute for neural qualification. Preserve the old profile and do not rename a failed target as a pass.

**Semantic foreground append:** measure actual `Runtime.append_event` at the declared 10k scale, with query/index jobs both enabled and disabled as explicitly different configurations. Include encoding and durable outbox costs; the existing raw-write-plus-batch-index profile does not qualify inline append. If encoding dominates, assess an explicit deferred-index option using the existing durable jobs: no new scheduler/framework. Test pending-vector visibility, restart replay, deletion during encoding, supersession and concurrent lease ownership. Document the resulting consistency contract; do not silently change default behavior.

**Exit evidence:** unchanged functional assertions pass; Windows 3.12 full installed suite passes reproducibly after the repair; original isolated workloads are rerun on declared hardware; every resource target either passes or remains visibly incomplete. Fixes here are prerequisites to trustworthy Phase 09 experiments, not a declaration that Phase 10 is complete.

### 2. Make full evaluation practical without changing the experiment

The current runner already implements immutable native corpus/index reuse keyed by source checksum, complete event-content identity, SDK/evaluator identities and encoder fingerprint. Verify its share of remaining run time and retain isolated mutable copies per question/system. Do not share question-derived facts, previous predictions, gold answers, judge state or tuned results between questions/systems. Record one-time build costs and per-question costs separately instead of erasing ingestion overhead; do not rebuild this facility without a measured need.

Test reuse against fresh reconstruction: identical ranked candidates, evidence, namespaces and input prompts; changed content/configuration must invalidate; crash/resume must not duplicate rows. Keep this small and file/SQLite based. No distributed benchmark platform is needed.

Run a fixed development pilot across the three answering sizes, full S/M corpus lengths and required competitor ingestion. Measure ingestion/build/answer/judge time and peak memory. Use the pilot for resource provisioning, not a replacement for holdout coverage. Estimate total compute as measured build time plus required calls times measured call latency, including management and grading calls. Do not promise a completion date before this measurement.

### 3. Freeze the final campaign

After prerequisite changes, freeze SDK source, evaluator/scorer revisions, asset hashes, profiles, prompts, tokenizer/window handling, model digests, hardware, seed, tuning budget and analysis policy. Use new run directories; previous partial rows belong to older SDK snapshots. Resume only an identical experiment; a substantive SDK/evaluator change requires a new campaign.

Keep the verified datasets and partitions: LoCoMo has 1,621 holdout QA; LongMemEval S and M each have 500. Complete all designated model/profile comparisons. With nine existing profiles and three model sizes, running LoCoMo once and S/M separately entails **70,767 baseline question/profile/model records**; competitor, ablation, management and judge calls are additional. Declare this arrangement before running it and keep dataset identities distinct. Every required pair gets a row, including timeout, overflow, ingestion and retrieval failures.

LongMemEval cases currently connect into one cluster through shared sessions/content; LoCoMo has eight holdout conversation clusters. Inspect the scientific consequences before promising paired intervals. Retain the existing conservative cluster analysis. Do not treat correlated questions as independent or choose a primary comparison after seeing its scores. Predeclare the primary comparison and any supplemental analysis/independent-data requirement, acknowledging the partial holdout already inspected. If the required claim cannot be supported by the frozen clustering, that uncertainty gate remains open until an explicit justified study-design decision—not a denominator change—resolves it.

### 4. Qualify and run actual competitors

Implement remaining adapters for the pinned actual Mem0, Graphiti, current Letta and A-MEM versions. Test native role/date preservation, isolated namespaces, full ingestion, query execution, restart, complete management-call logs and source lineage where supported. Import success is not evaluation success. Development adapter fixtures must use actual SDK/service calls; test doubles can exercise failures but cannot supply quality scores.

Mem0's exact Qwen tokenizer/template counter is already qualified: the original 35,516-byte prompt needs 8,057 input tokens and cannot fit matched 2k. The expanded-local 8k request counted 8,062 + 128 and was dispatched, then timed out at 120s before ingestion. Diagnose that native timeout and validate returned counts before accepting untruncated extraction evidence. Preserve matched failure and separate expanded/recommended configurations. Do not shorten extraction prompts or remove native turns just to make the test pass.

Keep two tracks: identical agreed resource constraints, and actual documented recommended configurations with their full resource/token costs. Verify recommended settings from pinned vendor documentation/code when implementing. Expanded local 8k/16k experiments are not automatically recommended configurations. Missing services or credentials remain explicit exclusions, not fabricated zero accuracy or substituted RAG results. Full comparison acceptance stays open while required systems are unevaluated.

### 5. Complete answering, independent grading and human review

Run the frozen 2B/4B/8B answering campaign with all rows persisted and resumable. Verify expected identity coverage before reporting full accuracy. Retain full-context overflows and every other failure. Do not run another broad smoke round in place of completing coverage.

Use the pinned upstream deterministic scoring where applicable. Grade LongMemEval with an explicitly configured independent reference judge; log prompt, checkpoint, rubric, response, rationale and grading failure. Calibrate it on a predeclared development sample with human review. Existing same-family 8B judge results remain diagnostic until agreement is validated. Historical answering-model self-judge scores stay excluded permanently.

A human fills the existing frozen review workflow's labels/rationales; the AI may prepare and validate the queue, not impersonate the reviewer. Predeclare treatment of disagreement/uncertainty and adjudication. Report ungraded items with full-denominator bounds; never drop them from accuracy.

### 6. Execute public ablations and regenerate the final Phase 09 report

Evaluate the plan's lexical/dense/hybrid and minimal/semantic contrasts, pooled/token-vector representations, temporal state, provenance policy, graph, reranker, consolidation, facts-plus-raw, and context budgeting. Each comparison keeps questions, generator and budgets fixed; disabled components must actually be disabled, including management work. An unimplemented optional feature is reported as absent, not a successful ablation. Start development probes before frozen confirmatory runs; do not tune on holdout results.

Use a paired baseline for every contrast. Include raw scores and failures, retrieval/evidence support, valid abstention/stale/conflict annotations, complete relationship paths and applicable agent task success. Predeclare exploratory versus confirmatory analyses and multiple-comparison treatment. Cluster bootstrap intervals require scored pairs and sufficient independent clusters; more resampling iterations cannot fix one cluster. Synthetic graph/procedure/consolidation fixtures do not replace these public experiments.

Regenerate every reported value from raw rows using a separate report invocation. Audit manifests, denominators, source/model/SDK checksums, categories, failures, costs and recommended/matched labels. Update Phase 09 handoff and index only when all its acceptance gates are evidenced. A negative quality result is valid research; a superior result cannot be assumed. If the +5-point target is not met, report the miss and any necessary development iteration rather than inventing a win.

## Phase 10: qualify the final candidate after evaluation

### 7. Execute release compatibility and recovery checks

Build a clean wheel from the evaluated source. Install that exact artifact outside the checkout across Windows/Linux Python 3.9/3.10/3.12, with no inherited site-packages. Re-run the declared core suites, optional integration checks and fresh semantic dependency resolver. Test network-denied imports/operation, streaming cancellation, budget enforcement, namespace isolation, legacy WAL backup/migration, malformed backup rollback, crash recovery, durable indexing replay and prepared-model restart.

Execute the actual declared hosted CI matrix as well as local checks. Preserve failures and exclusions. Re-run isolated installed minimal and real semantic process-tree resource profiles after any runtime/dependency change; count encoder/children but exclude the answering model. Keep repeated/fresh queries and raw/batch/inline ingestion distinct. No weaker assertions, easier corpus or hardware result relabelled as a functional pass.

### 8. Close documentation and owner gates

Have the owner select the source redistribution license explicitly; review package/dependency/dataset notices against that choice. Keep downloaded datasets/model assets outside the wheel. Until the decision exists, leave license metadata unresolved.

Update `plan.md`, the implementation index, both handoffs, user guide, changelog, wheel metadata and checksums together. Include exact commands, environment/model/asset identities, raw-report links, limitations and continuation instructions. Review the final diff for compatibility, unnecessary complexity and source/data leakage. Check documentation claims against regenerated artifacts and the actual installed wheel.

No publication/deployment is part of this completion plan. All gates passing establishes a qualified local release candidate; any later publication requires a separate explicit instruction. Competitor leadership is claimed only if the completed comparable results support it. Frontier equivalence additionally requires a genuine frontier-reference experiment; completing the library does not establish it.

## Immediate next assignment

Memory verification checkpoint: [2026-10-04 report](results/10-memory-verification-20261004/report.md) qualifies core functional behavior on a new installed wheel, with 18 new tests and 300 passing functional tests. Historical/session filtering and closed maintenance operations were repaired. Retain the original Windows 3.9 optional-enrichment failure alongside separate confirmation. Resource and broader quality/release gates below remain open; use the new candidate for future qualification without relabeling earlier measurements.

Start work package **09-A**: repair physical JSONL parsing/reporting and verify continuation provenance. Then tackle measured resource fixes (**09-B**) and actual native competitor adapters (**09-C**). Preserve the active full-S process, snapshots and raw rows; do not start a duplicate writer or modify a frozen snapshot. Complete **09-D/E/F**, then final **10-A/B** acceptance. Arrange sustained compute, human review, hosted CI access and the owner's license choice while finishing independent work. New runtime changes start new campaigns rather than rewriting old rows. See both handoffs for commands and historical evidence.
