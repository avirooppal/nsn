# NSN implementation handoff pack

Created: 2026-10-03. Updated: 2026-10-04. Status: phases 00-08 implemented; Phase 09 partially evaluated with full S/M inputs verified; Phase 10 local candidate tested with remaining resource, evaluation, hosted-CI and owner-license gates. See handoffs for exact evidence and limitations.

Use this folder to give another LLM one focused implementation phase at a time. Root [plan.md](../../plan.md) remains the architecture/research reference. Each numbered file includes the task, inspection paths, acceptance criteria, recommendations, and handoff requirements.

For the remaining work, follow the [completion execution plan](completion-plan.md): repair the measured prerequisites, freeze and complete Phase 09 evaluation, then qualify Phase 10. It lists required compute, grading, service and owner inputs; existing partial statuses remain unchanged.

The updated execution order reuses the current assets: 09-A JSONL/reporting repair, 09-B measured resource fixes, 09-C native competitor adapters, 09-D frozen full evaluation, 09-E independent/human/reference checks, 09-F public ablations/report audit, then 10-A installed qualification and 10-B documentation/license. These are assignments inside the existing phases. Production readiness, measured SLM improvement and competitor/larger-model claims have separate evidence gates.

Earlier checkpoint (2026-10-04 07:41 UTC): the full-S 2B run had 126/4,500 rows; that historical checkpoint does not establish a currently active process; the 70,767-row local campaign remains incomplete. Cloud independent development grading/reference runs completed 27 rows; human labels, full competitors and public ablations remain open. Current wheel SHA is `ce32c7573366aa9d706889b042224f2d974edf2197e17aed34dac8fcc79e1431`; all six local installed compatibility cells pass. Optional ONNX/deferred resource gains are measured separately from retained fresh-query/default-semantic misses. See [Phase 09](handoffs/09.md) and [Phase 10](handoffs/10.md) for reproducible evidence; hosted CI and owner license remain unresolved.

Latest requested comparison: [NSN versus non-NSN measured report](results/09-nsn-vs-no-nsn-20261004/report.md) completes 420 original rows plus 105 separate failed 4B-template experiment rows on 15 frozen public questions. Mixed quality results, provisional independent judging and performance/coverage limitations remain explicit; Phase 09 and 10 statuses stay partial.

Latest runtime candidate: [prototype readiness](prototype-readiness.md) records targeted ranking, evidence-boundary, storage-error and lifecycle repairs, a separately built wheel, fresh six-cell installed checks and honest resource/quality limits. This supersedes ce32 as the current SDK candidate; its historical benchmark remains intact.

Latest memory verification: [installed memory-contract report](results/10-memory-verification-20261004/report.md) records 18 new tests, historical/session retrieval and closed-runtime repairs, 300 passing functional tests, fresh offline minimal/ONNX installations and an actual local-model restart smoke. Candidate is now `dist/memory-verification-20261004/nsn-0.3.0-py3-none-any.whl`, SHA-256 `423370a5b09e7104627dbad911e39c6a8b3f8c0c974d2dc6799ed6e726af6f3b`. Earlier candidate performance/QA artifacts remain historical. Phases 09/10 remain partial; the first Windows 3.9 optional-enrichment failure and separate confirmation are both retained.

## Product requirements from the user

- Lightweight, self-hosted memory and context for SLMs and AI agents.
- Few-line setup: `import nsn`, `nsn.init('')`, `nsn.wrap(model)`.
- Advanced capabilities available through optional commands and integrations.
- Architecture informed by cited research.
- Reproducible evaluation against competitors, aiming for best performance.

Memory-dependent task improvement is testable. General frontier intelligence transfer and universal superiority are not established by the current implementation.

## Recommended build order

I recommend smaller phases than P0–P7 in the original plan. Basic facts and retrieval come before the public wrapper so its claims can be tested against working memory. Graph, consolidation, and procedures are separate gates to prevent a large speculative rewrite.

| Phase | Brief | Dependency | Outcome | Status |
| --- | --- | --- | --- | --- |
| 00 | [Repair evaluation integrity](00-evaluation-integrity.md) | None | Make failures visible and establish a trustworthy way to measure later changes. | Complete |
| 01 | [Scoped durable storage and runtime ownership](01-durable-storage.md) | 00 | Establish one recoverable source of truth without cross-namespace access or silent data loss. | Complete |
| 02 | [Facts, corrections, history, and provenance](02-facts-and-history.md) | 01 | Preserve changing knowledge and distinguish evidence from generated claims. | Complete |
| 03 | [Useful minimal retrieval and SLM context packs](03-retrieval-and-context.md) | 02 | Deliver useful offline memory under a strict small-model context allowance. | Complete |
| 04 | [Three-line SDK and supported model adapters](04-sdk-and-minimal-package.md) | 01–03 | Make the intended integration actually work in a clean installation. | Complete |
| 05 | [Optional compact semantic retrieval](05-compact-semantic-memory.md) | 04 | Improve paraphrase recall while keeping installation and runtime cost measured. | Complete |
| 06 | [Source-backed bounded relationship retrieval](06-bounded-graph.md) | 05 | Improve multi-hop evidence access without inventing relationships or requiring a graph server. | Complete — optional; development ablation only |
| 07 | [Recoverable consolidation and optional local enrichment](07-consolidation.md) | 06 | Organize memory in bounded background work without losing or manufacturing evidence. | Complete - explicit; no measured retrieval gain |
| 08 | [Validated procedures and advanced integration surface](08-procedures-and-integrations.md) | 07 | Support agent experience and advanced users without increasing default setup complexity. | Complete - explicit procedures; optional integrations |
| 09 | [Public benchmarks, real SLMs, and competitors](09-real-evaluation.md) | 08 | Determine whether NSN improves memory-dependent answers and where it stands against actual competitors. | Partial — full S/M inputs frozen; holdout coverage, reference/human grading, public ablations and competitor gates open |
| 10 | [Clean installation, migration, documentation, and release readiness](10-release-and-documentation.md) | 09 accepted or incompleteness documented | Deliver a reproducible self-hosted product with a simple default and accurate documentation. | Partial — installed wheels, fresh semantic dependencies and local matrix checked; resource/license/hosted CI/evaluation gates open |

**First useful milestone:** complete Phases 00–04 for reliable offline memory and the requested API. Optional semantic memory is Phase 05. Keep each optional feature disabled until its quality/resource gate passes. Phase 09 still evaluates the simpler profiles as separate systems.

## Master prompt to give the building LLM

Copy this and replace the phase filename:

```text
Implement NSN phase docs/implementation/00-evaluation-integrity.md.

Read applicable AGENTS.md, plan.md, docs/implementation/README.md,
the selected phase brief, and predecessor handoffs first.

Inspect the actual code before editing. Implement the selected phase,
preserve existing user changes, and verify its acceptance criteria.
Prefer small compatible changes; do not rewrite the whole project.

Follow the shared requirements and scope boundaries in these documents.
Research-inspired changes remain hypotheses until measured.
Do not fabricate scores or skip failing benchmark questions.
Do not add cloud processing or model downloads to the default path.

Write the required phase handoff with exact verification commands/results,
decisions, changed contracts, limitations, and next-phase prerequisites.
Update the index status accurately. Stop after the selected phase.
```

For a new LLM conversation, supply the repository plus the selected brief, this index, root plan, and predecessor handoffs. The LLM must inspect files rather than rely only on a pasted summary. If a predecessor's gate failed, repair the necessary prerequisite or report the gap; do not assume it passed.

## My additional suggestions

1. **Get a real end-to-end example working early.** A fixture should prove the actual model sees prior evidence; a local SLM smoke run should prove persistence can change its answer. Tests of helper functions alone cannot verify the product promise.
2. **Store events first and derive facts second.** User assertions, questions, tool observations, and assistant output have different meanings. Never turn an assistant's guess into trusted memory by repetition.
3. **Start with structured facts and conservative extraction.** Automatic interpretation of arbitrary text is a research problem. Preserve uncertain text and expose that uncertainty instead of losing information.
4. **Use SQLite as the authoritative store.** FTS, vectors, graphs, summaries, and caches are derived artifacts. They should be rebuildable and unable to resurrect deleted facts.
5. **Make the simple path truly small.** Move heavy ML libraries to extras, load lazily, and provision semantic assets explicitly. Report minimal and semantic profiles separately.
6. **Measure useful evidence per token.** SLMs benefit from small precise context. Preserve dates, identifiers, negation, conflicts, and complete paths; avoid redundant summaries.
7. **Avoid unnecessary ANN and orchestration.** Compare pooled-vector exact search first at small corpus sizes. Introduce indexing/worker complexity only when measurements justify it.
8. **Treat history as a first-class capability.** “Current truth” and “what was true then” require different queries. Preserve corrections, delayed events, and unresolved conflicts.
9. **Keep graph and sleep optional.** Their names do not prove usefulness. Accept each feature only when ablation shows value within the resource budget.
10. **Offer supported adapters, not a universal promise.** Start with text callables and OpenAI-compatible chat completions; unusual clients need explicit adapter contracts.
11. **Define memory failure policy.** Default should surface errors clearly. An explicit fail-open mode may call the original model without memory, but must report degraded operation; never silently return a fabricated fallback.
12. **Make immediate recall behavior explicit.** Durably captured events should be searchable through lexical/pending-tail retrieval immediately; expensive semantic/fact enrichment may lag. flush provides the documented consistency barrier.
13. **Version plans and handoffs.** The existing .gitignore excludes plan.md. If implementing agents are expected to share work through Git, include the plan deliberately and keep handoffs tracked; do not rely on an ignored file reaching another checkout.
14. **Judge both quality and efficiency.** Report accuracy, stale answers, abstention, latency, RAM, disk, tokens, and all background model work. Publish losses as well as wins.

These are my recommendations. They refine implementation choices while retaining the user's requirements. If evidence favors a different choice, record the tradeoff rather than follow a suggestion mechanically.

## Shared engineering rules

- Preserve legacy public imports and data through documented compatibility/migrations.
- Read the actual current tree and use the project's tools; do not assume a command in an old README works.
- Do not install mandatory cloud services or use secrets/paid endpoints without an explicit configured scope.
- Use purpose-driven regressions for isolation, temporal updates, deletion, recovery, token budgets, adapters, and benchmark coverage.
- Mark initial resource targets as targets until measured on declared hardware/workloads.
- Where a tokenizer is absent, label token counts as estimates.
- Existing tests passing does not establish fairness or SLM/competitor quality.
- A phase completion report must cite evidence for each gate. Unavailable data, hardware, models, or compatibility results remain visible.
- Preparation of release artifacts does not authorize external publication or deployment.

## Handoff template

Create the phase's required file in `handoffs/` with:

```markdown
# Phase NN handoff
Status: COMPLETE / PARTIAL / BLOCKED
Revision / working-tree description:
Prerequisites inspected:

## Behavior implemented
## Files changed
## Contracts and decisions
## Acceptance criteria and evidence
## Commands and results
## Existing failures and new limitations
## Migration / dependency / asset changes
## Next phase prerequisites and suggested commands
```

Use honest phase-local status. Do not call a phase complete because its budget or context ran out. A new LLM resumes from the handoff and repository state.

