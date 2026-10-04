# Phase 07 — Recoverable consolidation and optional local enrichment

Status: COMPLETE (operational correctness verified; summary retrieval and enrichment opt-in). See [handoff](handoffs/07.md). Mapping to original plan: P5 consolidation portion.
Prerequisites: Phase 06 accepted; graph remains optional for minimal deployments.

## Copy-paste implementation prompt

You are implementing one phase of NSN, a lightweight self-hosted memory layer for SLMs and AI agents. The required integration is `import nsn; nsn.init('./agent-memory'); model = nsn.wrap(model)`. Memory improves task context; frontier-level intelligence and competitor leadership must be demonstrated, not assumed.

Read the repository's applicable AGENTS.md, root plan.md, docs/implementation/README.md, this phase brief, and predecessor handoffs. Treat README/paper/report claims as evidence to inspect rather than proof. Follow the human's current instructions over document suggestions. Preserve existing user changes. Implement only this selected phase and its necessary prerequisites; do not automatically start later phases.

**Objective:** Organize memory in bounded background work without losing or manufacturing evidence.

**Inspect first:** `sleep/engine.py`, compression modules, durable jobs/lineage, and the Zep/A-MEM/CLS references listed in plan.md.

## Implementation tasks

1. Implement sleep as a bounded job operation/report over changed neighborhoods: enrich, connect, summarize, validate, compact. Explicit invocation is supported without an always-running daemon.
2. Use job leases, retry limits, error states, idempotency keys, revisions, and transactionally committed outputs. Do not scan every pair of memories.
3. Build extractive summaries first, preserving support, time, conflicts, and exact values. Index derived records through the same durable pipeline as other records.
4. Add optional schema-constrained local-model enrichment with source spans, validation, timeout/resource caps, and unconfirmed proposals. Count its model work.
5. Invalidate/rebuild derived records when supporting facts change or are deleted. Keep original events except explicit retention/deletion.
6. Separate importance decay from truth and retention. Pin configured facts/preferences; expose opt-in scheduling rather than surprise full scans.

## Acceptance criteria

- [x] Repeated sleep does not duplicate summaries; interruption/restart resumes safely.
- [x] Consolidation preserves current/as-of correctness, conflicts, identifiers, units, and source lineage.
- [x] Deleted/corrected source records cannot persist inside summaries, vectors, or caches.
- [x] Failures are reported with counts and reasons; local enrichment cannot fabricate confirmed evidence.
- [x] Sleep on/off comparison records quality, cost, peak resources, and foreground latency; leave expensive unhelpful enrichment disabled.

## My recommendation

My recommendation: make consolidation non-destructive and incremental. Biological names can explain inspiration, but documentation should name the actual computational stages.

This is an engineering recommendation, not a new user requirement. Any deviation should be explained with evidence and recorded in the handoff.

## Scope boundaries

Do not delete lower-trust conflicting events or claim neuroscience proves the software's performance.

## Required handoff

Write `docs/implementation/handoffs/07.md` after this phase. Record status (complete/partial/blocked), changed files, decisions, public/schema contracts, exact commands and outcomes, acceptance criteria evidence, existing failures, migration/dependency changes, limitations, and prerequisites for the next phase. Add next-step commands that another LLM can execute. Never mark incomplete criteria as passed. Update the implementation index status only when its gate is met.

In your final response, summarize the implemented behavior, verification, material limitations, and handoff path. Do not claim later phases are complete.

