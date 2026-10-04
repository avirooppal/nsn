# Phase 01 — Scoped durable storage and runtime ownership

Status: NOT STARTED. Mapping to original plan: P1 (storage portion).
Prerequisites: Phase 00 accepted.

## Copy-paste implementation prompt

You are implementing one phase of NSN, a lightweight self-hosted memory layer for SLMs and AI agents. The required integration is `import nsn; nsn.init('./agent-memory'); model = nsn.wrap(model)`. Memory improves task context; frontier-level intelligence and competitor leadership must be demonstrated, not assumed.

Read the repository's applicable AGENTS.md, root plan.md, docs/implementation/README.md, this phase brief, and predecessor handoffs. Treat README/paper/report claims as evidence to inspect rather than proof. Follow the human's current instructions over document suggestions. Preserve existing user changes. Implement only this selected phase and its necessary prerequisites; do not automatically start later phases.

**Objective:** Establish one recoverable source of truth without cross-namespace access or silent data loss.

**Inspect first:** `neurosleepnet/storage/`, `neurosleepnet/memory/schemas.py`, configuration files, current memory constructor, and Phase 00 handoff.

## Implementation tasks

1. Introduce schema-versioned migrations and scoped repository methods. Start with events, source metadata, schema migrations, jobs/outbox, and index state; add later tables when their owning phases need them.
2. Store raw event role, namespace, session, source identity, observed/event time, turn status, idempotency key, and retention metadata. UTC storage is explicit; unknown event time remains unknown.
3. Enforce namespace on ID reads/deletes, list/search, duplicate checks, caches, and any legacy graph/vector access still exposed. Until a backend is safe, isolate it physically or disable it clearly.
4. Commit events and enrichment/outbox jobs atomically. Introduce bounded transactions, connection handling, writer/index ownership, leases with recovery, and visible failure reporting.
5. Implement lifecycle foundations for flush/close and restart replay. Define what flush guarantees and how timeout/failure is returned.
6. Back up and migrate a legacy database fixture without changing its original raw content. Rebuild derived indexes instead of trusting incompatible old files.

## Acceptance criteria

- [ ] Two namespaces sharing a database cannot read/delete/deduplicate each other's identical-content records, even by known ID.
- [ ] Crash injection around commit and index processing preserves committed events and replays pending work exactly once in effect.
- [ ] Concurrent writers/readers obey the documented ownership policy without corrupted state or duplicate jobs.
- [ ] Migration is repeatable, versioned, preserves legacy evidence, and fails visibly on unsupported schema.
- [ ] Flush reports pending-job failure rather than falsely claiming consistency; close is idempotent.

## My recommendation

My recommendation: avoid creating every proposed table immediately. SQLite plus a transactionally written outbox is enough to establish durability; add schemas only with a concrete consumer.

This is an engineering recommendation, not a new user requirement. Any deviation should be explained with evidence and recorded in the handoff.

## Scope boundaries

Do not add a distributed queue, graph server, automatic model download, or a second authoritative data store.

## Required handoff

Write `docs/implementation/handoffs/01.md` after this phase. Record status (complete/partial/blocked), changed files, decisions, public/schema contracts, exact commands and outcomes, acceptance criteria evidence, existing failures, migration/dependency changes, limitations, and prerequisites for the next phase. Add next-step commands that another LLM can execute. Never mark incomplete criteria as passed. Update the implementation index status only when its gate is met.

In your final response, summarize the implemented behavior, verification, material limitations, and handoff path. Do not claim later phases are complete.

