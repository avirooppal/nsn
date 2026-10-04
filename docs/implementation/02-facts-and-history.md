# Phase 02 — Facts, corrections, history, and provenance

Status: NOT STARTED. Mapping to original plan: P1 fact semantics + P3 updates.
Prerequisites: Phase 01 accepted.

## Copy-paste implementation prompt

You are implementing one phase of NSN, a lightweight self-hosted memory layer for SLMs and AI agents. The required integration is `import nsn; nsn.init('./agent-memory'); model = nsn.wrap(model)`. Memory improves task context; frontier-level intelligence and competitor leadership must be demonstrated, not assumed.

Read the repository's applicable AGENTS.md, root plan.md, docs/implementation/README.md, this phase brief, and predecessor handoffs. Treat README/paper/report claims as evidence to inspect rather than proof. Follow the human's current instructions over document suggestions. Preserve existing user changes. Implement only this selected phase and its necessary prerequisites; do not automatically start later phases.

**Objective:** Preserve changing knowledge and distinguish evidence from generated claims.

**Inspect first:** Event repository and migrations from Phase 01; `perception/detector.py`, trust modules, memory observation paths, and temporal regressions.

## Implementation tasks

1. Add facts, fact support spans, versions, statuses, and conflict groups. Represent typed values/units, qualifiers, observed/valid times, and predicate cardinality.
2. Offer structured fact ingestion first. Add conservative text heuristics with explicit coverage limits; retain raw text when extraction is uncertain.
3. Handle single-valued updates, set-valued additions, explicit preference corrections, out-of-order events, and current/as-of lookup. Unknown dates must not become invented chronology.
4. Deduplicate scoped exact repeats/idempotency keys. Semantic similarity alone cannot reject changes to values, dates, qualifiers, or negation.
5. Separate source authority, extraction confidence, support, importance, and relevance. Assistant replies remain assistant events unless a supported confirmation policy promotes a claim.
6. Retain unresolved conflicting claims and supporting records. Integrate source deletion with dependency invalidation instead of destructive contradiction pruning.

## Acceptance criteria

- [ ] Port 8080 followed by port 9090 retains both observations and yields correct current and historical fact versions.
- [ ] An older event ingested later does not incorrectly supersede a newer valid event.
- [ ] Set-valued membership additions do not erase existing members; units/qualifiers prevent invalid merges.
- [ ] Questions and fabricated assistant replies do not become confirmed facts; repeated replies do not count as independent corroboration.
- [ ] Equal-authority unresolved conflicts survive retrieval preparation; deleted support cannot leave a derived confirmed fact.

## My recommendation

My recommendation: start with explicit structured facts plus reliable raw-event retrieval. Arbitrary natural-language extraction is a separate quality problem; do not hide its uncertainty behind a trust score.

This is an engineering recommendation, not a new user requirement. Any deviation should be explained with evidence and recorded in the handoff.

## Scope boundaries

Do not train a model or implement all-pairs lexical contradiction deletion.

## Required handoff

Write `docs/implementation/handoffs/02.md` after this phase. Record status (complete/partial/blocked), changed files, decisions, public/schema contracts, exact commands and outcomes, acceptance criteria evidence, existing failures, migration/dependency changes, limitations, and prerequisites for the next phase. Add next-step commands that another LLM can execute. Never mark incomplete criteria as passed. Update the implementation index status only when its gate is met.

In your final response, summarize the implemented behavior, verification, material limitations, and handoff path. Do not claim later phases are complete.

