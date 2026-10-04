# Phase 06 — Source-backed bounded relationship retrieval

Status: COMPLETE (optional capability; development evidence evaluation). See [handoff](handoffs/06.md). Mapping to original plan: P5 graph portion.
Prerequisites: Phase 05 accepted; Phase 02 fact semantics are required.

## Copy-paste implementation prompt

You are implementing one phase of NSN, a lightweight self-hosted memory layer for SLMs and AI agents. The required integration is `import nsn; nsn.init('./agent-memory'); model = nsn.wrap(model)`. Memory improves task context; frontier-level intelligence and competitor leadership must be demonstrated, not assumed.

Read the repository's applicable AGENTS.md, root plan.md, docs/implementation/README.md, this phase brief, and predecessor handoffs. Treat README/paper/report claims as evidence to inspect rather than proof. Follow the human's current instructions over document suggestions. Preserve existing user changes. Implement only this selected phase and its necessary prerequisites; do not automatically start later phases.

**Objective:** Improve multi-hop evidence access without inventing relationships or requiring a graph server.

**Inspect first:** `graph/`, graph storage methods, source support schema, retrieval planner, and research references in plan.md.

## Implementation tasks

1. Introduce scoped entities, aliases, relations, temporal validity, and support links in SQLite. Entity identity and every traversal include namespace.
2. Distinguish explicit factual predicates from co-occurrence/association. Store support spans and extraction confidence; do not turn RELATED_TO into a logical rule.
3. Support structured relationships and conservative local extraction. Route uncertain reference/entity resolution to explicit unresolved records.
4. Implement bounded traversal by depth, node/edge count, candidate count, time, and token budget. Make limits visible in diagnostics.
5. Integrate relation paths as evidence candidates with their required source records and temporal status.
6. Measure graph on/off on a frozen multi-hop development set; keep it optional when it cannot justify its cost.

## Acceptance criteria

- [x] Two namespaces with an Alice entity cannot share nodes, aliases, edges, or source memories.
- [x] Historical/current relation versions yield appropriate paths, and a complete-path metric requires all necessary evidence.
- [x] Ambiguous co-occurrence cannot be presented as a proved relation.
- [x] Cycles, dense neighborhoods, missing support, and deletions obey bounds without stale evidence.
- [x] Graph ablation reports complete evidence-path correctness/coverage plus latency and storage changes, including regressions. Generated-answer accuracy remains Phase 09.

## My recommendation

My recommendation: keep graph storage embedded and make structured relations the correctness reference. A small reliable graph beats a large graph full of guessed verbs.

This is an engineering recommendation, not a new user requirement. Any deviation should be explained with evidence and recorded in the handoff.

## Scope boundaries

Do not add Neo4j as a default dependency or claim graph gains before matched ablations.

## Required handoff

Write `docs/implementation/handoffs/06.md` after this phase. Record status (complete/partial/blocked), changed files, decisions, public/schema contracts, exact commands and outcomes, acceptance criteria evidence, existing failures, migration/dependency changes, limitations, and prerequisites for the next phase. Add next-step commands that another LLM can execute. Never mark incomplete criteria as passed. Update the implementation index status only when its gate is met.

In your final response, summarize the implemented behavior, verification, material limitations, and handoff path. Do not claim later phases are complete.

