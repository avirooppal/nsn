# Phase 08 — Validated procedures and advanced integration surface

Status: COMPLETE (engineering gate; see handoffs/08.md for evidence and limitations). Mapping to original plan: P5 procedures + P7 optional integrations.
Prerequisites: Phase 07 accepted.

## Copy-paste implementation prompt

You are implementing one phase of NSN, a lightweight self-hosted memory layer for SLMs and AI agents. The required integration is `import nsn; nsn.init('./agent-memory'); model = nsn.wrap(model)`. Memory improves task context; frontier-level intelligence and competitor leadership must be demonstrated, not assumed.

Read the repository's applicable AGENTS.md, root plan.md, docs/implementation/README.md, this phase brief, and predecessor handoffs. Treat README/paper/report claims as evidence to inspect rather than proof. Follow the human's current instructions over document suggestions. Preserve existing user changes. Implement only this selected phase and its necessary prerequisites; do not automatically start later phases.

**Objective:** Support agent experience and advanced users without increasing default setup complexity.

**Inspect first:** MemoryTool, LangChain/API modules, wrapper adapter contracts, source lineage, and any available agent trace fixtures.

## Implementation tasks

1. Represent procedures with prerequisites, steps, environment/tool constraints, outcomes, status, and source traces.
2. Capture observed successful/failed task traces or explicit confirmation. Proposed assistant workflows remain unverified until outcomes support them.
3. Retrieve procedures and failure warnings with provenance and compatible environment context; never execute stored instructions automatically.
4. Adapt MemoryTool and LangChain paths to the shared runtime/policy. Implement explicit invoke/ainvoke interception if those paths are declared supported.
5. Implement scoped timeline/explain/stats/export/import and namespace-level deletion using common repositories. Validate round-trip preservation and schema handling.
6. Keep REST/service dependencies optional. Add auth and request scoping if supporting a network-exposed multi-user deployment; bind the simple self-hosted example to localhost and document its trust boundary.

## Acceptance criteria

- [x] An invented success claim does not create a confirmed procedure; an observed outcome does.
- [x] Failed procedures retain failure status; environment-incompatible workflows are excluded or flagged.
- [x] Declared agent adapters pass the same observation, prior-recall, failure, and isolation contract tests.
- [x] Export/import round trip preserves roles, scopes, dates, versions, and lineage without silent overwrite.
- [x] Optional integrations can be absent without affecting import or minimal operation.

## My recommendation

My recommendation: use a small real task suite to evaluate procedure reuse. Recall metrics alone cannot show an agent learned to complete tasks better.

This is an engineering recommendation, not a new user requirement. Any deviation should be explained with evidence and recorded in the handoff.

## Scope boundaries

Do not turn stored procedures into execution authorization or make dashboards/framework packages mandatory.

## Required handoff

Write `docs/implementation/handoffs/08.md` after this phase. Record status (complete/partial/blocked), changed files, decisions, public/schema contracts, exact commands and outcomes, acceptance criteria evidence, existing failures, migration/dependency changes, limitations, and prerequisites for the next phase. Add next-step commands that another LLM can execute. Never mark incomplete criteria as passed. Update the implementation index status only when its gate is met.

In your final response, summarize the implemented behavior, verification, material limitations, and handoff path. Do not claim later phases are complete.

