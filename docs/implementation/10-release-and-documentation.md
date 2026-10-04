# Phase 10 — Clean installation, migration, documentation, and release readiness

Status: PARTIAL — local candidate implemented and verified; see [handoff](handoffs/10.md) for remaining publication/platform/evaluation gates. Mapping to original plan: P7.
Prerequisites: Phase 09 accepted or honestly documented as incomplete; performance claims require completed evidence.

## Copy-paste implementation prompt

You are implementing one phase of NSN, a lightweight self-hosted memory layer for SLMs and AI agents. The required integration is `import nsn; nsn.init('./agent-memory'); model = nsn.wrap(model)`. Memory improves task context; frontier-level intelligence and competitor leadership must be demonstrated, not assumed.

Read the repository's applicable AGENTS.md, root plan.md, docs/implementation/README.md, this phase brief, and predecessor handoffs. Treat README/paper/report claims as evidence to inspect rather than proof. Follow the human's current instructions over document suggestions. Preserve existing user changes. Implement only this selected phase and its necessary prerequisites; do not automatically start later phases.

**Objective:** Deliver a reproducible self-hosted product with a simple default and accurate documentation.

**Inspect first:** Packaging, Dockerfile/Compose, README, examples, test configuration, migration notes, and final evaluation reports.

## Implementation tasks

1. Build and install wheels in fresh environments outside the repository. Verify optional extras and resources on declared supported Windows/Linux/Python combinations.
2. Add CI for scope isolation, recovery, migration, import side effects, adapter contracts, token budgets, and offline execution; separate slow/model tests.
3. Repair Docker references and provide a local persistent-data configuration with optional prepared semantic assets. Keep the embedded library the primary experience.
4. Write a three-line quickstart, restart demo, local-model example, supported-adapter matrix, asset preparation, profile limitations, advanced APIs, and migration/backup guide.
5. Consolidate dashboard implementations only if useful to the release; defer a dashboard if it distracts from memory reliability.
6. Check plan.md's measured resource/correctness gates. Replace unsupported README performance claims with run-linked evidence and explicit limitations.
7. Prepare release artifacts, compatibility policy, changelog, and checksums. Do not publish packages or deploy externally unless the human has authorized that action.

## Acceptance criteria

- [x] The three-line integration and persistence example run from the clean installed wheel.
- [x] Minimal operation passes network-denied tests and starts without cloud keys/assets; semantic mode runs offline with explicit prepared assets.
- [x] Installed local adapter/platform checks and fresh semantic dependency resolution executed; exact successes and failures are recorded in the handoff. Hosted CI execution remains unverified.
- [ ] Migration preserves legacy rows and backup/recovery instructions/notices are included; an NSN redistribution license is still undeclared.
- [x] Every current public performance statement is linked to a reproducible run or labeled a goal; earlier misses are retained. Repeated-query targets pass. Optional ONNX/deferred RSS/append targets pass as a separate configuration; fresh-query latency, default inline semantic append and default PyTorch RSS misses remain visible.

## My recommendation

My recommendation: release the useful minimal milestone internally after Phase 04, then graduate semantic/graph/consolidation features based on evidence. Publication readiness and proven competitor leadership are different gates.

This is an engineering recommendation, not a new user requirement. Any deviation should be explained with evidence and recorded in the handoff.

## Scope boundaries

Do not publish, merge, deploy, or claim all-competitor leadership merely because packaging succeeds.

## Required handoff

Write `docs/implementation/handoffs/10.md` after this phase. Record status (complete/partial/blocked), changed files, decisions, public/schema contracts, exact commands and outcomes, acceptance criteria evidence, existing failures, migration/dependency changes, limitations, and prerequisites for the next phase. Add next-step commands that another LLM can execute. Never mark incomplete criteria as passed. Update the implementation index status only when its gate is met.

In your final response, summarize the implemented behavior, verification, material limitations, and handoff path. Do not claim later phases are complete.

