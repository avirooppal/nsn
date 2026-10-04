# Phase 00 — Repair evaluation integrity

Status: NOT STARTED. Mapping to original plan: P0.
Prerequisites: None. This is the first implementation phase.

## Copy-paste implementation prompt

You are implementing one phase of NSN, a lightweight self-hosted memory layer for SLMs and AI agents. The required integration is `import nsn; nsn.init('./agent-memory'); model = nsn.wrap(model)`. Memory improves task context; frontier-level intelligence and competitor leadership must be demonstrated, not assumed.

Read the repository's applicable AGENTS.md, root plan.md, docs/implementation/README.md, this phase brief, and predecessor handoffs. Treat README/paper/report claims as evidence to inspect rather than proof. Follow the human's current instructions over document suggestions. Preserve existing user changes. Implement only this selected phase and its necessary prerequisites; do not automatically start later phases.

**Objective:** Make failures visible and establish a trustworthy way to measure later changes.

**Inspect first:** `benchmarks/runners/evaluator.py`, `benchmarks/metrics/`, `benchmarks/baselines/`, `benchmarks/run_head_to_head.py`, `benchmarks/run.py`, and committed reports.

## Implementation tasks

1. Freeze question IDs and original evidence requirements before ingestion. Evaluate every eligible question for every system. Missing/rejected required evidence stays in the denominator, including partially missing evidence. Equivalent retained evidence needs an explicit validated mapping.
2. Separate retrieval diagnostics, deterministic extraction fixtures, and actual model-generated answers. Rename mislabeled baselines. Add a common generator interface with a mock for tests and a configured real model for experiments; never silently replace a failed real model with extraction.
3. Correct BM25 retrieval and baseline descriptions. Preserve a pooled-vector baseline distinct from the existing token-vector implementation; defer its new backend to Phase 05 if necessary and label the interim baseline honestly.
4. Pass requested seeds through generators. Use isolated temporary run directories and an artifact manifest rather than filename assumptions. Test repeated runs cannot reuse old indexes.
5. Generate rankings from actual values; remove unconditional BEST labels. Include total/evaluated/failed questions, profile, dataset hash, revision, hardware, model settings, raw logs, and stage timing.
6. Keep historical reports intact with a new audit note. Capture a baseline run only if configured resources exist; report unavailable data/models explicitly.

## Acceptance criteria

- [ ] A fixture with four questions and rejected gold evidence still reports four evaluated questions and counts missing evidence as failure.
- [ ] A multi-evidence fixture cannot improve recall by dropping unavailable gold IDs.
- [ ] A crashing generator records a failed question without a fallback answer; retrieval and generation timers remain distinct.
- [ ] Repeated isolated runs have identical input hashes and no leftover index contamination; different requested seeds reach the generator.
- [ ] Existing benchmark tests plus new integrity regressions pass, or previously incorrect expectations are updated with explanations.

## My recommendation

My recommendation: make fixed coverage and honest baseline labels mandatory before tuning anything. Add a tiny end-to-end local-model smoke fixture now, but reserve costly public runs for Phase 09.

This is an engineering recommendation, not a new user requirement. Any deviation should be explained with evidence and recorded in the handoff.

## Scope boundaries

Do not optimize NSN scores, introduce graph features, or publish new superiority claims.

## Required handoff

Write `docs/implementation/handoffs/00.md` after this phase. Record status (complete/partial/blocked), changed files, decisions, public/schema contracts, exact commands and outcomes, acceptance criteria evidence, existing failures, migration/dependency changes, limitations, and prerequisites for the next phase. Add next-step commands that another LLM can execute. Never mark incomplete criteria as passed. Update the implementation index status only when its gate is met.

In your final response, summarize the implemented behavior, verification, material limitations, and handoff path. Do not claim later phases are complete.

