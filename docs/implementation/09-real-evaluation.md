# Phase 09 — Public benchmarks, real SLMs, and competitors

Status: PARTIAL. Full LongMemEval S/M native inputs and frozen partitions verified; partial real holdout runs, independent grading and human-review support implemented. Full evaluation, human/reference validation, public feature ablations and competitor gates remain open. See [handoff](handoffs/09.md). Mapping to original plan: P6.
Prerequisites: Phases 00–08 accepted; optional components can be evaluated disabled when their gates failed.

## Copy-paste implementation prompt

You are implementing one phase of NSN, a lightweight self-hosted memory layer for SLMs and AI agents. The required integration is `import nsn; nsn.init('./agent-memory'); model = nsn.wrap(model)`. Memory improves task context; frontier-level intelligence and competitor leadership must be demonstrated, not assumed.

Read the repository's applicable AGENTS.md, root plan.md, docs/implementation/README.md, this phase brief, and predecessor handoffs. Treat README/paper/report claims as evidence to inspect rather than proof. Follow the human's current instructions over document suggestions. Preserve existing user changes. Implement only this selected phase and its necessary prerequisites; do not automatically start later phases.

**Objective:** Determine whether NSN improves memory-dependent answers and where it stands against actual competitors.

**Inspect first:** Repaired harness, official dataset sources and scoring, profile manifests, all handoffs, and plan.md sections 8–9.

## Implementation tasks

1. Implement native-schema conversion and official scoring for pinned LongMemEval and LoCoMo variants; preserve sessions/speakers/timestamps and dataset checksums/licenses.
2. Freeze development/evaluation conversation groups and tuning budgets. Start with development smoke runs, then complete designated evaluation sets for publishable claims.
3. Run the same configured real answering SLM for matched stateless, rolling-window, full-context, BM25, pooled dense, hybrid, and NSN comparisons. Add oracle evidence to identify reasoning limits.
4. Integrate actual pinned Mem0 OSS, Graphiti, Letta/MemGPT, and A-MEM where feasible. Refresh current official documentation/code before implementation. Log unavailable/incompatible systems rather than invent results.
5. Use separate matched-resource and recommended-configuration tracks. Record all extraction/consolidation/reranker calls, stage latency, cold/warm behavior, tokens, peak RSS, and disk.
6. Evaluate approximately 1–2B, 3–4B, and 7–8B local model sizes subject to hardware availability; record exact assets, quantization, tokenizer, context, prompt, and decoding.
7. Calculate paired conversation-cluster bootstrap intervals, per-category errors, abstention/stale-answer/conflict metrics, complete evidence coverage, and applicable task success. Add the plan's ablations.
8. Produce neutral reports and reproducible commands from raw manifests. Frontier reference is optional and separately configured; do not incur unconfigured paid calls.

## Acceptance criteria

- [x] Every compared development profile evaluates the same frozen eligible questions; failures remain in the denominator. Full evaluation sets remain pending.
- [x] Real model generation is verified and cannot silently fall back; retrieval-only and answer results are separate.
- [ ] Reports include full provenance, confidence intervals, category coverage, failures, excluded systems, and resource costs.
- [x] At least one independent rerun can regenerate summary values from raw artifacts.
- [x] If resources/models/data prevent a required run, mark the phase partially complete with exact missing inputs; never label superiority achieved.

## My recommendation

My recommendation: treat the +5-point gain and resource targets as hypotheses. Publish the quality/latency/RAM tradeoff and remaining reasoning gap even if another system wins.

This is an engineering recommendation, not a new user requirement. Any deviation should be explained with evidence and recorded in the handoff.

## Scope boundaries

Do not cherry-pick questions, tune on the test set, compare percentages from different papers as head-to-head evidence, or fabricate benchmark numbers.

## Required handoff

Write `docs/implementation/handoffs/09.md` after this phase. Record status (complete/partial/blocked), changed files, decisions, public/schema contracts, exact commands and outcomes, acceptance criteria evidence, existing failures, migration/dependency changes, limitations, and prerequisites for the next phase. Add next-step commands that another LLM can execute. Never mark incomplete criteria as passed. Update the implementation index status only when its gate is met.

In your final response, summarize the implemented behavior, verification, material limitations, and handoff path. Do not claim later phases are complete.

