# Phase 05 — Optional compact semantic retrieval

Status: NOT STARTED. Mapping to original plan: P4.
Prerequisites: Phase 04 accepted; minimal product remains usable independently.

## Copy-paste implementation prompt

You are implementing one phase of NSN, a lightweight self-hosted memory layer for SLMs and AI agents. The required integration is `import nsn; nsn.init('./agent-memory'); model = nsn.wrap(model)`. Memory improves task context; frontier-level intelligence and competitor leadership must be demonstrated, not assumed.

Read the repository's applicable AGENTS.md, root plan.md, docs/implementation/README.md, this phase brief, and predecessor handoffs. Treat README/paper/report claims as evidence to inspect rather than proof. Follow the human's current instructions over document suggestions. Preserve existing user changes. Implement only this selected phase and its necessary prerequisites; do not automatically start later phases.

**Objective:** Improve paraphrase recall while keeping installation and runtime cost measured.

**Inspect first:** Embedding/vector modules, lazy package extras, index outbox, retrieval/packer APIs, and baseline manifest.

## Implementation tasks

1. Run a short measured backend/encoder comparison on frozen development queries. Compare normalized pooled vectors and exact search for small corpora before selecting ANN.
2. Define provider/backend protocols and persist vector model revision, dimension, preprocessing fingerprint, and index revision. Fail explicitly on incompatible assets.
3. Store compact vectors, avoiding per-token JSON payloads by default. Partition search by scope and support pending-tail visibility, incremental writes, tombstones, and rebuild.
4. Add explicit prepare-assets functionality and local asset paths. Initialization/import must not download; offline mode rejects remote processing.
5. Fuse lexical and dense ranks with RRF and deterministic tie-breaking. Record actual query embedding, search, and pack timings.
6. Add pooled dense/hybrid benchmark baselines and compare them with minimal and legacy token-vector diagnostics using matched encoders and workloads.

## Acceptance criteria

- [ ] Frozen paraphrase development queries improve relative to minimal mode without regressing identifier/update cases beyond a stated tolerance.
- [ ] Network-denied execution succeeds with prepared assets; missing assets give actionable errors.
- [ ] Dimension/revision mismatch is detected; deletion and replay/rebuild produce correct scoped results.
- [ ] Report cold/warm latency, incremental peak RSS, asset/disk size, and corpus length at 1k/10k events.
- [ ] Semantic imports remain optional and absent from minimal runtime; chosen backend has a documented measured reason.

## My recommendation

My recommendation: prefer one vector per fact/chunk and test a small quantized CPU encoder. Keep the current token-MaxSim approach as an experimental comparator until it earns its footprint.

This is an engineering recommendation, not a new user requirement. Any deviation should be explained with evidence and recorded in the handoff.

## Scope boundaries

Do not describe MiniLM token MaxSim as trained ColBERT or report semantic-mode scores as minimal-mode results.

## Required handoff

Write `docs/implementation/handoffs/05.md` after this phase. Record status (complete/partial/blocked), changed files, decisions, public/schema contracts, exact commands and outcomes, acceptance criteria evidence, existing failures, migration/dependency changes, limitations, and prerequisites for the next phase. Add next-step commands that another LLM can execute. Never mark incomplete criteria as passed. Update the implementation index status only when its gate is met.

In your final response, summarize the implemented behavior, verification, material limitations, and handoff path. Do not claim later phases are complete.

