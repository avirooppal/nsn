# Phase 04 — Three-line SDK and supported model adapters

Status: NOT STARTED. Mapping to original plan: P2, deliberately moved after basic retrieval.
Prerequisites: Phases 01–03 accepted; this is the first usable product milestone.

## Copy-paste implementation prompt

You are implementing one phase of NSN, a lightweight self-hosted memory layer for SLMs and AI agents. The required integration is `import nsn; nsn.init('./agent-memory'); model = nsn.wrap(model)`. Memory improves task context; frontier-level intelligence and competitor leadership must be demonstrated, not assumed.

Read the repository's applicable AGENTS.md, root plan.md, docs/implementation/README.md, this phase brief, and predecessor handoffs. Treat README/paper/report claims as evidence to inspect rather than proof. Follow the human's current instructions over document suggestions. Preserve existing user changes. Implement only this selected phase and its necessary prerequisites; do not automatically start later phases.

**Objective:** Make the intended integration actually work in a clean installation.

**Inspect first:** `sdk/wrapper.py`, `sdk/async_memory.py`, integration adapters, `__init__.py`, `pyproject.toml`, settings, and evidence-pack API.

## Implementation tasks

1. Add an importable nsn facade. init(directory, ...) returns a runtime and registers the documented default; init('') resolves a local default directory. wrap(model, ...) attaches the explicit/default runtime.
2. Move ML and integration dependencies into optional extras and use lazy imports. Verify wheel discovery and bundled configuration data; retain compatible neurosleepnet imports.
3. Implement shared call lifecycle: retrieve prior evidence, durably capture input, invoke once, record actual completion, enqueue enrichment. Honor all observation flags.
4. Support generic text callables and OpenAI-compatible chat completion clients first, including declared sync/async/stream paths. Supply explicit adapter hooks for other callable signatures.
5. Preserve messages, return types, chunks, exceptions, parameters, and tool calls. Define failed/cancelled/partial turn behavior. Unsupported entry points produce a clear diagnostic instead of silently bypassing memory.
6. Define pre-init errors, repeated init with compatible/incompatible settings, double wrapping, runtime sharing, context-manager cleanup, flush/close, and legacy init(model, ...) deprecation.
7. Create a runnable offline example using a deterministic model fixture and a separate local-model example for real generation.

## Acceptance criteria

- [ ] A built wheel installed outside the checkout supports exactly import nsn; nsn.init(path); model = nsn.wrap(model).
- [ ] Import has no file writes, model loading, network calls, or background thread creation; the minimal install works without Torch, FAISS, sentence-transformers, or spaCy.
- [ ] Spy-model tests verify previous memory reaches the actual invocation, the current input is not self-recalled, and caller message objects remain unchanged.
- [ ] Flags, single invocation, stream cancellation, async exceptions, tool messages, return identity/contracts, and double-wrap behavior are verified.
- [ ] Close/restart then recall works through the documented public API; legacy examples have explicit compatibility coverage.

## My recommendation

My recommendation: promise a small tested adapter matrix first. Universal wrapping cannot safely guess every object's input/output contract. Preserve the simple API while exposing adapter= for unusual clients.

This is an engineering recommendation, not a new user requirement. Any deviation should be explained with evidence and recorded in the handoff.

## Scope boundaries

Do not monkey-patch providers globally, silently choose a model, or require a cloud key.

## Required handoff

Write `docs/implementation/handoffs/04.md` after this phase. Record status (complete/partial/blocked), changed files, decisions, public/schema contracts, exact commands and outcomes, acceptance criteria evidence, existing failures, migration/dependency changes, limitations, and prerequisites for the next phase. Add next-step commands that another LLM can execute. Never mark incomplete criteria as passed. Update the implementation index status only when its gate is met.

In your final response, summarize the implemented behavior, verification, material limitations, and handoff path. Do not claim later phases are complete.

