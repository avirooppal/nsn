# Changelog

Memory verification (2026-10-04): historical raw-event and semantic session eligibility now apply before retrieval limits; closed runtimes reject indexing and optional maintenance writes. Added 18 memory-contract tests, verified 300 functional tests and fresh installed minimal/ONNX behavior, and checked restart recall with a real local model. Earlier performance misses and the initial optional-enrichment compatibility failure remain recorded in [the verification report](docs/implementation/results/10-memory-verification-20261004/report.md).

Prototype hardening (2026-10-04): preserves BM25 event relevance, escapes evidence fields before budgeting, raises SQLite retrieval errors and rejects closed-runtime append/retrieval/fact calls. A separate candidate is verified outside the checkout; resource misses and incomplete release gates remain in [prototype readiness](docs/implementation/prototype-readiness.md). The original candidate and its results are retained.

## 0.3.0 — local candidate, unpublished

- Three-line `nsn.init` / `nsn.wrap` facade, scoped durable event storage, temporal facts and budgeted evidence.
- Optional compact semantic retrieval, source-backed relationship paths, explicit recoverable consolidation and validated procedures.
- Supported sync/async text and OpenAI-format streaming lifecycle; optional text LangChain and authenticated host REST integrations.
- Corrected evaluation denominators, pinned public native data/scoring, real local development model runs, resumable frozen selection and honest judge/competitor limitations.
- Minimal wheel resources, network-denied installed verification, persistence example, Docker fixture, CI definitions and backup/migration documentation.
- Legacy backup now includes committed WAL records through SQLite backup. Semantic extras exclude legacy FAISS/spaCy; old `Memory` users install `nsn[legacy]`.
- Runtime-owned serialized SQLite connections reduce WAL/schema setup overhead; cross-owner idempotency remains transactional. A bounded SQL retrieval cache invalidates on internal/external writes and retains namespace isolation.
- Cancelled/deleted events are excluded before lexical and dense retrieval. Graph reads restore connection state and clear progress handlers after bounded traversal.
- Full S/M LongMemEval preparation supports checksum-verified range resume; frozen source/SDK identities, independent grading and immutable human-review samples prevent mixed experiments and invalid self-judge claims.
- Isolated process-tree resource profiles, fresh semantic dependencies and installed Windows/Linux matrix checks are recorded with the final wheel checksum. Failed resource gates remain visible; no release qualification is implied.
- Optional local ONNX MiniLM mean pooling and explicit deferred indexing reuse durable jobs; defaults remain inline and self-hosted. Full-haystack inference batches are bounded.
- Evaluation-only Ollama Cloud support validates provider revisions, separates independent judging/larger references from local SLMs, and keeps credentials outside artifacts. Immutable native-corpus reuse isolates each question's mutable state.

No public release, universal compatibility, full benchmark qualification or superiority is implied. Phase handoffs record evidence and open gates.
