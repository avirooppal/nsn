# Memory-layer verification — 2026-10-04

Core memory behavior is verified on the installed candidate. This is functional prototype evidence, not complete production qualification or model-intelligence/competitor claims.

Wheel: `dist\memory-verification-20261004\nsn-0.3.0-py3-none-any.whl`, 104,567 bytes, SHA-256 `423370a5b09e7104627dbad911e39c6a8b3f8c0c974d2dc6799ed6e726af6f3b`. SDK source bytes match the wheel. Owner license remains unresolved; nothing published.

## Repairs and meaningful tests

Added 18 memory-contract tests. They check prior-memory injection with one model call, no self-recall, separate-process persistence, namespaces, history larger than context, token bounds, fact corrections and provenance deletion, export/restore/delete scope, encoder failure and durable replay, lifecycle guards, dense filtering scores/cache behavior, and prepared real ONNX offline scoping. Deterministic hash encoders and spy models verify mechanics; they are not semantic-quality or answer-accuracy evidence.

Reproduced and repaired future raw events appearing in historical retrieval, wrong-session dense hits exhausting top-k, and indexing/rebuild/relation/pin writes after runtime close. Temporal raw-event filtering uses event_time (otherwise observed_at) and UTC-aware SQLite date comparisons before limits. Dense event eligibility is applied before top-k while preserving matrix cache and score mapping. Namespace facts remain shared across sessions; session filters constrain events and are not tenant security boundaries. Scoped dense filtering scans eligible IDs, so its large-scale latency is not qualified.

## Actual verification results

- Full functional suite: **300 passed, 2 skipped, 1 deselected, 3 warnings in 166.18s (0:02:46)**. One hardware test deliberately deselected; two skips retained.
- Fresh minimal wheel outside checkout: network denied, no ML dependencies, unchanged import threads, restart recall, namespace isolation and bundled config passed.
- Fresh `[semantic-onnx]` installation outside checkout: **23 passed** (18 memory contracts and 5 encoder tests); prepared-model offline durable replay passed without Torch/Transformers imports.
- Windows first attempt: `{'3.9': False, '3.10': True, '3.12': True}`; separate confirmation: `{'3.9': True, '3.10': True, '3.12': True}`. Linux containers: `{'3.9': True, '3.10': True, '3.12': True}`. Core cells use installed wheels with explicit REST/legacy exclusions and unavailable ONNX asset skips.
- First Windows 3.9 run had 99 passed, 2 skipped, 2 deselected and one optional enrichment failure during concurrent workloads. The failure remains in windows-matrix.json. No assertion, workload, timeout or resource limit was relaxed. Its root cause is not proven; separate confirmation is recorded independently.
- Actual local `qwen3:1.7b`: stored nonce recalled on both subsequent and post-restart turns, prior evidence injected and other namespace empty. Single synthetic smoke only, not a comparable accuracy benchmark. See local-model-smoke.json for actual prompts/responses/model digest.

## Reproduce

Run from the repository unless stated otherwise:
```powershell
python -m pytest tests/ benchmarks/tests/ -m 'not performance' -q -p no:cacheprovider
python scripts/local_matrix.py --wheel dist/memory-verification-20261004/nsn-0.3.0-py3-none-any.whl --output windows-matrix-confirmation.json
python scripts/local_linux_matrix.py --wheel dist/memory-verification-20261004/nsn-0.3.0-py3-none-any.whl --output linux-matrix.json
python docs/implementation/results/10-memory-verification-20261004/regenerate.py
```

Install this wheel into new environments with no inherited packages. From outside the checkout run scripts/verify_install.py in the minimal environment, scripts/verify_onnx.py with prepared assets in the semantic environment, and absolute-path tests/test_memory_layer_contract.py plus tests/test_onnx_encoder.py there. local-model-smoke.py requires the existing local Ollama server and pinned model; no downloads are performed.

## Remaining gates

Phases 09 and 10 remain PARTIAL. Historical performance misses are retained in ../10-prototype-hardening-20261004/readiness.json; they are not measurements of this new candidate. Fresh-query latency, default inline semantic resource qualification, full public/competitor evaluation, human grading, hosted CI and source-license choice remain open. No hardware assertion was weakened or marked passed by functional verification.
