# NSN versus non-NSN paired benchmark

This is a bounded exploratory run requested on 2026-10-04. It does not replace the complete Phase 09 campaign or establish production readiness, competitor leadership or frontier equivalence.

## Frozen inputs

- `sample.json` SHA-256: `2cd6da8d9fc77ad7c54ec113cd28fe0bb1b29af7feac8540840594affa32ff43`.
- Fifteen questions: nine LoCoMo development questions across all five categories and two conversation clusters; six full-haystack LongMemEval S questions, one per category, in one connected cluster. LoCoMo category 3 has one available development cluster in this selection. No answer/score-based selection or tuning.
- Qwen3 approximately 2B, 4B and 8B, Q4_K_M; separate Cloud Gemma4 31B reference. Exact local digests and Cloud catalog revision are in each manifest. Cloud weight hashes, quantization and server hardware are unavailable.
- Seven configurations: no memory (`stateless`), recent context (`rolling`), full context, BM25, hybrid lexical/dense RAG, NSN lexical and NSN semantic. Non-NSN algorithms use the common benchmark storage harness; the stateless model receives no retrieved memory.
- Same questions/system instruction, temperature 0, seed 42, thinking requested disabled (the installed 4B Thinking-only checkpoint does not honor that behavior), 2,048-token input window, 32 reserved output tokens and 512 UTF-8 bytes of packed reference evidence. Full-context overflow is a recorded failure; no silent truncation or substitute answer.
- Actual offline pinned MiniLM ONNX neural embeddings. Graph, consolidation, procedures and SDK wrapper dispatch are not exercised; this measures core memory retrieval and resulting QA.

## Metrics and interpretation

| Metric | Definition |
| --- | --- |
| LoCoMo answer quality | Pinned upstream category-aware per-question F1; failures count as zero |
| LongMemEval correctness | Pinned upstream rubric with independent Cloud GPT-OSS 120B, 512 output budget; human calibration remains pending |
| Missing grades | Full-denominator lower/upper bounds; no favorable subset averages |
| Paired differences | Same-question NSN contrasts against every non-NSN baseline; conversation-cluster bootstrap, seed 42, 2,000 samples |
| Uncertainty limits | Only two LoCoMo clusters; intervals are exploratory and weak for generalization. One LongMemEval cluster cannot yield a valid cluster interval |
| Retrieval evidence | Gold evidence recall, precision among selected evidence and complete-support rate; questions without released evidence labels excluded only from evidence metrics |
| Latency | Observed retrieval, generation and retrieval-plus-generation p50/p95; nonstreaming wall time. Percentiles select the nearest observed order statistic |
| First-use cost | Semantic vector materialization/shared caches can affect retrieval timing; raw records separately retain native ingestion/embedding/cache costs |
| Tokens | Actual provider-reported prompt/output counts; missing counts for attempted failed calls are explicit, never invented as zero-cost requests |
| Failures/abstention | Every expected identity retained, failure reason, total abstentions; no answering-model self-judge accuracy |
| Memory/disk | Shared client-harness RSS and disk explicitly labeled. Incremental NSN footprint comes from separate isolated installed 10k profiles, not subtracting mixed harness samples |
| Management calls | Actual recorded extraction/consolidation calls; these disabled configurations make zero such calls |

No aggregate mixes LoCoMo F1 with LongMemEval binary grading into a single “accuracy.” Category results and responses remain in the raw/regenerated reports. Full reports are independently regenerated from complete 105-row runs per model and validated for matched question/profile/budget/source/encoder inputs.

Validated stale/conflict/hallucination annotations and agent task success are unavailable in this QA sample. Dollar billing, remote peak memory and time-to-first-token are unavailable. Those metrics must stay unqualified rather than being fabricated or inferred from unrelated scores. Human review queues are prepared separately and contain no AI-generated human labels.

## Separately recorded failed serving-configuration attempt

The completed native 4B run revealed its template always opens a thinking block; responses consumed the 32-token budget on reasoning despite `think=false`. The original run remains intact. A local alias, `nsn-benchmark-qwen3-4b-nothink:latest`, was created from the same installed weights using the installed Qwen3 8B template that closes thinking when disabled. The original model was not replaced. The new digest, template hashes and unchanged model metadata are recorded in `qwen4b-template-repair.json`.

GGUF metadata identifies the installed weights as Qwen3-4B-Thinking-2507. The template change still emits reasoning and hits the 32-token cap, so it did not fix the answering-mode contract. Both 4B configurations remain unqualified for a no-thinking model-size comparison. This adds 105 separately identified rows with identical questions, budgets and baselines; it is a failed serving-configuration experiment, not a replacement of poor scores. The original 420-row coverage and native limitation remain explicit. Fixed profile ordering and shared caches mean latency samples are descriptive, not randomized performance qualification.

## Reproduction

Use the existing isolated `nsn_onnx_spike` evaluation Python and separately prepared ONNX assets. Each answering invocation is `python -m benchmarks.public_eval.runner` with `--case-ids sample.json --profiles stateless rolling full_context bm25 hybrid nsn_lexical nsn_semantic --longmemeval-variant s --encoder-backend onnx`, the pinned model/provider, explicit local assets and the dedicated native corpus cache. Model runs are sequential on this host. All defaults and older snapshots are preserved.

Grade complete runs using `python -m benchmarks.public_eval.grading RUN --judge-provider cloud --judge-model gpt-oss:120b --judge-digest d98fe6ba01e6 --judge-output-tokens 512 --output NEW_GRADED_RUN`. Authentication remains outside the repository.

Regenerate comparison JSON with `python -m benchmarks.public_eval.compare RUN_GRADED... --output comparison.json`. Verify all manifests, row identities and hashes before interpreting results. Fresh comparison configurations need new run directories; never merge older experiment rows to fill coverage.
