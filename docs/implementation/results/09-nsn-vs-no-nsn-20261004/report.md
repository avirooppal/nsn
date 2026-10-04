# NSN versus non-NSN: measured paired benchmark

Completed bounded comparison: 15 frozen public questions × 7 configurations × 4 real answering models = **420 original rows**, plus **105 separately pinned template-change attempt 4B rows**. Nine LoCoMo questions cover all five categories; six full-S LongMemEval questions cover its six categories. This is an exploratory sample, not the complete public campaign or a production/superiority qualification.

Same 2,048-token input window, 32 output tokens, 512 bytes of memory evidence, seed 42 and decoding per model/configuration. Cloud 31B is a separate reference. LongMemEval uses an independent Cloud GPT-OSS 120B judge; human calibration remains pending. See [protocol](protocol.md) and [frozen sample](sample.json).

Native Qwen3 4B has an unconditional opening thinking block in its installed template. Despite requesting `think=false`, it spent the answer budget on reasoning; its native scores are retained but not qualified as a no-thinking model-size comparison. The additional `nsn-benchmark-qwen3-4b-nothink:latest` run uses the same weights with the installed 8B Qwen3 no-thinking template, new digest and unchanged original model. GGUF metadata identifies these as Qwen3-4B-Thinking-2507 weights. The template attempt still emits reasoning and exhausts the budget, so both 4B configurations are unqualified for a no-thinking comparison. See [failed template-attempt provenance](qwen4b-template-repair.json). No samples or budgets changed.

## LoCoMo upstream F1 (%)

| Model | No memory | Recent context | Full context | BM25 | Hybrid RAG | NSN lexical | NSN semantic |
| --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| qwen3:1.7b | 22.2 | 22.2 | 0.0 | 25.0 | 27.8 | 35.6 | 26.7 |
| qwen3:4b | 1.2 | 1.1 | 0.0 | 1.2 | 1.2 | 1.2 | 1.2 |
| qwen3:8b | 22.2 | 22.2 | 0.0 | 27.8 | 22.2 | 34.7 | 17.5 |
| gemma4:31b | 22.2 | 24.7 | 0.0 | 33.3 | 33.3 | 33.3 | 33.3 |
| nsn-benchmark-qwen3-4b-nothink:latest | 0.6 | 0.6 | 0.0 | 1.2 | 1.2 | 1.2 | 1.2 |

## LongMemEval independent-judge correctness (%) — provisional

| Model | No memory | Recent context | Full context | BM25 | Hybrid RAG | NSN lexical | NSN semantic |
| --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| qwen3:1.7b | 0.0 | 0.0 | 0.0 | 33.3 | 16.7 | 0.0 | 16.7 |
| qwen3:4b | 0.0 | 0.0 | 0.0 | 16.7 | 0.0 | 0.0 | 0.0 |
| qwen3:8b | 0.0 | 0.0 | 0.0 | 33.3 | 16.7 | 0.0 | 16.7 |
| gemma4:31b | 0.0 | 0.0 | 0.0 | 33.3 | 16.7 | 0.0 | 16.7 |
| nsn-benchmark-qwen3-4b-nothink:latest | 0.0 | 0.0 | 0.0 | 0.0 | 0.0 | 0.0 | 0.0 |

## Measured latency, evidence and tokens

Retrieval includes first-use vector materialization and shared-cache effects. Retrieval plus generation excludes offline native-corpus preparation. Percentiles use the nearest observed order statistic; only 15 observations per configuration. Full-context failure paths are not useful inference throughput.

| Model | Configuration | Evidence recall | Evidence precision | Complete evidence | Retrieval p50 / p95 ms | Generation p50 / p95 ms | Retrieval + generation p95 ms | Prompt / output tokens | Failures |
| --- | --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| qwen3:1.7b | stateless | 0.000 | 0.000 | 0.000 | 0.0 / 0.0 | 653.7 / 1688.0 | 1688.0 | 1326 / 76 | 0 |
| qwen3:1.7b | rolling | 0.000 | 0.000 | 0.000 | 0.9 / 1.2 | 2805.0 / 3276.0 | 3276.5 | 4186 / 60 | 0 |
| qwen3:1.7b | full_context | 0.000 | 0.000 | 0.000 | 0.9 / 2.0 | 0.0 / 0.0 | 2.0 | 0 / 0 | 15 |
| qwen3:1.7b | bm25 | 0.267 | 0.200 | 0.200 | 32.7 / 71.4 | 2434.7 / 2967.7 | 3039.2 | 3662 / 92 | 0 |
| qwen3:1.7b | hybrid | 0.200 | 0.133 | 0.133 | 28.8 / 32.9 | 2249.6 / 3166.1 | 3195.0 | 3712 / 100 | 0 |
| qwen3:1.7b | nsn_lexical | 0.100 | 0.100 | 0.067 | 5.3 / 8.0 | 2212.0 / 2608.4 | 2614.5 | 3301 / 76 | 0 |
| qwen3:1.7b | nsn_semantic | 0.200 | 0.200 | 0.133 | 137.3 / 166.4 | 642.2 / 2515.5 | 2649.7 | 3251 / 93 | 0 |
| qwen3:4b | stateless | 0.000 | 0.000 | 0.000 | 0.0 / 0.0 | 3903.2 / 5490.8 | 5490.8 | 1236 / 480 | 0 |
| qwen3:4b | rolling | 0.000 | 0.000 | 0.000 | 1.1 / 1.5 | 8462.8 / 12806.7 | 12808.6 | 3804 / 448 | 1 |
| qwen3:4b | full_context | 0.000 | 0.000 | 0.000 | 0.9 / 1.9 | 0.0 / 0.0 | 1.9 | 0 / 0 | 15 |
| qwen3:4b | bm25 | 0.267 | 0.200 | 0.200 | 41.6 / 76.9 | 9415.1 / 10949.7 | 11006.0 | 3572 / 480 | 0 |
| qwen3:4b | hybrid | 0.200 | 0.133 | 0.133 | 30.0 / 45.2 | 8922.7 / 10975.5 | 11004.6 | 3622 / 480 | 0 |
| qwen3:4b | nsn_lexical | 0.100 | 0.100 | 0.067 | 6.2 / 10.1 | 8308.1 / 9604.6 | 9610.0 | 3211 / 480 | 0 |
| qwen3:4b | nsn_semantic | 0.200 | 0.200 | 0.133 | 152.6 / 172.1 | 3480.1 / 9964.9 | 10123.3 | 3161 / 480 | 0 |
| qwen3:8b | stateless | 0.000 | 0.000 | 0.000 | 0.0 / 0.0 | 3114.6 / 5383.3 | 5383.3 | 1326 / 60 | 0 |
| qwen3:8b | rolling | 0.000 | 0.000 | 0.000 | 1.2 / 2.1 | 15346.1 / 17420.5 | 17422.0 | 4186 / 60 | 0 |
| qwen3:8b | full_context | 0.000 | 0.000 | 0.000 | 1.0 / 2.8 | 0.0 / 0.0 | 2.8 | 0 / 0 | 15 |
| qwen3:8b | bm25 | 0.267 | 0.200 | 0.200 | 40.0 / 54.5 | 13338.8 / 16721.1 | 16763.3 | 3662 / 103 | 0 |
| qwen3:8b | hybrid | 0.200 | 0.133 | 0.133 | 31.9 / 97.9 | 12358.9 / 15241.8 | 15288.0 | 3712 / 106 | 0 |
| qwen3:8b | nsn_lexical | 0.100 | 0.100 | 0.067 | 6.6 / 13.1 | 11355.8 / 13788.9 | 13794.8 | 3301 / 83 | 0 |
| qwen3:8b | nsn_semantic | 0.200 | 0.200 | 0.133 | 155.9 / 224.2 | 2371.3 / 13145.2 | 13288.5 | 3251 / 86 | 0 |
| gemma4:31b | stateless | 0.000 | 0.000 | 0.000 | 0.0 / 0.0 | 807.9 / 1634.0 | 1634.0 | 1313 / 60 | 0 |
| gemma4:31b | rolling | 0.000 | 0.000 | 0.000 | 2.1 / 3.1 | 1140.3 / 1746.9 | 1749.1 | 4279 / 73 | 0 |
| gemma4:31b | full_context | 0.000 | 0.000 | 0.000 | 1.7 / 3.3 | 0.0 / 0.0 | 3.3 | 0 / 0 | 15 |
| gemma4:31b | bm25 | 0.267 | 0.200 | 0.200 | 66.6 / 80.4 | 1644.2 / 1775.7 | 1851.5 | 3733 / 73 | 0 |
| gemma4:31b | hybrid | 0.200 | 0.133 | 0.133 | 53.1 / 128.3 | 1592.5 / 2046.7 | 2631.1 | 3790 / 71 | 0 |
| gemma4:31b | nsn_lexical | 0.100 | 0.100 | 0.067 | 12.4 / 17.5 | 902.1 / 2170.3 | 2182.1 | 3349 / 59 | 0 |
| gemma4:31b | nsn_semantic | 0.200 | 0.200 | 0.133 | 278.2 / 321.8 | 883.9 / 1672.1 | 1990.4 | 3313 / 70 | 0 |
| nsn-benchmark-qwen3-4b-nothink:latest | stateless | 0.000 | 0.000 | 0.000 | 0.0 / 0.0 | 4641.3 / 6629.2 | 6629.2 | 1326 / 480 | 0 |
| nsn-benchmark-qwen3-4b-nothink:latest | rolling | 0.000 | 0.000 | 0.000 | 1.4 / 1.6 | 9827.1 / 13309.5 | 13311.0 | 4186 / 480 | 0 |
| nsn-benchmark-qwen3-4b-nothink:latest | full_context | 0.000 | 0.000 | 0.000 | 1.2 / 2.3 | 0.0 / 0.0 | 2.3 | 0 / 0 | 15 |
| nsn-benchmark-qwen3-4b-nothink:latest | bm25 | 0.267 | 0.200 | 0.200 | 43.9 / 56.6 | 10026.1 / 12789.3 | 12830.0 | 3662 / 480 | 0 |
| nsn-benchmark-qwen3-4b-nothink:latest | hybrid | 0.200 | 0.133 | 0.133 | 37.3 / 86.5 | 9726.4 / 12525.7 | 12558.2 | 3712 / 480 | 0 |
| nsn-benchmark-qwen3-4b-nothink:latest | nsn_lexical | 0.100 | 0.100 | 0.067 | 9.2 / 11.5 | 9259.2 / 10517.5 | 10528.7 | 3301 / 480 | 0 |
| nsn-benchmark-qwen3-4b-nothink:latest | nsn_semantic | 0.200 | 0.200 | 0.133 | 203.6 / 238.7 | 4441.8 / 11462.6 | 11676.3 | 3251 / 480 | 0 |

## Resource costs and interpretation

Per-model client RSS/disk, token-count failures and management-call counts are in [comparison.json](comparison.json). The shared harness stores the native corpus even for the no-memory profile; those RSS/disk samples cannot measure incremental NSN overhead. Every configuration here has extraction/consolidation disabled. Original ingestion/embedding/cache-build costs remain in each raw row and regenerated run summary.

Separate previously measured installed-wheel 10k profiles (same SDK candidate, different workload): minimal append p95 **2.71ms**, repeated retrieval **2.13ms**, fresh-query retrieval **85.88ms**; optional ONNX/deferred append **5.52ms**, repeated retrieval **48.35ms**, fresh-query retrieval **143.88ms**, incremental memory **231.12MiB**. Fresh-query targets still fail; deferred indexing does not qualify default inline append. See [minimal](../10-completion-minimal-profile-20261004.json) and [semantic](../10-completion-onnx-profile-20261004.json). These are not new baseline-versus-NSN process-isolation measurements.

## Uncertainty and unavailable metrics

All paired differences and cluster intervals are in comparison.json. LoCoMo has only two sampled conversation clusters, so its intervals are exploratory and weak for generalization. LongMemEval has one connected cluster and no valid conversation-cluster confidence interval. Fixed profile order, shared caches and model loading affect wall latency; it is not a randomized performance qualification. No comparison establishes general superiority.

Abstention uses literal canonical phrases, not a human-validated semantic classifier. Additional category scores, abstention, MRR and NDCG over the selected evidence appear in [additional-metrics.json](additional-metrics.json). Native raw answers, full prompts, provider token counts/timings and grading responses are retained in the model directories. Human review queues are unfilled.

Validated hallucination/stale/conflict rates and agent task success are unavailable in this sample. Remote memory, billed dollar cost and time-to-first-token are unavailable. Failures and missing scores/counts remain visible rather than being silently omitted. Core retrieval, not full SDK-wrapper throughput or optional graph/consolidation/procedure behavior, is measured.
