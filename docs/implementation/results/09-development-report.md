# Phase 09 development evidence

Status: PARTIAL. This checks execution and reproducibility; it does not establish NSN superiority or frontier intelligence.

Three frozen development questions were selected before observing answers: `locomo:conv-42:0000`, `longmemeval:031748ae_abs`, and `longmemeval:07741c44`. Every model ran the same nine profiles, producing 27 raw rows. Each full-context profile exceeded the declared 2,048-token conservative input allowance on all three cases; those failures remain in the denominator. No model-generation fallback was used.

| Installed answering model | Reported parameters | Quantization | Digest | Answer run |
| --- | --- | --- | --- | --- |
| qwen3:1.7b | 2.0B | Q4_K_M | `8f68893c685c3ddff2aa3fffce2aa60a30bb2da65ca488b61fff134a4d1730e7` | [raw run](09-qwen3-1p7b-dev/manifest.json) |
| qwen3:4b | 4.0B | Q4_K_M | `359d7dd4bcdab3d86b87d73ac27966f4dbb9f5efdfcc75d34a8764a09474fae7` | [raw run](09-qwen3-4b-dev/manifest.json) |
| qwen3:8b | 8.2B | Q4_K_M | `500a1f067a9f782620b40bee6f7b0c89e17ae61f686b92c24933e4ca4b2b8b41` | [raw rerun](09-qwen3-8b-rerun/manifest.json) |

The two smaller tags were explicitly downloaded through the local Ollama service. The 4B download first timed out and then resumed successfully; preparation failures are retained. Models ran locally on CPU. Prepared MiniLM embeddings were used offline.

## What the results support

The one-question LoCoMo sample gives zero F1 for both NSN profiles at every size. At 4B the stateless answer has F1 0.0667 and oracle 0.0714; this single-item difference has no generalization value. The original roughly 2B/8B LoCoMo outputs also score zero across all profiles. Raw prompts, context, answers, actual provider token counts and stage costs are retained. No aggregate across different datasets or self-judges is reported.

The initial LongMemEval self-judge calls mistakenly inherited the answering system instruction. Their scores are unusable, including superficially valid yes/no responses. Original rows/summaries remain unchanged for audit; [campaign JSON](09-development-campaign.json) excludes those scores. A regression now verifies separate grading instructions and per-row judge provenance. The [corrected 4B rerun](09-qwen3-4b-judge-repaired/manifest.json) still returns nonbinary, length-limited grading text; its LongMemEval accuracy remains unknown with full-denominator bounds. This model/limit combination does not provide a qualified judge.

Independent summary regeneration exactly reproduces the raw run summaries, and source snapshots match their manifest hashes. Reproducibility cannot validate an unsuitable judge. Full-context overflow and judge failure remain visible; retrieval coverage is separate from answer scoring.

## Competitors and remaining gates

[Actual pinned Mem0 qualification](09-mem0-qualified.json) successfully stores and retrieves one synthetic port observation using real local extraction. It incurred one extraction call (1,026 input and 36 output tokens), about 75.5 seconds of generation, about 78.4 seconds ingestion and 1.63 seconds retrieval on this host. This is a setup check, not a matched public QA comparison. Earlier setup errors are preserved. [Readiness registry](09-competitor-readiness.json) marks Mem0, Graphiti, Letta and A-MEM public QA unevaluated, with no invented scores.

Only the cleaned LongMemEval oracle sessions are prepared; full S/M distractor haystacks remain missing. The frozen 2,122-question holdout is unrun. Independent grading/human validation, actual competitor QA, recommended-configuration tracks, feature ablations, validated stale/conflict/task annotations and isolated peak/cold/warm resource measurement remain required. Sampled client RSS excludes the Ollama process. Three development clusters and conservative 512-byte whole-turn memory limits cannot establish the desired quality/efficiency target.

See [handoff](../handoffs/09.md) for exact commands and continuation gates. Phase 10 has not started.
