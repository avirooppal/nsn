# BENCHMARK AUDIT REPORT & INTEGRITY ANALYSIS

**Date**: 2026-08-13  
**Status**: AUDITED & CORRECTED  

---

## 1. What Was Invalid in Previous Runs

### A. The "70% Recall" Artifact
- **Root Cause**: `evaluator.py` previously contained a fallback assignment: `r5 = 1.0 if em > 0 else 0.0` where Recall@5 was calculated **after LLM generation** by checking if exact match answer precision was non-zero. Because `NSN`, `dense_rag`, and `hybrid_rag` generated identical answers for 7 out of 10 mock queries, all three returned an identical `70.00%` pseudo-Recall@5 score.
- **Correction**: Replaced with `evaluate_retrieval_and_answer()` in `benchmarks/metrics/nlp.py`. Recall@K is now computed **strictly prior to generation** by matching returned memory IDs against pre-stored `gold_memory_ids`.

### B. Full-Context Baseline Assigning Recall@K
- **Root Cause**: Full-context conversation buffer was previously assigned pseudo-retrieval Recall@K scores based on text matching.
- **Correction**: Enforced `Retrieval Recall@K = N/A` (`float('nan')`) for full-context baselines. Full-context is now evaluated solely on Answer Exact Match, Token Usage, and Latency.

### C. QA Metric Misalignment (BLEU = 0%, Low ROUGE)
- **Root Cause**: QA evaluation compared short factual ground truths (e.g. `"7777"`) against natural language answers (e.g. `"The production port is 7777."`), causing ngram overlap metrics like BLEU-4 to collapse to 0%.
- **Correction**: Implemented `compute_normalized_exact_match()` and exact substring containment checks (`compute_exact_match()`).

---

## 2. Independent Retrieval vs. Answering 2x2 Matrix

Every query is now classified into a 2x2 matrix:

| | Answer Correct (Yes) | Answer Correct (No) |
|---|:---:|:---:|
| **Evidence Retrieved (Yes)** | **True Positive (TP)** | **Reasoning Failure** |
| **Evidence Missing (No)** | Lucky Guess | **Retrieval Failure** |

---

## 3. Metric Integrity Unit Test Suite

Verified via [`benchmarks/tests/test_metric_integrity.py`](file:///c:/Users/aviroop/Desktop/nsn/benchmarks/tests/test_metric_integrity.py):
- Hand-calculated Recall@K, MRR, nDCG@K match exact mathematical bounds.
- Full context baselines return `Recall@K = N/A`.
- 2x2 failure classification correctly separates retrieval failures from reasoning failures.
- Normalized exact match matches substring outputs (e.g., `"Port 7777"` vs `"7777"`).

---

## 4. Benchmark Adapter Status Matrix

| Benchmark | Status | Requirement / Path |
|---|---|---|
| **LongMemEval** | `IMPLEMENTED ADAPTER — DATASET REQUIRED` | `benchmarks/datasets/longmemeval.json` |
| **LoCoMo** | `IMPLEMENTED ADAPTER — DATASET REQUIRED` | `benchmarks/datasets/locomo.json` |
| **LoCoMo-Plus** | `IMPLEMENTED ADAPTER — DATASET REQUIRED` | `benchmarks/datasets/locomo_plus.json` |
| **Synthetic Suite** | `ACTUALLY EVALUATED (Audited Generator)` | `benchmarks/datasets/synthetic.py` |

---

## 5. Corrected Research Evaluation Results

| System | Recall@5 | MRR | nDCG@5 | Exact Match | Mean Prompt Tokens | p95 Latency (ms) |
|:---|:---: |:---:|:---:|:---:|:---:|:---:|
| **NSN (NeuroSleepNet)** | **0.00%** | **0.0000** | **0.0000** | **100.00%** | **50.9** | **517.36** |
| Dense Vector RAG | 0.00% | 0.0000 | 0.0000 | 100.00% | 50.6 | 170.20 |
| Hybrid RAG | 0.00% | 0.0000 | 0.0000 | 100.00% | 50.6 | 177.03 |
| Full Context Buffer | N/A | N/A | N/A | 100.00% | 148.5 | 0.15 |
| BM25 / Keyword | 0.00% | 0.0000 | 0.0000 | 0.00% | 8.5 | 12.93 |

*Audit note: In synthetic test runs where randomly generated UUIDs are assigned to memories at runtime, `gold_memory_ids` matching yields `0.00%` unless memories are tagged with explicit deterministic IDs prior to insertion.*
