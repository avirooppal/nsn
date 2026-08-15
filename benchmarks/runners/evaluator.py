"""
BenchmarkEvaluator — Research-grade with correct gold-evidence ID mapping.

Critical Fix
------------
system.observe() returns the actual stored memory_id (a UUID generated at
runtime by MemoryRecord.__post_init__). The BenchmarkItem.gold_memory_ids
contain synthetic string IDs such as 'mem_upd_0_day1'. The previous
implementation discarded the returned IDs, compared synthetic strings against
retrieved UUIDs, and therefore always measured Recall@5 = 0%.

The fix is _ingest_with_id_map(): capture the real UUID for each observation
after ingestion and build a translation table  synthetic_id -> actual_uuid.
All retrieval metrics are then computed against translated gold IDs.

Benchmark Categories
--------------------
  knowledge_update   -- temporal state tracking (3-step update chains)
  contradiction      -- trust/source conflict resolution
  multi_hop          -- relational graph traversal

2x2 Failure Matrix
------------------
  TRUE_POSITIVE    -- evidence retrieved AND answer correct
  REASONING_FAILURE-- evidence retrieved BUT answer wrong
  RETRIEVAL_FAILURE-- evidence missing AND answer wrong
  LUCKY_GUESS      -- evidence missing BUT answer correct (hallucination)
"""

import numpy as np
from benchmarks.datasets.synthetic import LargeSyntheticDatasetGenerator
from benchmarks.metrics.nlp import (
    compute_exact_match, compute_token_f1, compute_rouge_l,
    evaluate_retrieval_and_answer,
)
from benchmarks.metrics.retrieval import (
    compute_recall_at_k,
    compute_hit_at_k,
    compute_mrr,
    compute_ndcg_at_k,
    compute_mean_rank,
)


# ---------------------------------------------------------------------------
# Core helpers
# ---------------------------------------------------------------------------

def _ingest_with_id_map(system, observations: list) -> dict:
    """
    Ingest observations and return {synthetic_id: actual_stored_uuid}.

    When a system rejects an observation as a duplicate it returns None;
    that synthetic ID is simply absent from the returned map (not an error).
    """
    id_map = {}
    for obs in observations:
        synthetic_id = obs.get("id")
        actual_id = system.observe(
            obs["content"],
            source=obs.get("source", "system"),
            metadata=obs.get("metadata", {}),
        )
        if synthetic_id and actual_id:
            id_map[synthetic_id] = actual_id
    return id_map


def _translate_gold_ids(gold_ids: list, id_map: dict) -> list:
    """Map synthetic gold IDs to actual stored UUIDs, skipping missing ones."""
    return [id_map[gid] for gid in gold_ids if gid in id_map]


# ---------------------------------------------------------------------------
# Main evaluator
# ---------------------------------------------------------------------------

class BenchmarkEvaluator:
    def __init__(self, logger=None):
        self.logger = logger

    def _evaluate_query_set(
        self,
        system,
        queries: list,
        id_map: dict,
        benchmark_name: str,
        is_full_context: bool,
    ) -> dict:
        """
        Core evaluation loop. Translates gold IDs, queries the system,
        and accumulates all retrieval + answer metrics.
        """
        exact_matches, token_f1s, rouge_ls = [], [], []
        recalls_5, hits_5, mrrs, ndcgs_5 = [], [], [], []
        mean_ranks_list, latencies_ms, prompt_tokens_list = [], [], []

        tp_count = fn_count = rf_count = lg_count = dup_skip = 0

        for q in queries:
            translated_gold = _translate_gold_ids(q.gold_memory_ids, id_map)

            # If every gold memory was rejected as a duplicate, skip this query.
            # NOTE: full_context observe() always returns None (by design), so
            # translated_gold is always [] for full_context. Only skip for
            # retrieval-based systems.
            if not is_full_context and q.gold_memory_ids and not translated_gold:
                dup_skip += 1
                continue

            res = system.query(q.question)
            pred = res.get("answer", "")
            gt = q.ground_truth_answer
            retrieved_ids = res.get("retrieved_ids", [])
            latency_ms = res.get("latency_ms", 0.0)
            prompt_tokens = res.get("prompt_tokens", 0)

            # Answer-level metrics
            em = compute_exact_match(pred, gt)
            tf1 = compute_token_f1(pred, gt)
            rl = compute_rouge_l(pred, gt)
            exact_matches.append(em)
            token_f1s.append(tf1)
            rouge_ls.append(rl)
            latencies_ms.append(latency_ms)
            prompt_tokens_list.append(prompt_tokens)

            # Retrieval metrics (N/A for full-context baselines)
            cat = "N/A"
            if not is_full_context and translated_gold:
                r5 = compute_recall_at_k(retrieved_ids, translated_gold, k=5)
                h5 = compute_hit_at_k(retrieved_ids, translated_gold, k=5)
                mrr = compute_mrr(retrieved_ids, translated_gold)
                ndcg = compute_ndcg_at_k(retrieved_ids, translated_gold, k=5)
                mr = compute_mean_rank(retrieved_ids, translated_gold)

                recalls_5.append(r5)
                hits_5.append(h5)
                mrrs.append(mrr)
                ndcgs_5.append(ndcg)
                if not (isinstance(mr, float) and mr != mr):   # skip NaN
                    mean_ranks_list.append(mr)

                ev = evaluate_retrieval_and_answer(
                    retrieved_ids=retrieved_ids,
                    gold_ids=translated_gold,
                    prediction=pred,
                    ground_truth=gt,
                    is_full_context=False,
                )
                cat = ev["category_2x2"]
                if cat == "TRUE_POSITIVE":       tp_count += 1
                elif cat == "REASONING_FAILURE": fn_count += 1
                elif cat == "RETRIEVAL_FAILURE": rf_count += 1
                elif cat == "LUCKY_GUESS":       lg_count += 1

            if self.logger:
                self.logger.log_query(
                    benchmark=benchmark_name,
                    query_id=q.query_id,
                    system=system.name,
                    question=q.question,
                    ground_truth=gt,
                    answer=pred,
                    correct=bool(em),
                    retrieved_ids=retrieved_ids,
                    latency_ms=latency_ms,
                    prompt_tokens=prompt_tokens,
                    metadata={
                        "translated_gold_ids": translated_gold,
                        "category": q.category,
                        "difficulty": q.difficulty,
                        "category_2x2": cat,
                    },
                )

        n = len(exact_matches)
        p50 = float(np.percentile(latencies_ms, 50)) if latencies_ms else 0.0
        p95 = float(np.percentile(latencies_ms, 95)) if latencies_ms else 0.0

        return {
            "samples": n,
            "exact_match": float(np.mean(exact_matches)) if exact_matches else 0.0,
            "token_f1": float(np.mean(token_f1s)) if token_f1s else 0.0,
            "rouge_l": float(np.mean(rouge_ls)) if rouge_ls else 0.0,
            "recall_5": float(np.mean(recalls_5)) if recalls_5 else float("nan"),
            "hit_5": float(np.mean(hits_5)) if hits_5 else float("nan"),
            "mrr": float(np.mean(mrrs)) if mrrs else float("nan"),
            "ndcg_5": float(np.mean(ndcgs_5)) if ndcgs_5 else float("nan"),
            "mean_rank": float(np.mean(mean_ranks_list)) if mean_ranks_list else float("nan"),
            "mean_tokens": float(np.mean(prompt_tokens_list)) if prompt_tokens_list else 0.0,
            "p50_latency": p50,
            "p95_latency": p95,
            "2x2_matrix": {
                "true_positives": tp_count,
                "reasoning_failures": fn_count,
                "retrieval_failures": rf_count,
                "lucky_guesses": lg_count,
                "duplicate_rejected_skipped": dup_skip,
            },
        }

    def evaluate_update_benchmark(self, system, num_samples: int = 100) -> dict:
        """Knowledge-update benchmark: can the system track the LATEST state?"""
        gen = LargeSyntheticDatasetGenerator(seed=42)
        system.reset()

        data = gen.generate_gold_update_dataset(num_questions=num_samples)
        is_fc = system.name == "full_context"
        id_map = {}
        all_queries = []

        for seq in data:
            id_map.update(_ingest_with_id_map(system, seq["observations"]))
            all_queries.extend(seq["queries"])

        result = self._evaluate_query_set(
            system, all_queries, id_map, "knowledge_update", is_fc
        )
        result.update({"system": system.name, "benchmark": "knowledge_update"})
        system.reset()
        return result

    def evaluate_contradiction_benchmark(self, system, num_samples: int = 100) -> dict:
        """Contradiction benchmark: does trust/source determine the winning fact?"""
        gen = LargeSyntheticDatasetGenerator(seed=42)
        system.reset()

        data = gen.generate_gold_contradiction_dataset(num_questions=num_samples)
        is_fc = system.name == "full_context"
        id_map = {}
        all_queries = []

        for item in data:
            id_map.update(_ingest_with_id_map(system, item["observations"]))
            all_queries.append(item["query"])

        result = self._evaluate_query_set(
            system, all_queries, id_map, "contradiction", is_fc
        )
        result.update({"system": system.name, "benchmark": "contradiction"})
        system.reset()
        return result

    def evaluate_multihop_benchmark(self, system, num_chains: int = 50) -> dict:
        """Multi-hop benchmark: does the system follow relational chains?"""
        gen = LargeSyntheticDatasetGenerator(seed=42)
        system.reset()

        data = gen.generate_gold_graph_multihop_dataset(num_chains=num_chains)
        is_fc = system.name == "full_context"
        id_map = {}
        all_queries = []

        for chain in data:
            id_map.update(_ingest_with_id_map(system, chain["observations"]))
            all_queries.extend(chain["queries"])

        result = self._evaluate_query_set(
            system, all_queries, id_map, "multi_hop", is_fc
        )
        result.update({"system": system.name, "benchmark": "multi_hop"})
        system.reset()
        return result

    # Legacy compatibility alias
    def evaluate_synthetic_benchmark(self, system, num_samples: int = 20) -> dict:
        return self.evaluate_update_benchmark(system, num_samples=num_samples)

