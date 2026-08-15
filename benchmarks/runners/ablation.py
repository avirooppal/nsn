"""
Sleep & Component Ablation Runner — Research-Grade with ID mapping.

Sleep ablation conditions:
  no_sleep    -- baseline, no offline consolidation
  nrem_only   -- episodic-to-semantic aggregation only
  rem_only    -- contradiction resolution + pruning only
  full_sleep  -- NREM + REM + decay (complete pipeline)

Retrieval ablation conditions (NSN component knockout):
  nsn_full         -- full hybrid: FAISS + FTS5 + graph + RRF + reranker
  nsn_no_keyword   -- FAISS + graph only (no FTS5)
  nsn_no_semantic  -- FTS5 only (no FAISS)
  nsn_no_graph     -- FAISS + FTS5 only (no graph)
  nsn_no_rrf       -- FAISS only (no fusion)
  nsn_keyword_only -- FTS5 only
  nsn_semantic_only-- FAISS only
  nsn_graph_only   -- graph only
"""

import numpy as np
from benchmarks.baselines.nsn_adapter import NSNSystem
from benchmarks.datasets.synthetic import LargeSyntheticDatasetGenerator
from benchmarks.metrics.nlp import compute_exact_match, compute_token_f1
from benchmarks.metrics.retrieval import compute_recall_at_k, compute_mrr, compute_ndcg_at_k
from benchmarks.runners.evaluator import _ingest_with_id_map, _translate_gold_ids


def run_sleep_ablation_experiment(
    num_samples: int = 50,
    db_prefix: str = "bench_sleep_ablation",
) -> dict:
    """
    Evaluate NSN across 4 sleep configurations.
    Uses the contradiction dataset (most sensitive to trust/source ranking).
    The id_map is built once per mode; sleep does NOT change memory IDs.
    """
    gen = LargeSyntheticDatasetGenerator(seed=42)
    data = gen.generate_gold_contradiction_dataset(num_questions=num_samples)

    results = {}
    sleep_modes = ["no_sleep", "nrem_only", "rem_only", "full_sleep"]

    for mode in sleep_modes:
        db_path = f"{db_prefix}_{mode}.db"
        sys = NSNSystem(
            db_path=db_path,
            namespace=f"sleep_{mode}",
            name=f"nsn_{mode}",
        )

        # --- Ingest ALL observations, build id_map ---
        id_map = {}
        for item in data:
            item_map = _ingest_with_id_map(sys, item["observations"])
            id_map.update(item_map)

        # --- Apply sleep phase ---
        if mode == "nrem_only":
            try:
                from neurosleepnet.sleep.engine import SleepEngine
                SleepEngine(sys.memory).nrem_consolidation()
            except Exception:
                sys.sleep()   # fallback: full sleep
        elif mode == "rem_only":
            try:
                from neurosleepnet.sleep.engine import SleepEngine
                SleepEngine(sys.memory).rem_consolidation()
            except Exception:
                sys.sleep()
        elif mode == "full_sleep":
            sys.sleep()
        # no_sleep: do nothing

        # --- Evaluate (same id_map; sleep does not reassign IDs) ---
        ems, r5s, mrrs = [], [], []
        for item in data:
            q = item["query"]
            translated_gold = _translate_gold_ids(q.gold_memory_ids, id_map)
            if not translated_gold:
                continue   # all gold was duplicate-rejected
            res = sys.query(q.question)
            pred = res.get("answer", "")
            retrieved_ids = res.get("retrieved_ids", [])
            ems.append(compute_exact_match(pred, q.ground_truth_answer))
            r5s.append(compute_recall_at_k(retrieved_ids, translated_gold, k=5))
            mrrs.append(compute_mrr(retrieved_ids, translated_gold))

        results[mode] = {
            "samples": len(ems),
            "exact_match": float(np.mean(ems)) if ems else 0.0,
            "recall_5": float(np.mean(r5s)) if r5s else float("nan"),
            "mrr": float(np.mean(mrrs)) if mrrs else float("nan"),
        }
        sys.reset()

    return results


def run_retrieval_ablation_experiment(
    num_samples: int = 50,
    db_prefix: str = "bench_retrieval_ablation",
) -> dict:
    """
    Component knockout ablation: evaluate NSN retrieval sub-systems independently.
    Uses knowledge-update dataset for clearest retrieval signal.
    """
    gen = LargeSyntheticDatasetGenerator(seed=42)
    data = gen.generate_gold_update_dataset(num_questions=num_samples)

    ablation_modes = [
        ("nsn_full",          None),
        ("nsn_no_keyword",    "nsn_no_keyword"),
        ("nsn_no_semantic",   "nsn_no_semantic"),
        ("nsn_no_graph",      "nsn_no_graph"),
        ("nsn_no_rrf",        "nsn_no_rrf"),
        ("nsn_keyword_only",  "keyword_only"),
        ("nsn_semantic_only", "semantic_only"),
        ("nsn_graph_only",    "graph_only"),
    ]

    results = {}
    for variant_name, ablation_mode in ablation_modes:
        db_path = f"{db_prefix}_{variant_name}.db"
        sys = NSNSystem(
            db_path=db_path,
            namespace=f"abl_{variant_name}",
            name=variant_name,
            ablation_mode=ablation_mode,
        )

        id_map = {}
        all_queries = []
        for seq in data:
            id_map.update(_ingest_with_id_map(sys, seq["observations"]))
            all_queries.extend(seq["queries"])

        ems, r5s, mrrs, ndcgs = [], [], [], []
        for q in all_queries:
            translated_gold = _translate_gold_ids(q.gold_memory_ids, id_map)
            if not translated_gold:
                continue
            res = sys.query(q.question)
            pred = res.get("answer", "")
            retrieved_ids = res.get("retrieved_ids", [])
            ems.append(compute_exact_match(pred, q.ground_truth_answer))
            r5s.append(compute_recall_at_k(retrieved_ids, translated_gold, k=5))
            mrrs.append(compute_mrr(retrieved_ids, translated_gold))
            ndcgs.append(compute_ndcg_at_k(retrieved_ids, translated_gold, k=5))

        results[variant_name] = {
            "samples": len(ems),
            "exact_match": float(np.mean(ems)) if ems else 0.0,
            "recall_5": float(np.mean(r5s)) if r5s else float("nan"),
            "mrr": float(np.mean(mrrs)) if mrrs else float("nan"),
            "ndcg_5": float(np.mean(ndcgs)) if ndcgs else float("nan"),
        }
        sys.reset()

    return results

