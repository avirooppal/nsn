"""
Retrieval quality metrics (Recall@K, Precision@K, Hit@K, MRR, nDCG@K, Mean Rank).
"""
import math

def compute_recall_at_k(retrieved_ids: list, ground_truth_ids: list, k: int) -> float:
    if not ground_truth_ids:
        return 0.0
    top_k = set(retrieved_ids[:k])
    gt_set = set(ground_truth_ids)
    hits = len(top_k.intersection(gt_set))
    return hits / len(gt_set)

def compute_precision_at_k(retrieved_ids: list, ground_truth_ids: list, k: int) -> float:
    """Fraction of top-K retrieved items that are relevant."""
    if not retrieved_ids or k == 0:
        return 0.0
    top_k = retrieved_ids[:k]
    gt_set = set(ground_truth_ids)
    hits = sum(1 for id_ in top_k if id_ in gt_set)
    return hits / min(k, len(top_k))

def compute_hit_at_k(retrieved_ids: list, ground_truth_ids: list, k: int) -> float:
    if not ground_truth_ids:
        return 0.0
    top_k = set(retrieved_ids[:k])
    gt_set = set(ground_truth_ids)
    return 1.0 if len(top_k.intersection(gt_set)) > 0 else 0.0

def compute_mrr(retrieved_ids: list, ground_truth_ids: list) -> float:
    if not ground_truth_ids:
        return 0.0
    gt_set = set(ground_truth_ids)
    for rank, id_ in enumerate(retrieved_ids, 1):
        if id_ in gt_set:
            return 1.0 / rank
    return 0.0

def compute_ndcg_at_k(retrieved_ids: list, ground_truth_ids: list, k: int) -> float:
    if not ground_truth_ids:
        return 0.0
    gt_set = set(ground_truth_ids)
    dcg = 0.0
    for rank, id_ in enumerate(retrieved_ids[:k], 1):
        if id_ in gt_set:
            dcg += 1.0 / math.log2(rank + 1)
            
    idcg = sum(1.0 / math.log2(r + 1) for r in range(1, min(len(gt_set), k) + 1))
    return dcg / idcg if idcg > 0 else 0.0

def compute_mean_rank(retrieved_ids: list, ground_truth_ids: list) -> float:
    if not ground_truth_ids:
        return float('nan')
    gt_set = set(ground_truth_ids)
    ranks = []
    for rank, id_ in enumerate(retrieved_ids, 1):
        if id_ in gt_set:
            ranks.append(rank)
    return sum(ranks) / len(ranks) if ranks else float('nan')

