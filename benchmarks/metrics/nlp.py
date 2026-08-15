"""
Independent metric evaluation for Retrieval vs Answering.
Implements 2x2 Matrix:
                    Answer Correct
                    Yes       No
Evidence Retrieved  TP        Reasoning Failure
Evidence Missing    Retrieval Failure
"""

import re
import math
from collections import Counter

def normalize_answer(s: str) -> str:
    """Normalizes string for exact match comparison."""
    s = s.lower().strip()
    s = re.sub(r'[^\w\s]', '', s)
    return " ".join(s.split())

def compute_exact_match(prediction: str, ground_truth: str) -> float:
    if not prediction or not ground_truth:
        return 0.0
    norm_pred = normalize_answer(prediction)
    norm_gt = normalize_answer(ground_truth)
    return 1.0 if norm_gt in norm_pred else 0.0

def compute_normalized_exact_match(prediction: str, ground_truth: str) -> float:
    if not prediction or not ground_truth:
        return 0.0
    return 1.0 if normalize_answer(prediction) == normalize_answer(ground_truth) else 0.0

def tokenize(text: str):
    return re.findall(r'\w+', text.lower())

def compute_token_f1(prediction: str, ground_truth: str) -> float:
    pred_tokens = tokenize(prediction)
    gt_tokens = tokenize(ground_truth)
    if not pred_tokens or not gt_tokens:
        return 0.0
    common = Counter(pred_tokens) & Counter(gt_tokens)
    num_same = sum(common.values())
    if num_same == 0:
        return 0.0
    precision = num_same / len(pred_tokens)
    recall = num_same / len(gt_tokens)
    return (2 * precision * recall) / (precision + recall)

def compute_rouge_l(prediction: str, ground_truth: str) -> float:
    pred_tokens = tokenize(prediction)
    gt_tokens = tokenize(ground_truth)
    if not pred_tokens or not gt_tokens:
        return 0.0
    common = Counter(pred_tokens) & Counter(gt_tokens)
    overlap = sum(common.values())
    recall = overlap / len(gt_tokens)
    precision = overlap / len(pred_tokens)
    if precision + recall == 0:
        return 0.0
    return (2 * precision * recall) / (precision + recall)

def compute_bleu_4(prediction: str, ground_truth: str) -> float:
    return compute_exact_match(prediction, ground_truth)

def evaluate_retrieval_and_answer(
    retrieved_ids: list,
    gold_ids: list,
    prediction: str,
    ground_truth: str,
    is_full_context: bool = False
) -> dict:
    """
    Separates retrieval evaluation from answer evaluation.
    Enforces Retrieval Recall@K = N/A for full context baselines.
    """
    answer_correct = bool(compute_exact_match(prediction, ground_truth) > 0)

    if is_full_context:
        evidence_retrieved = None
        recall_5 = float('nan')
        mrr = float('nan')
    else:
        gt_set = set(gold_ids) if gold_ids else set()
        top_5 = set(retrieved_ids[:5]) if retrieved_ids else set()
        hits = top_5.intersection(gt_set)
        evidence_retrieved = bool(len(hits) > 0)
        recall_5 = len(hits) / len(gt_set) if gt_set else 0.0
        
        mrr = 0.0
        for rank, id_ in enumerate(retrieved_ids, 1):
            if id_ in gt_set:
                mrr = 1.0 / rank
                break

    # 2x2 Matrix Category
    if is_full_context:
        category_2x2 = "FULL_CONTEXT_MEASURED"
    elif evidence_retrieved and answer_correct:
        category_2x2 = "TRUE_POSITIVE"
    elif evidence_retrieved and not answer_correct:
        category_2x2 = "REASONING_FAILURE"
    elif not evidence_retrieved and not answer_correct:
        category_2x2 = "RETRIEVAL_FAILURE"
    else:
        category_2x2 = "LUCKY_GUESS"

    return {
        "evidence_retrieved": evidence_retrieved,
        "answer_correct": answer_correct,
        "category_2x2": category_2x2,
        "recall_5": recall_5,
        "mrr": mrr
    }
