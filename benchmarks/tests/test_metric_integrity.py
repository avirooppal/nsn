"""
Metric integrity tests — prevents regressions in all retrieval and NLP metrics.
All arithmetic is independently verified by hand in docstrings.
"""

import sys, os, math
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "../..")))

import pytest
from benchmarks.metrics.retrieval import (
    compute_recall_at_k, compute_precision_at_k, compute_hit_at_k,
    compute_mrr, compute_ndcg_at_k, compute_mean_rank,
)
from benchmarks.metrics.nlp import (
    compute_exact_match, compute_token_f1, compute_rouge_l, normalize_answer,
)


# ============================================================
# Retrieval Metrics
# ============================================================

class TestRecallAtK:
    def test_perfect_recall(self):
        assert compute_recall_at_k(["a","b","c"], ["a","b"], k=5) == 1.0

    def test_zero_recall(self):
        assert compute_recall_at_k(["x","y","z"], ["a","b"], k=5) == 0.0

    def test_partial_recall_one_of_two(self):
        # 1 hit out of 2 gold = 0.5
        assert compute_recall_at_k(["a","x","y","z","w"], ["a","b"], k=5) == 0.5

    def test_empty_gold(self):
        assert compute_recall_at_k(["a","b"], [], k=5) == 0.0

    def test_gold_at_exactly_k(self):
        # "a" is at rank 5 (index 4 = 5th element)
        assert compute_recall_at_k(["x","y","z","w","a"], ["a"], k=5) == 1.0

    def test_gold_beyond_k(self):
        # "a" is at rank 6 — beyond k=5 cutoff
        assert compute_recall_at_k(["x","y","z","w","v","a"], ["a"], k=5) == 0.0

    def test_denominator_is_gold_not_k(self):
        # 5 gold items, 3 retrieved, recall = 3/5
        assert abs(compute_recall_at_k(["a","b","c","x","y"], ["a","b","c","d","e"], k=5) - 3/5) < 1e-9


class TestPrecisionAtK:
    def test_perfect_precision(self):
        assert compute_precision_at_k(["a","b"], ["a","b"], k=2) == 1.0

    def test_zero_precision(self):
        assert compute_precision_at_k(["x","y"], ["a","b"], k=2) == 0.0

    def test_one_of_two_at_k2(self):
        # 1 hit in top-2 = 0.5
        assert compute_precision_at_k(["a","x"], ["a","b"], k=2) == 0.5

    def test_empty_retrieved(self):
        assert compute_precision_at_k([], ["a"], k=5) == 0.0

    def test_k_zero(self):
        assert compute_precision_at_k(["a","b"], ["a"], k=0) == 0.0


class TestHitAtK:
    def test_hit(self):
        assert compute_hit_at_k(["a","b","c"], ["a"], k=5) == 1.0

    def test_miss(self):
        assert compute_hit_at_k(["x","y","z"], ["a"], k=5) == 0.0

    def test_empty_gold(self):
        assert compute_hit_at_k(["a"], [], k=5) == 0.0

    def test_binary_any_hit_counts(self):
        # Only need 1 of many gold items
        assert compute_hit_at_k(["a","x","y"], ["b","a","c"], k=3) == 1.0


class TestMRR:
    def test_rank1(self):
        assert compute_mrr(["a","b","c"], ["a"]) == 1.0

    def test_rank2(self):
        # 1/2 = 0.5
        assert abs(compute_mrr(["x","a","c"], ["a"]) - 0.5) < 1e-9

    def test_rank3(self):
        # 1/3
        assert abs(compute_mrr(["x","y","a"], ["a"]) - 1/3) < 1e-9

    def test_not_retrieved(self):
        assert compute_mrr(["x","y","z"], ["a"]) == 0.0

    def test_empty_gold(self):
        assert compute_mrr(["a","b"], []) == 0.0

    def test_first_gold_determines_rank(self):
        # gold=[a,b], a is at rank 2, b at rank 3 -> MRR uses first hit -> 1/2
        assert abs(compute_mrr(["x","a","b"], ["a","b"]) - 0.5) < 1e-9


class TestNDCG:
    def test_perfect_rank1(self):
        # DCG = 1/log2(2) = 1, iDCG = 1/log2(2) = 1 -> nDCG = 1.0
        assert compute_ndcg_at_k(["a","b","c"], ["a"], k=5) == 1.0

    def test_rank2(self):
        # DCG = 1/log2(3), iDCG = 1/log2(2) = 1 -> nDCG = log2(2)/log2(3)
        expected = (1/math.log2(3)) / (1/math.log2(2))
        assert abs(compute_ndcg_at_k(["x","a","c"], ["a"], k=5) - expected) < 1e-9

    def test_zero_ndcg(self):
        assert compute_ndcg_at_k(["x","y","z"], ["a"], k=5) == 0.0

    def test_empty_gold(self):
        assert compute_ndcg_at_k(["a","b"], [], k=5) == 0.0

    def test_two_gold_at_rank1_and_2(self):
        # DCG = 1/log2(2) + 1/log2(3) = 1 + 0.631
        # iDCG = 1/log2(2) + 1/log2(3) = same -> nDCG = 1.0
        assert abs(compute_ndcg_at_k(["a","b","c"], ["a","b"], k=5) - 1.0) < 1e-9


class TestMeanRank:
    def test_rank1(self):
        assert compute_mean_rank(["a","b","c"], ["a"]) == 1.0

    def test_rank3(self):
        assert compute_mean_rank(["x","y","a"], ["a"]) == 3.0

    def test_not_retrieved(self):
        assert math.isnan(compute_mean_rank(["x","y"], ["a"]))

    def test_two_gold_mean(self):
        # "a" at rank 1, "b" at rank 3 -> mean = 2.0
        assert compute_mean_rank(["a","x","b"], ["a","b"]) == 2.0


# ============================================================
# NLP Metrics
# ============================================================

class TestNormalizeAnswer:
    def test_lowercase(self):
        assert normalize_answer("HELLO") == "hello"

    def test_strip_punctuation(self):
        assert normalize_answer("hello!") == "hello"
        assert normalize_answer("9003.") == "9003"

    def test_collapse_whitespace(self):
        assert normalize_answer("hello   world") == "hello world"

    def test_combined(self):
        assert normalize_answer("  Hello, World!  ") == "hello world"


class TestExactMatch:
    def test_matching_number_in_content(self):
        # The NSN baseline returns full memory content; EM uses substring
        pred = "Day 20: PostgreSQL production port migrated to 9003."
        gt = "9003"
        assert compute_exact_match(pred, gt) == 1.0

    def test_no_match(self):
        assert compute_exact_match("5003", "9003") == 0.0

    def test_case_insensitive(self):
        assert compute_exact_match("PostgreSQL", "postgresql") == 1.0

    def test_punctuation_stripped(self):
        assert compute_exact_match("9003.", "9003") == 1.0

    def test_empty_prediction(self):
        assert compute_exact_match("", "9003") == 0.0

    def test_empty_ground_truth(self):
        assert compute_exact_match("9003", "") == 0.0


class TestTokenF1:
    def test_perfect_overlap(self):
        assert compute_token_f1("hello world", "hello world") == 1.0

    def test_zero_overlap(self):
        assert compute_token_f1("aaa bbb", "xxx yyy") == 0.0

    def test_empty_inputs(self):
        assert compute_token_f1("", "answer") == 0.0
        assert compute_token_f1("answer", "") == 0.0

    def test_partial_overlap(self):
        # pred="the correct answer", gt="the answer"
        # common={"the":1,"answer":1} -> 2 common tokens
        # precision=2/3, recall=2/2=1.0 -> F1=2*(2/3*1)/(2/3+1)=0.8
        f1 = compute_token_f1("the correct answer", "the answer")
        assert abs(f1 - 0.8) < 1e-9


class TestRougeL:
    def test_perfect_score(self):
        assert compute_rouge_l("hello world", "hello world") == 1.0

    def test_zero_score(self):
        assert compute_rouge_l("aaa bbb", "xxx yyy") == 0.0

    def test_partial_overlap(self):
        score = compute_rouge_l("hello world extra", "hello world")
        assert 0 < score <= 1.0

    def test_empty_inputs(self):
        assert compute_rouge_l("", "world") == 0.0
        assert compute_rouge_l("world", "") == 0.0
