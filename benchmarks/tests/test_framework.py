"""
Unit tests for the evaluation framework itself (metrics, adapters, persistence, and synthetic generators).
"""

import unittest
import math
from benchmarks.metrics.nlp import compute_exact_match, compute_token_f1, compute_rouge_l, compute_bleu_4
from benchmarks.metrics.retrieval import compute_recall_at_k, compute_mrr, compute_ndcg_at_k, compute_hit_at_k
from benchmarks.metrics.statistics import compute_summary_statistics, paired_t_test
from benchmarks.datasets.adapters import LongMemEvalAdapter
from benchmarks.datasets.synthetic import SyntheticBenchmarkGenerator

class TestBenchmarkFramework(unittest.TestCase):
    def test_nlp_metrics(self):
        pred = "The production port is 9999."
        gt = "9999"
        self.assertEqual(compute_exact_match(pred, gt), 1.0)
        self.assertGreater(compute_token_f1(pred, gt), 0.0)
        self.assertGreater(compute_rouge_l(pred, gt), 0.0)
        
    def test_retrieval_metrics(self):
        retrieved = ["m1", "m2", "m3", "m4", "m5"]
        ground_truth = ["m3"]
        self.assertEqual(compute_hit_at_k(retrieved, ground_truth, k=5), 1.0)
        self.assertEqual(compute_recall_at_k(retrieved, ground_truth, k=5), 1.0)
        self.assertEqual(compute_mrr(retrieved, ground_truth), 1.0 / 3.0)
        self.assertAlmostEqual(compute_ndcg_at_k(retrieved, ground_truth, k=5), 1.0 / math.log2(4), places=4)

    def test_statistics(self):
        vals = [1.0, 2.0, 3.0, 4.0, 5.0]
        stats = compute_summary_statistics(vals)
        self.assertEqual(stats["mean"], 3.0)
        self.assertGreater(stats["std"], 0.0)
        
    def test_synthetic_generator(self):
        gen = SyntheticBenchmarkGenerator(seed=42)
        updates = gen.generate_knowledge_update_test(num_sequences=2)
        self.assertEqual(len(updates), 2)
        self.assertIn("observations", updates[0])
        self.assertIn("queries", updates[0])

    def test_adapter_availability(self):
        adapter = LongMemEvalAdapter(data_dir="non_existent_dir")
        self.assertFalse(adapter.is_available())

if __name__ == "__main__":
    unittest.main()
