"""
Tests for Phase 00 — Evaluation Integrity and Fairness.

Verifies:
1. A fixture with four questions and rejected gold evidence still reports four evaluated
   questions and counts missing evidence as failure.
2. A multi-evidence fixture cannot improve recall by dropping unavailable gold IDs.
3. A crashing generator records a failed question without a fallback answer;
   retrieval and generation timers remain distinct.
4. Repeated isolated runs have identical input hashes and no leftover index contamination;
   different requested seeds reach the generator.
"""

import os
import sys
import tempfile
import pytest

from benchmarks.runners.evaluator import BenchmarkEvaluator, _translate_gold_ids
from benchmarks.runners.generator import MockGenerator
from benchmarks.metrics.retrieval import compute_recall_at_k


class DummyQuery:
    def __init__(self, query_id, question, gold_memory_ids, ground_truth_answer="answer"):
        self.query_id = query_id
        self.question = question
        self.gold_memory_ids = gold_memory_ids
        self.ground_truth_answer = ground_truth_answer
        self.category = "test"
        self.difficulty = "easy"


class MockRejectingSystem:
    """A system that rejects all observations (returns None), simulating duplicate drop."""
    def __init__(self, name="mock_reject"):
        self.name = name

    def observe(self, content, source="system", metadata=None):
        return None

    def query(self, question, limit=5):
        return {
            "answer": "wrong guess",
            "retrieved_ids": [],
            "latency_ms": 5.0,
            "retrieval_latency_ms": 4.0,
            "generation_latency_ms": 1.0,
        }

    def reset(self):
        pass


class MockCrashingGeneratorSystem:
    """A system where the generator crashes during query."""
    def __init__(self, generator=None, name="mock_crash"):
        self.name = name
        self.generator = generator or MockGenerator(fail_on_prompts=["crash"])
        self.seed = None

    def set_seed(self, seed: int):
        self.seed = seed
        if self.generator:
            self.generator.set_seed(seed)

    def observe(self, content, source="system", metadata=None):
        return "stored_id_1"

    def query(self, question, limit=5):
        retrieval_latency = 2.0
        retrieved_ids = ["stored_id_1"]
        # Generator crashes on purpose:
        self.generator.generate(question)
        return {
            "answer": "ok",
            "retrieved_ids": retrieved_ids,
            "retrieval_latency_ms": retrieval_latency,
            "generation_latency_ms": 3.0,
            "latency_ms": 5.0,
        }

    def reset(self):
        pass


def test_fixture_four_questions_rejected_gold():
    """
    Acceptance criterion 1:
    A fixture with four questions and rejected gold evidence still reports four evaluated
    questions and counts missing evidence as failure (no skipping).
    """
    evaluator = BenchmarkEvaluator()
    system = MockRejectingSystem()

    queries = [
        DummyQuery(f"q_{i}", f"Question {i}?", [f"gold_{i}"], ground_truth_answer=f"gt_{i}")
        for i in range(4)
    ]
    # No gold IDs could be translated because all observations were rejected
    id_map = {}

    result = evaluator._evaluate_query_set(
        system, queries, id_map, benchmark_name="test_bench", is_full_context=False
    )

    assert result["samples"] == 4, f"Expected 4 samples, got {result['samples']}"
    assert result["2x2_matrix"]["retrieval_failures"] == 4
    assert result["recall_5"] == 0.0
    assert result["hit_5"] == 0.0
    assert result["2x2_matrix"]["duplicate_rejected_skipped"] == 4


def test_multievidence_cannot_drop_unavailable_ids():
    """
    Acceptance criterion 2:
    A multi-evidence fixture cannot improve recall by dropping unavailable gold IDs.
    If 2 gold IDs are required, but 1 was rejected and 1 retrieved, recall must be 0.5, NOT 1.0.
    """
    evaluator = BenchmarkEvaluator()

    class MockPartialRetrievalSystem:
        def __init__(self):
            self.name = "mock_partial"

        def observe(self, content, source="system", metadata=None):
            return "actual_uuid_1"

        def query(self, question, limit=5):
            return {
                "answer": "partial",
                "retrieved_ids": ["actual_uuid_1"],
                "latency_ms": 3.0,
            }

        def reset(self):
            pass

    system = MockPartialRetrievalSystem()
    # gold requires both mem_1 and mem_2. mem_2 was rejected (not in id_map).
    queries = [
        DummyQuery("q_multi", "Multi-hop question?", ["mem_1", "mem_2"], ground_truth_answer="ans")
    ]
    id_map = {"mem_1": "actual_uuid_1"}  # mem_2 is absent

    result = evaluator._evaluate_query_set(
        system, queries, id_map, benchmark_name="test_bench", is_full_context=False
    )

    assert result["samples"] == 1
    # Recall must be 1 / 2 = 0.5, never 1.0!
    assert result["recall_5"] == 0.5, f"Expected Recall@5 to be 0.5, got {result['recall_5']}"


def test_crashing_generator_records_failed_question_no_fallback():
    """
    Acceptance criterion 3:
    A crashing generator records a failed question without a fallback answer;
    retrieval and generation timers remain distinct.
    """
    evaluator = BenchmarkEvaluator()
    generator = MockGenerator(fail_on_prompts=["crash"])
    system = MockCrashingGeneratorSystem(generator=generator)

    queries = [
        DummyQuery("q_crash", "Please crash now", ["mem_1"], ground_truth_answer="expected")
    ]
    id_map = {"mem_1": "stored_id_1"}

    result = evaluator._evaluate_query_set(
        system, queries, id_map, benchmark_name="test_bench", is_full_context=False
    )

    assert result["samples"] == 1
    assert result["failed_questions"] == 1
    assert result["exact_match"] == 0.0
    # Generation and retrieval latency stats are recorded separately
    assert "p50_retrieval_latency" in result
    assert "p50_generation_latency" in result


def test_repeated_isolated_runs_seeds_and_clean_state():
    """
    Acceptance criterion 4:
    Repeated isolated runs have identical input hashes and no leftover index contamination;
    different requested seeds reach the generator.
    """
    evaluator = BenchmarkEvaluator()
    gen1 = MockGenerator()
    sys1 = MockCrashingGeneratorSystem(generator=gen1)

    res1 = evaluator.evaluate_update_benchmark(sys1, num_samples=10, seed=123)
    res2 = evaluator.evaluate_update_benchmark(sys1, num_samples=10, seed=123)
    res_diff_seed = evaluator.evaluate_update_benchmark(sys1, num_samples=10, seed=999)

    # Identical seed -> identical dataset hash
    assert res1["dataset_hash"] == res2["dataset_hash"]
    # Different seed -> different dataset hash
    assert res1["dataset_hash"] != res_diff_seed["dataset_hash"]
    # Seed reached the system/generator
    assert sys1.seed == 999
    assert gen1.seed == 999


def test_isolated_cleanup_leaves_no_index_contamination():
    """
    Acceptance criterion 4 (part 2):
    Verify that baseline systems wipe databases and FAISS/WAL/SHM index artifacts cleanly on reset.
    """
    from benchmarks.baselines.bm25 import BM25System

    with tempfile.TemporaryDirectory() as tmpdir:
        db_path = os.path.join(tmpdir, "test_clean.db")
        system = BM25System(db_path=db_path)
        system.observe("hello clean world")
        assert os.path.exists(db_path)

        system.reset()
        # Storage was re-initialized clean, no leftover observations
        res = system.query("hello")
        assert res["retrieved_ids"] == []
