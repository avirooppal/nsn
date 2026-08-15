"""
Unit tests verifying the Gold Evidence ID mapping pipeline.

Tests:
1. _ingest_with_id_map builds a correct synthetic_id -> actual_uuid map
2. _translate_gold_ids correctly maps synthetic IDs to actual IDs
3. Recall@5 is non-zero when gold IDs are correctly translated
4. Recall@5 is ZERO when gold IDs are NOT translated (demonstrates the bug)
5. Duplicate-rejected observations produce None and are excluded from map
"""

import sys, os
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "../..")))

import pytest
from benchmarks.runners.evaluator import _ingest_with_id_map, _translate_gold_ids
from benchmarks.metrics.retrieval import compute_recall_at_k, compute_mrr, compute_ndcg_at_k


class MockSystem:
    """Minimal mock system that returns sequential IDs."""
    def __init__(self, name="mock"):
        self.name = name
        self._db = {}
        self._n = 0

    def observe(self, content, source="system", metadata=None):
        actual_id = f"actual_uuid_{self._n}"
        self._db[actual_id] = content
        self._n += 1
        return actual_id

    def query(self, question, limit=5):
        ids = list(self._db.keys())
        return {
            "answer": list(self._db.values())[0] if self._db else "",
            "retrieved_ids": ids,
        }

    def reset(self):
        self._db = {}
        self._n = 0


class MockSystemDupReject:
    """Second observe() call always returns None (duplicate-rejected)."""
    def __init__(self):
        self.name = "mock_dup"
        self._n = 0

    def observe(self, content, source="system", metadata=None):
        self._n += 1
        if self._n == 2:
            return None
        return f"actual_uuid_{self._n}"

    def reset(self):
        self._n = 0


# ===== ID MAP TESTS =====

def test_ingest_id_map_basic():
    sys = MockSystem()
    obs = [
        {"id": "mem_0", "content": "Content A"},
        {"id": "mem_1", "content": "Content B"},
        {"id": "mem_2", "content": "Content C"},
    ]
    id_map = _ingest_with_id_map(sys, obs)
    assert "mem_0" in id_map
    assert "mem_1" in id_map
    assert "mem_2" in id_map
    assert id_map["mem_0"] == "actual_uuid_0"
    assert id_map["mem_1"] == "actual_uuid_1"
    assert id_map["mem_2"] == "actual_uuid_2"


def test_ingest_id_map_unique_values():
    sys = MockSystem()
    obs = [{"id": f"s_{i}", "content": f"Content {i}"} for i in range(10)]
    id_map = _ingest_with_id_map(sys, obs)
    actual_ids = list(id_map.values())
    assert len(actual_ids) == len(set(actual_ids)), "All actual IDs should be unique"


def test_ingest_id_map_no_id_key():
    """Observations without 'id' key should be ingested but not mapped."""
    sys = MockSystem()
    obs = [{"content": "No ID field here"}]
    id_map = _ingest_with_id_map(sys, obs)
    assert len(id_map) == 0   # no synthetic ID -> nothing to map


def test_translate_gold_ids_basic():
    id_map = {"mem_a": "uuid-001", "mem_b": "uuid-002", "mem_c": "uuid-003"}
    gold = ["mem_a", "mem_c"]
    assert _translate_gold_ids(gold, id_map) == ["uuid-001", "uuid-003"]


def test_translate_gold_ids_missing_key():
    """Missing synthetic IDs (duplicate-rejected) are silently excluded."""
    id_map = {"mem_a": "uuid-001"}
    gold = ["mem_a", "mem_b"]   # mem_b was rejected
    result = _translate_gold_ids(gold, id_map)
    assert result == ["uuid-001"]
    assert "uuid-002" not in result


def test_translate_gold_ids_all_missing():
    assert _translate_gold_ids(["mem_x", "mem_y"], {}) == []


def test_translate_gold_ids_empty_inputs():
    assert _translate_gold_ids([], {}) == []
    assert _translate_gold_ids([], {"mem_a": "uuid-1"}) == []


# ===== RECALL BUG REGRESSION TESTS =====

def test_recall_nonzero_after_correct_translation():
    """The bug fix: Recall@5 must be non-zero after ID translation."""
    retrieved = ["actual_uuid_0", "actual_uuid_1", "actual_uuid_2"]
    gold_synthetic = ["mem_0", "mem_1"]
    id_map = {"mem_0": "actual_uuid_0", "mem_1": "actual_uuid_1"}
    translated_gold = _translate_gold_ids(gold_synthetic, id_map)
    recall = compute_recall_at_k(retrieved, translated_gold, k=5)
    assert recall == 1.0, f"After correct ID translation, Recall@5 must be 1.0, got {recall}"


def test_recall_zero_without_translation():
    """The original bug: Recall@5 = 0 when synthetic IDs are used directly."""
    retrieved = ["actual_uuid_0", "actual_uuid_1", "actual_uuid_2"]
    gold_synthetic = ["mem_0", "mem_1"]   # synthetic IDs never appear in retrieved list
    recall = compute_recall_at_k(retrieved, gold_synthetic, k=5)
    assert recall == 0.0, "Without ID translation, Recall@5 must be 0 (demonstrating the bug)"


# ===== DUPLICATE REJECTION TESTS =====

def test_duplicate_rejection_excluded_from_map():
    sys = MockSystemDupReject()
    obs = [
        {"id": "obs_0", "content": "First"},
        {"id": "obs_1", "content": "Dup"},    # returns None
        {"id": "obs_2", "content": "Third"},
    ]
    id_map = _ingest_with_id_map(sys, obs)
    assert "obs_0" in id_map
    assert "obs_1" not in id_map   # duplicate-rejected
    assert "obs_2" in id_map


# ===== MRR POSITION TESTS =====

def test_mrr_rank1():
    assert compute_mrr(["uuid-001", "uuid-002"], ["uuid-001"]) == 1.0

def test_mrr_rank2():
    assert abs(compute_mrr(["uuid-X", "uuid-001", "uuid-Y"], ["uuid-001"]) - 0.5) < 1e-9

def test_mrr_rank3():
    assert abs(compute_mrr(["uuid-X", "uuid-Y", "uuid-001"], ["uuid-001"]) - 1/3) < 1e-9

def test_mrr_not_retrieved():
    assert compute_mrr(["uuid-X", "uuid-Y"], ["uuid-001"]) == 0.0
