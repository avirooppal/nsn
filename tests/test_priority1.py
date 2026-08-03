import pytest
import os
import json
import numpy as np
import faiss
from neurosleepnet.storage.sqlite import SQLiteAdapter
from neurosleepnet.storage.local_vector import TieredVectorStore
from neurosleepnet.memory.schemas import MemoryRecord

def test_sqlite_fts5(tmp_path):
    db_path = str(tmp_path / "test.db")
    storage = SQLiteAdapter(db_path=db_path)
    
    # Insert some data
    storage.store("m1", "The quick brown fox jumps over the lazy dog", "2026-01-01T00:00:00Z")
    storage.store("m2", "A fast brown fox", "2026-01-01T00:00:00Z")
    storage.store("m3", "Unrelated sentence about cats", "2026-01-01T00:00:00Z")
    
    results = storage.search_keyword("brown fox")
    assert len(results) == 2
    
    results_lazy = storage.search_keyword("lazy")
    assert len(results_lazy) == 1
    assert results_lazy[0]["id"] == "m1"

def test_tiered_vector_store_hnsw(tmp_path):
    db_path = str(tmp_path / "test.db")
    storage = SQLiteAdapter(db_path=db_path)
    
    # Store with token embeddings
    emb1 = [[1.0] + [0.0]*383, [0.0, 1.0] + [0.0]*382]  # Token 1 and 2
    emb2 = [[0.0, 1.0] + [0.0]*382, [0.0, 0.0, 1.0] + [0.0]*381] # Token 2 and 3
    
    storage.store("m1", "content 1", "2026-01-01T00:00:00Z", embedding=json.dumps(emb1))
    storage.store("m2", "content 2", "2026-01-01T00:00:00Z", embedding=json.dumps(emb2))
    
    vector_store = TieredVectorStore(storage, index_path=str(tmp_path / "test.faiss"), ids_path=str(tmp_path / "test_ids.json"))
    
    assert vector_store.hnsw_index.ntotal == 4 # 4 tokens total
    assert vector_store.ivfpq_index is None
    
    query = [[1.0] + [0.0]*383] # Search for Token 1
    results = vector_store.search(query, limit=2)
    assert len(results) >= 1
    assert results[0]["id"] == "m1"
    
def test_tiered_vector_store_ivfpq(tmp_path):
    db_path = str(tmp_path / "test.db")
    storage = SQLiteAdapter(db_path=db_path)
    
    vector_store = TieredVectorStore(storage, index_path=str(tmp_path / "test.faiss"), ids_path=str(tmp_path / "test_ids.json"))
    vector_store.buffer_limit = 1000 # Trigger IVFPQ after 1000 items
    
    # Add enough vectors to bypass the 1000-vector training threshold in local_vector.py
    for i in range(1005):
        emb = [list(np.random.rand(384))]
        storage.store(f"m_{i}", f"content {i}", "2026-01-01", embedding=json.dumps(emb))
        vector_store.add(f"m_{i}", emb, persist=False)
        
    # Trigger final persist/build to ensure state is clean
    if vector_store.hnsw_index.ntotal > vector_store.buffer_limit:
        vector_store._build_from_storage_ivfpq()
        
    assert vector_store.ivfpq_index is not None
    assert vector_store.ivfpq_index.ntotal == 1005
    assert vector_store.hnsw_index.ntotal == 0
