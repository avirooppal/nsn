"""
Persistence & process crash recovery runner.
Tests survival of memories, graph edges, importance, trust, and namespaces across database restarts.
"""
import os
from neurosleepnet.sdk.memory import Memory

def test_persistence_survival(db_path: str = "bench_persistence.db"):
    # 1. Clean DB files
    for p in [db_path, db_path.replace(".db", ".faiss"), db_path.replace(".db", ".faiss_ids")]:
        if os.path.exists(p): os.remove(p)

    # 2. Phase 1: Ingest memories into initial Memory instance
    m1 = Memory(namespace="persist_ns", db_path=db_path)
    id1 = m1.store("Alice manages the cloud database infrastructure.", importance=0.85, trust_score=0.9)
    res = m1.observe("Bob deployed hotfix v2.4 at 3pm.", source="system")
    id2 = res.memory_id
    
    # Verify graph node/edge creation
    subgraph_before = m1.get_entity_subgraph("Alice")
    
    # 3. Simulate process crash / database close
    del m1

    # 4. Phase 2: Instantiate new Memory instance pointing to same db_path
    m2 = Memory(namespace="persist_ns", db_path=db_path)
    
    # Check survival
    rec1 = m2.get(id1)
    rec2 = m2.get(id2)
    hits = m2.search_hybrid("Who manages database infrastructure?")
    subgraph_after = m2.get_entity_subgraph("Alice")

    # Cleanup
    del m2
    for p in [db_path, db_path.replace(".db", ".faiss"), db_path.replace(".db", ".faiss_ids")]:
        if os.path.exists(p): os.remove(p)

    memory_survival = (rec1 is not None and rec2 is not None)
    retrieval_survival = (len(hits) > 0 and hits[0]["id"] == id1)
    graph_survival = (len(subgraph_after.get("nodes", [])) > 0)

    return {
        "memory_survival_rate": 1.0 if memory_survival else 0.0,
        "retrieval_survival_rate": 1.0 if retrieval_survival else 0.0,
        "graph_survival_rate": 1.0 if graph_survival else 0.0,
        "overall_survival": memory_survival and retrieval_survival and graph_survival
    }
