import os
import sys

# Ensure local package can be imported
sys.path.insert(0, os.path.abspath(os.path.dirname(__file__)))

from neurosleepnet import Memory

def main():
    print("Initializing Memory (Intelligence Orchestrator)...")
    # Clean up old test db if it exists
    if os.path.exists("test_memory.db"):
        os.remove("test_memory.db")
    if os.path.exists("test_memory.faiss"):
        os.remove("test_memory.faiss")
    if os.path.exists("test_memory.faiss_ids"):
        os.remove("test_memory.faiss_ids")
        
    m = Memory(namespace="test_agent", db_path="test_memory.db")
    
    statements = [
        "Alice is a senior AI researcher at OpenAI.",
        "Bob built a new vector database.",
        "Alice and Bob are collaborating on a knowledge graph.",
        "NeuroSleepNet is a framework for SLM cognition.",
        "Alice hates traditional stateless RAG systems."
    ]
    
    print("\nIngesting statements...")
    for s in statements:
        res = m.observe(s)
        print(f"Stored: {res.stored} | Type: {res.memory_type} | Trust: {res.trust_score}")
        
    print("\nTriggering Sleep Cycle (Consolidation + Graph Analysis)...")
    m.trigger_sleep()
    
    print("\nRunning Hybrid Search for 'Alice knowledge graph'...")
    results = m.search_hybrid("Alice knowledge graph", limit=3)
    
    for i, r in enumerate(results):
        print(f"[{i+1}] {r['content']} (Hybrid Score: {r.get('hybrid_score', 0):.2f})")
        
    print("\nTesting Graph Subgraph for 'Alice'...")
    subgraph = m.get_entity_subgraph("Alice", depth=1)
    print(f"Nodes found: {len(subgraph['nodes'])}")
    print(f"Edges found: {len(subgraph['edges'])}")
    
    print("\nVerification Complete. No crashes detected.")

if __name__ == "__main__":
    main()
