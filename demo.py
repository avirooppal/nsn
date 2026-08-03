import os
import sys
import json

# Ensure local package can be imported if run from repo root
sys.path.insert(0, os.path.abspath(os.path.dirname(__file__)))

from neurosleepnet import Memory

def main():
    print("="*60)
    print(">> NeuroSleepNet v0.2.0 - Capability Demonstration")
    print("="*60)
    
    # Cleanup old test databases if they exist
    for f in ["demo_memory.db", "demo_memory.faiss", "demo_memory.faiss_ids"]:
        if os.path.exists(f):
            try:
                os.remove(f)
            except OSError:
                pass
            
    print("\n[1] Initializing Intelligence Orchestrator (Namespace: demo_agent)")
    memory = Memory(namespace="demo_agent", db_path="demo_memory.db")
    
    print("\n[2] Simulating Agent Observations (Ingestion Pipeline)")
    statements = [
        "Alice is the lead architect for project Alpha.",
        "Project Alpha is a revolutionary AI framework.",
        "Bob works under Alice on project Alpha.",
        "The project relies heavily on vector databases.",
        "To start the server, you must run: docker-compose up -d.",
        "Alice is the lead architect for project Alpha." # Intentional duplicate
    ]
    
    for s in statements:
        print(f"\nObserving: '{s}'")
        res = memory.observe(s, source="agent_sensor")
        if res.is_duplicate:
            print(" -> [Result] DUPLICATE DETECTED. Ignored.")
        else:
            print(f" -> [Result] Stored! Type: {res.memory_type.upper()}, Trust Score: {res.trust_score}")
            
    print("\n[3] Testing Hybrid Search (Semantic + Keyword + Graph)")
    query = "Who leads the AI framework project?"
    print(f"\nQuery: '{query}'")
    results = memory.search_hybrid(query, limit=3)
    for i, r in enumerate(results):
        print(f"  {i+1}. {r['content']} (Hybrid Score: {r.get('hybrid_score', 0):.4f})")
        
    print("\n[4] Knowledge Graph Subgraph Extraction")
    print("Extracting subgraph for 'Alice' (Depth=1)...")
    subgraph = memory.get_entity_subgraph("Alice", depth=1)
    if subgraph and subgraph.get('nodes'):
        print(f" -> Nodes: {[n['name'] for n in subgraph['nodes']]}")
        print(f" -> Edges: {[e['relation'] + ' -> ' + e['target']['name'] for e in subgraph.get('edges', [])]}")
    else:
        print(" -> No subgraph found. Graph extraction might not have triggered properly.")
        
    print("\n[5] Proactive Memory Surfacing")
    print("Context: 'I need to deploy the application.'")
    # Low threshold because RRF scores tend to be very small decimals
    surfaced = memory.surface_relevant("I need to deploy the application.", threshold=0.01)
    for s in surfaced:
        print(f" -> Surfaced ({s['memory_type']}): {s['content']} (Relevance: {s['relevance_score']:.4f})")
        
    print("\n[6] Generating Reasoning Pack for SLM")
    topic = "Project Alpha"
    print(f"Topic: '{topic}'")
    pack_str = memory.reasoning_pack(topic)
    pack = json.loads(pack_str)
    print("\n--- Reasoning Pack JSON ---")
    print(json.dumps(pack, indent=2))
    print("---------------------------")
    
    print("\n[7] Triggering Sleep Cycle (Consolidation & Decay)")
    memory.trigger_sleep()
    print(" -> NREM Synthesis, REM Pruning, and Importance Decay completed offline.")
    
    print("\n[8] Memory Timeline (Procedural Only)")
    timeline = memory.timeline(memory_type="procedural")
    for t in timeline:
        print(f" -> [{t['created_at']}] {t['content']}")
        
    print("\n" + "="*60)
    print("[OK] Demonstration Complete!")
    print("="*60)

if __name__ == "__main__":
    main()
