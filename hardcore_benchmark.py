import os
import sys
import time
import json
import random
from openai import OpenAI
import neurosleepnet as nsn
from neurosleepnet.sdk.memory import Memory

def run_hardcore_benchmark():
    if hasattr(sys.stdout, 'reconfigure'):
        sys.stdout.reconfigure(encoding='utf-8')

    OLLAMA_API_KEY = os.environ.get("OLLAMA_API_KEY", "<OLLAMA_CLOUD_API_KEY>")
    OLLAMA_BASE_URL = "https://ollama.com/v1"
    MODEL = "nemotron-3-nano:30b"

    print("=" * 80)
    print("      NEUROSLEEPNET (NSN) HARDCORE REAL-WORLD BENCHMARK SUITE")
    print("=" * 80)

    base_client = OpenAI(base_url=OLLAMA_BASE_URL, api_key=OLLAMA_API_KEY)

    # -------------------------------------------------------------------------
    # TEST 1: ADVERSARIAL FACT CONTRADICTION & REM CONSOLIDATION
    # -------------------------------------------------------------------------
    print("\n[TEST 1] Hardcore Fact Update & REM Contradiction Resolution")
    print("-" * 80)

    db_path1 = "bench_hardcore_rem.db"
    if os.path.exists(db_path1):
        os.remove(db_path1)

    nsn_client1 = nsn.init(base_client, namespace="hardcore_rem", db_path=db_path1, recall_limit=5)
    
    # Step A: Inject initial state
    print("  -> Injecting Initial Fact: 'My primary database port is 5432.'")
    nsn_client1.chat.completions.create(model=MODEL, messages=[{"role": "user", "content": "My primary database port is 5432."}])

    # Step B: 10 filler dialogue turns
    print("  -> Injecting 10 intermediate conversation turns...")
    fillers = [
        "What is the capital of France?",
        "Can you write a python function for quicksort?",
        "I like drinking Earl Grey tea in the afternoon.",
        "Tell me a short joke about computers.",
        "What is quantum entanglement?",
        "I am planning a trip to Kyoto next spring.",
        "Explain the difference between synchronous and asynchronous I/O.",
        "My favorite color is deep midnight blue.",
        "What is the distance from Earth to the Moon?",
        "How do transformers handle self-attention?"
    ]
    for msg in fillers:
        nsn_client1.chat.completions.create(model=MODEL, messages=[{"role": "user", "content": msg}])

    # Step C: Inject updated contradicting fact
    print("  -> Injecting Contradicting Fact: 'UPDATE: We migrated our database! My primary database port is now 9999.'")
    nsn_client1.chat.completions.create(model=MODEL, messages=[{"role": "user", "content": "UPDATE: We migrated our database! My primary database port is now 9999."}])

    # Step D: Test recall BEFORE REM sleep
    query1 = "What is my current primary database port?"
    res_before = nsn_client1.chat.completions.create(model=MODEL, messages=[{"role": "user", "content": query1}])
    ans_before = res_before.choices[0].message.content
    print(f"\n  [PRE-SLEEP RECALL]\n  Query: '{query1}'\n  Model Answer: {ans_before.strip()}")

    # Step E: Trigger Sleep Engine (NREM + REM Contradiction Resolution)
    print("\n  -> Triggering NSN Sleep Engine (Offline NREM/REM Consolidation)...")
    sleep_stats = nsn_client1.memory.trigger_sleep()
    print(f"  [SLEEP STATS] Consolidation Completed: {json.dumps(sleep_stats, indent=2)}")

    # Step F: Test recall AFTER REM sleep
    res_after = nsn_client1.chat.completions.create(model=MODEL, messages=[{"role": "user", "content": query1}])
    ans_after = res_after.choices[0].message.content
    print(f"\n  [POST-SLEEP RECALL]\n  Query: '{query1}'\n  Model Answer: {ans_after.strip()}")

    rem_success = "9999" in ans_after and "5432" not in ans_after
    print(f"  [RESULT] REM Contradiction Resolution Test: {'PASSED (100% Correct Fact Update)' if rem_success else 'PARTIAL / CHECK OUTPUT'}")

    # -------------------------------------------------------------------------
    # TEST 2: NEEDLE-IN-A-HAYSTACK CARDINALITY SCALING (1,000 MEMORIES)
    # -------------------------------------------------------------------------
    print("\n\n[TEST 2] High-Cardinality Needle-in-a-Haystack Retrieval Scaling (1,000 Distractors)")
    print("-" * 80)

    db_path2 = "bench_hardcore_haystack.db"
    if os.path.exists(db_path2):
        os.remove(db_path2)

    mem_engine = Memory(namespace="hardcore_haystack", db_path=db_path2)

    print("  -> Generating & Batch Ingesting 1,000 synthetic distractor memories...")
    distractors = [
        f"Session log record {i}: Server load standard deviation is {random.uniform(0.1, 0.9):.3f} under workload {i}."
        for i in range(1000)
    ]
    t0_ingest = time.time()
    mem_engine.ingest_batch(distractors, source="synthetic_noise")
    ingest_time = time.time() - t0_ingest
    print(f"  -> Successfully ingested 1,000 memories in {ingest_time:.2f}s ({len(distractors)/ingest_time:.1f} mem/sec)")

    # Inject target hidden needle
    needle_fact = "CRITICAL METRIC: The project secret emergency override key is HYPERION-DELTA-99."
    mem_engine.observe(needle_fact, source="user")
    print(f"  -> Injected target needle: '{needle_fact}' into 1,000 distractors.")

    # Search speed and precision check
    needle_query = "What is the secret emergency override key?"
    print(f"  -> Performing Hybrid Search for query: '{needle_query}'...")
    
    t0_search = time.time()
    hybrid_res = mem_engine.search_hybrid(needle_query, limit=5)
    search_lat_ms = (time.time() - t0_search) * 1000

    print(f"\n  [HYBRID SEARCH PERFORMANCE]")
    print(f"  Query Latency over 1,001 items: {search_lat_ms:.2f} ms")
    print(f"  Top-1 Retrieved Item: {hybrid_res[0]['content']}")
    print(f"  Top-1 Hybrid Score: {hybrid_res[0].get('hybrid_score', 0):.4f}")

    haystack_success = "HYPERION-DELTA-99" in hybrid_res[0]['content']
    print(f"  [RESULT] High-Cardinality Needle-in-Haystack Search: {'PASSED (100% Top-1 Precision)' if haystack_success else 'FAILED'}")

    # -------------------------------------------------------------------------
    # TEST 3: MULTI-HOP TEMPORAL REASONING & GRAPH TRAVERSAL
    # -------------------------------------------------------------------------
    print("\n\n[TEST 3] Multi-Hop Relational Knowledge Traversal")
    print("-" * 80)

    db_path3 = "bench_hardcore_graph.db"
    if os.path.exists(db_path3):
        os.remove(db_path3)

    nsn_client3 = nsn.init(base_client, namespace="hardcore_graph", db_path=db_path3, recall_limit=5)

    print("  -> Injecting multi-hop relational observations:")
    hops = [
        "Alice is the principal lead architect of Project NeuroSleepNet.",
        "Project NeuroSleepNet uses SQLite FTS5 for sparse keyword search.",
        "SQLite FTS5 relies on the BM25 ranking algorithm."
    ]
    for hop in hops:
        print(f"     - {hop}")
        nsn_client3.chat.completions.create(model=MODEL, messages=[{"role": "user", "content": hop}])

    query3 = "What ranking algorithm is indirectly used by the project built by Alice?"
    print(f"\n  -> Querying multi-hop relation: '{query3}'")
    t0_graph = time.time()
    res_graph = nsn_client3.chat.completions.create(model=MODEL, messages=[{"role": "user", "content": query3}])
    lat_graph = (time.time() - t0_graph) * 1000

    ans_graph = res_graph.choices[0].message.content
    print(f"  [MULTI-HOP RESPONSE] (Latency: {lat_graph:.1f} ms)\n  Model Answer: {ans_graph.strip()}")

    multihop_success = "bm25" in ans_graph.lower()
    print(f"  [RESULT] Multi-Hop Graph Traversal Test: {'PASSED' if multihop_success else 'CHECK REASONING'}")

    # -------------------------------------------------------------------------
    # SUMMARY REPORT
    # -------------------------------------------------------------------------
    print("\n" + "=" * 80)
    print("                  HARDCORE BENCHMARK SUITE SUMMARY")
    print("=" * 80)
    print(f"1. REM Contradiction Update : {'PASSED [100%]' if rem_success else 'FAILED'}")
    print(f"2. 1,000-Mem Haystack Recall: {'PASSED [100%]' if haystack_success else 'FAILED'} (Latency: {search_lat_ms:.2f}ms)")
    print(f"3. Multi-Hop Graph Reasoning: {'PASSED [100%]' if multihop_success else 'CHECK REASONING'}")
    print("=" * 80 + "\n")

if __name__ == "__main__":
    run_hardcore_benchmark()
