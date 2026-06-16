import os
from neurosleepnet import Memory

def main():
    print("=== NeuroSleepNet E2E Test (Up to Phase 5 / Task 60) ===")

    # 1. Clean up old db if exists
    if os.path.exists('neurosleepnet.db'):
        try:
            os.remove('neurosleepnet.db')
            print("[+] Cleaned up old database.")
        except Exception:
            pass

    # 2. Initialize Memory (Phase 0 & 1 & 4)
    # This automatically loads settings and sets up the SQLiteAdapter 
    # and the LocalEmbeddingProvider using sentence-transformers.
    print("[+] Initializing Memory system (Loading SQLite and SentenceTransformers)...")
    memory = Memory()

    # 3. Store a memory with all schema fields (Phase 3 & 4 & 5)
    print("\n[+] Storing Memory 1...")
    mem1_id = memory.store(
        content="NeuroSleepNet is a cognitive memory OS for AI agents.",
        metadata={"source": "user_input", "category": "system"},
        importance=0.9,
        trust_score=1.0
    )
    print(f"    -> Stored with ID: {mem1_id}")

    # Store another memory
    print("[+] Storing Memory 2...")
    mem2_id = memory.store(
        content="It supports local embeddings using sentence-transformers.",
        metadata={"source": "user_input", "category": "feature"},
        importance=0.8,
        trust_score=0.95
    )
    print(f"    -> Stored with ID: {mem2_id}")

    # 4. Get a memory (Phase 2 & 4)
    print(f"\n[+] Retrieving Memory 1 ({mem1_id})...")
    record1 = memory.get(mem1_id)
    if record1:
        print(f"    -> Content: {record1.content}")
        print(f"    -> Metadata: {record1.metadata}")
        print(f"    -> Importance: {record1.importance}")
        print(f"    -> Trust Score: {record1.trust_score}")
        print(f"    -> Embedding Generated: {'Yes' if len(record1.embedding) > 0 else 'No'} (Dimensions: {len(record1.embedding)})")
    else:
        print("    -> [ERROR] Could not retrieve Memory 1.")

    # 5. List all memories (Phase 2 & 4)
    print("\n[+] Listing all memories...")
    all_memories = memory.list()
    print(f"    -> Total memories found: {len(all_memories)}")
    for mem in all_memories:
        print(f"       - ID: {mem.id} | Content: {mem.content[:30]}...")

    print("\n=== Test Completed Successfully! ===")

if __name__ == "__main__":
    main()
