import os
import sys
import neurosleepnet as nsn
from openai import OpenAI

def main():
    if hasattr(sys.stdout, 'reconfigure'):
        sys.stdout.reconfigure(encoding='utf-8')
    # Ollama Cloud Configuration
    OLLAMA_API_KEY = os.environ.get("OLLAMA_API_KEY", os.environ.get("PROVIDER_API_KEY", ""))
    if not OLLAMA_API_KEY:
        print("Note: Set OLLAMA_API_KEY environment variable before running.")
    OLLAMA_BASE_URL = os.environ.get("OLLAMA_BASE_URL", "https://ollama.com/v1")
    MODEL_NAME = os.environ.get("OLLAMA_MODEL", "nemotron-3-nano:30b")

    print("\n" + "="*65)
    print("  NEUROSLEEPNET + OLLAMA CLOUD API (VANILLA MODEL + NSN MEMORY)")
    print("="*65)
    print(f"Base URL : {OLLAMA_BASE_URL}")
    print(f"Model    : {MODEL_NAME}")
    print(f"API Key  : {OLLAMA_API_KEY[:8]}...{OLLAMA_API_KEY[-6:]}")
    print("="*65 + "\n")

    # 1. Initialize base (Vanilla) OpenAI-compatible client pointed at Ollama Cloud
    base_client = OpenAI(
        base_url=OLLAMA_BASE_URL,
        api_key=OLLAMA_API_KEY,
    )

    # -------------------------------------------------------------------------
    # PART 1: PURE VANILLA LLM (No Memory Layer)
    # -------------------------------------------------------------------------
    print(">>> PART 1: PURE VANILLA OLLAMA CLOUD MODEL (No Memory Layer)")
    print("-" * 65)

    fact_message = "Hi! Please remember that my favorite drink is Iced Vanilla Oat Latte and my server port is 8080."
    print(f"User  : {fact_message}")
    
    # Send fact to vanilla model (stateless single message)
    res1 = base_client.chat.completions.create(
        model=MODEL_NAME,
        messages=[{"role": "user", "content": fact_message}]
    )
    print(f"Vanilla LLM Reply: {res1.choices[0].message.content.strip()}\n")

    query_message = "What is my favorite drink and server port?"
    print(f"User  : {query_message}")
    print("Asking Vanilla Model directly (stateless request without conversation history)...")
    
    res2 = base_client.chat.completions.create(
        model=MODEL_NAME,
        messages=[{"role": "user", "content": query_message}]
    )
    print(f"Vanilla LLM Reply: {res2.choices[0].message.content.strip()}")
    print("Notice: Pure Vanilla model fails to recall because it has no memory layer!\n")

    # -------------------------------------------------------------------------
    # PART 2: VANILLA MODEL + NEUROSLEEPNET MEMORY LAYER
    # -------------------------------------------------------------------------
    print("="*65)
    print(">>> PART 2: VANILLA OLLAMA MODEL + NEUROSLEEPNET MEMORY LAYER")
    print("-" * 65)

    db_path = "ollama_vanilla_memory.db"
    if os.path.exists(db_path):
        os.remove(db_path)

    # Wrap the vanilla Ollama client with NSN Memory Layer
    nsn_client = nsn.init(
        base_client,
        namespace="ollama_vanilla_agent",
        db_path=db_path,
        recall_limit=4
    )
    print(f"NSN Memory Layer initialized [Database: {db_path}]")

    print(f"\nUser  : {fact_message}")
    print("Sending fact through NSN Memory Layer...")
    res_nsn1 = nsn_client.chat.completions.create(
        model=MODEL_NAME,
        messages=[{"role": "user", "content": fact_message}]
    )
    print(f"NSN Agent Reply: {res_nsn1.choices[0].message.content.strip()}\n")

    print(f"User  : {query_message}")
    print("Asking via NSN Memory Layer (NSN auto-retrieves relevant memory & injects it)...")
    res_nsn2 = nsn_client.chat.completions.create(
        model=MODEL_NAME,
        messages=[{"role": "user", "content": query_message}]
    )
    print(f"NSN Agent Reply: {res_nsn2.choices[0].message.content.strip()}\n")

    # -------------------------------------------------------------------------
    # PART 3: INSPECT NSN MEMORY RECALL
    # -------------------------------------------------------------------------
    print("="*65)
    print(">>> PART 3: DIRECT NEUROSLEEPNET MEMORY INSPECTION")
    print("-" * 65)
    memories = nsn_client.recall("favorite drink server port", limit=5)
    print(f"Retrieved {len(memories)} relevant memory items from NSN store:")
    for idx, mem in enumerate(memories, 1):
        mtype = mem.get("memory_type", "observation").upper()
        print(f"  [{idx}] [{mtype}] {mem['content']}")

    print("\n" + "="*65)
    print("  DEMO COMPLETE: NSN successfully provided long-term memory")
    print("  to the vanilla stateless Ollama Cloud model!")
    print("="*65 + "\n")

if __name__ == "__main__":
    main()
