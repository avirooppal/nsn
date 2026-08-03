import os, json

# Clean slate
for f in ['smoke.db', 'smoke.faiss', 'smoke.faiss_ids']:
    if os.path.exists(f): os.remove(f)

print("--- MemoryTool smoke test ---")
from neurosleepnet.integrations.tool import MemoryTool
tool = MemoryTool(namespace="smoke", db_path="smoke.db")

r = tool.remember("Alice is the CEO of TechCorp.")
print(f"remember(): stored={r.stored}, type={r.memory_type}")

hits = tool.recall("Who is the CEO?")
content = hits[0]["content"]
print(f"recall(): {content}")

ctx = json.loads(tool.reasoning_context("TechCorp"))
print(f"reasoning_context(): {len(ctx['key_facts'])} key_facts, {len(ctx['context'])} context items")
print(f"key_facts[0]: {ctx['key_facts'][0] if ctx['key_facts'] else 'none'}")

print("\n--- OpenAI MemoryInjector smoke test ---")
from neurosleepnet.integrations.openai_adapter import MemoryInjector
inj = MemoryInjector(namespace="smoke", db_path="smoke.db")
msgs = [{"role": "user", "content": "Who is the CEO?"}]
enriched = inj.inject("Who is the CEO?", msgs)
has_system = enriched[0]["role"] == "system"
print(f"inject(): {len(enriched)} messages, system message present={has_system}")
print(f"system content preview: {enriched[0]['content'][:80]}...")

print("\n--- MemoryTool callable interface ---")
result = tool("recall", query="CEO")
print(f"tool('recall', query='CEO'): {len(result)} results")

print("\n--- All integration smoke tests PASSED ---")
