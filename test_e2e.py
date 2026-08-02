from neurosleepnet import Memory
import json
import asyncio
from neurosleepnet import AsyncMemory

# Initialize memory (will clear test DB manually if needed)
import os
if os.path.exists("test_e2e.db"): os.remove("test_e2e.db")
if os.path.exists("test_e2e.faiss"): os.remove("test_e2e.faiss")
if os.path.exists("test_e2e.faiss_ids"): os.remove("test_e2e.faiss_ids")

memory = Memory(namespace="my_agent", db_path="test_e2e.db")

# Full pipeline ingestion
memory.observe("Alice is the lead engineer at NeuroSleepNet.", source="system")
memory.observe("NeuroSleepNet is a cognitive memory OS for AI agents.", source="system")
memory.observe("Bob deployed the system using Docker Compose yesterday.", source="agent")
memory.observe("The system crashed due to a memory leak in the vector store.", source="agent")
memory.observe("To deploy: run docker compose up -d --build", source="system")

# Duplicate detection
r = memory.observe("NeuroSleepNet is a cognitive memory OS for AI agents.", source="system")
assert r.is_duplicate == True, "Duplicate not detected"
assert r.stored == False, "Duplicate was stored"

# Hybrid retrieval
results = memory.search_hybrid("Who leads engineering?", limit=3)
assert len(results) >= 1
assert "Alice" in results[0]['content'], "Alice not retrieved for engineering query"

# Reasoning pack
pack = json.loads(memory.reasoning_pack("NeuroSleepNet"))
assert "context" in pack
assert "logical_rules" in pack
assert "key_facts" in pack

# Timeline
timeline = memory.timeline(memory_type="procedural")
assert any("docker" in t["content"].lower() for t in timeline), "Procedural memory not in timeline"

# Sleep cycle
assert memory.trigger_sleep() == True

# Proactive surfacing
# Note: we need to use a very low threshold because RRF scores are small
surfaced = memory.surface_relevant("What happened with the deployment?", threshold=0.01)
assert len(surfaced) >= 1

# Event hook
events = []
memory.on("stored", lambda e: events.append(e))
memory.observe("New fact after hook setup.", source="test")
assert len(events) == 1

# Namespace isolation
m2 = Memory(namespace="other_agent", db_path="test_e2e.db")
m2.observe("Agent 2 fact only.")
assert len(memory.list()) != len(m2.list()), "Namespaces are leaking"

# Async
async def test_async():
    am = AsyncMemory(namespace="async_test", db_path="test_e2e.db")
    r_async = await am.observe("async test memory")
    assert r_async.stored == True
asyncio.run(test_async())

print("ALL ACCEPTANCE TESTS PASSED.")
