import os, json, logging

# Clean up previous run
for f in os.listdir("."):
    if f.startswith("agent_memory") and (f.endswith(".db") or f.endswith(".faiss") or f.endswith(".faiss_ids")):
        os.remove(f)

# ================================================================
# Step 1: Define / initialise your model (any LLM or callable)
# ================================================================
class MockLLM:
    """Stand-in for any real LLM (OpenAI, Ollama, HuggingFace, etc.)"""
    def __call__(self, prompt: str) -> str:
        return f"[LLM] Response to: {prompt[:60]}"

    def invoke(self, prompt: str) -> str:
        return self(prompt)

model = MockLLM()

# ================================================================
# Step 2: Wrap it with NSN — that's the entire integration.
# ================================================================
import neurosleepnet
model = neurosleepnet.NSN(model, namespace="my_agent", db_path="agent_memory.db")

# ================================================================
# From here, 'model' works exactly like before — but now has
# persistent cognitive memory attached automatically.
# ================================================================

SEP = "-" * 56

def section(n, title):
    print(f"\n{'='*56}")
    print(f"  {n}. {title}")
    print(f"{'='*56}")

# ---------------------------------------------------------------
# 1. Manually load facts into memory
# ---------------------------------------------------------------
section(1, "Loading facts into memory")

facts = [
    ("Alice is the lead engineer at NeuroSleepNet.", "system"),
    ("NeuroSleepNet is a cognitive memory OS for AI agents.", "system"),
    ("Bob deployed the system with docker-compose up -d.", "agent"),
    ("The vector store crashed due to a memory leak.", "agent"),
    ("To deploy: run docker-compose up -d --build", "system"),
    ("Alice is the lead engineer at NeuroSleepNet.", "system"),  # intentional duplicate
]

for content, source in facts:
    r = model.remember(content, source=source)
    if r.is_duplicate:
        print(f"  [DUPLICATE] {content[:52]}...")
    else:
        print(f"  [STORED]    type={r.memory_type:<11} trust={r.trust_score:.2f}  {content[:42]}...")

# ---------------------------------------------------------------
# 2. Call the model normally — memory is auto-injected & stored
# ---------------------------------------------------------------
section(2, "Calling the model (memory auto-injected)")

queries = [
    "Who leads the engineering team?",
    "How do I deploy the application?",
    "What went wrong with the vector store?",
]

for q in queries:
    response = model(q)
    print(f"  Q: {q}")
    print(f"  A: {response}")
    print(f"  {SEP}")

# ---------------------------------------------------------------
# 3. Recall — search memory directly
# ---------------------------------------------------------------
section(3, "Direct memory recall")

hits = model.recall("Who leads engineering?", limit=3)
for i, h in enumerate(hits, 1):
    print(f"  {i}. [{h['memory_type'].upper():<11}] {h['content']}")

# ---------------------------------------------------------------
# 4. Reasoning pack (SLM context injection)
# ---------------------------------------------------------------
section(4, "Reasoning pack for SLM injection")

pack = json.loads(model.memory.reasoning_pack("NeuroSleepNet"))
print(f"  Key Facts  ({len(pack['key_facts'])}):")
for f in pack["key_facts"]:
    print(f"    - {f}")
print(f"  Graph Rules: {pack['logical_rules'] or 'none extracted'}")

# ---------------------------------------------------------------
# 5. Sleep cycle (NREM + REM + Decay)
# ---------------------------------------------------------------
section(5, "Offline sleep cycle")

logging.disable(logging.CRITICAL)
ok = model.sleep()
logging.disable(logging.NOTSET)
print(f"  Sleep cycle ran successfully: {ok}")

# ---------------------------------------------------------------
# 6. Timeline
# ---------------------------------------------------------------
section(6, "Memory timeline (procedural only)")

for e in model.timeline(memory_type="procedural", limit=5):
    ts = e["created_at"][:19]
    print(f"  [{ts}] {e['content']}")

# ---------------------------------------------------------------
# 7. Wrapper introspection
# ---------------------------------------------------------------
section(7, "Wrapper introspection")

print(f"  repr:           {model}")
print(f"  model type:     {type(model._model).__name__}")
print(f"  namespace:      {model.memory.namespace}")
print(f"  original attr:  model.invoke exists = {callable(getattr(model, 'invoke', None))}")

# ---------------------------------------------------------------
print(f"\n{'='*56}")
print("  Done. Entire NSN integration = 2 lines.")
print(f"{'='*56}\n")
