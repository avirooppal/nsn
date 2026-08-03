# NeuroSleepNet

**Cognitive Memory Operating System for SLMs and AI Agents**

NeuroSleepNet gives any language model or AI agent persistent, long-term memory with two lines of code. It runs entirely locally — no cloud services, no external APIs, no data leaving your machine.

```python
import neurosleepnet

model = neurosleepnet.NSN(your_llm, namespace="my_agent")
```

That is the entire integration. From that point, every call your model makes automatically recalls relevant past memories and stores the new interaction — continuously building a self-organising knowledge base in the background.

---

## How It Works

When your model processes any input, NSN transparently:

1. **Recalls** relevant memories using hybrid semantic + keyword + graph search
2. **Injects** that context into the model's input before it runs
3. **Stores** the input and output back into long-term memory
4. **Classifies** each memory as Episodic, Semantic, or Procedural automatically
5. **Consolidates** memories offline during sleep cycles (NREM synthesis, REM contradiction resolution, importance decay)

---

## Installation

```bash
git clone https://github.com/avirooppal/nsn.git
cd nsn
pip install -e .
python -m spacy download en_core_web_sm
```

Optional extras:

```bash
pip install "neurosleepnet[api]"        # FastAPI REST microservice
pip install "neurosleepnet[langchain]"  # LangChain adapter
pip install "neurosleepnet[all]"        # All integrations
```

---

## Integrations

### Any callable model

```python
import neurosleepnet

model = neurosleepnet.NSN(your_model, namespace="my_agent")

# Use exactly as before — memory is injected and stored automatically
response = model("Summarise what we know about the project.")
```

### OpenAI / OpenAI-compatible APIs

Works with any API that follows the OpenAI chat completions format (GPT-4o, Claude via proxy, Gemini, Groq, Ollama, Together.ai, etc.).

```python
import neurosleepnet
from openai import OpenAI

client = OpenAI()
client = neurosleepnet.NSN(client, namespace="my_agent")

# Recalled memory is automatically injected as a system message
response = client.chat.completions.create(
    model="gpt-4o",
    messages=[{"role": "user", "content": "What do you know about Alice?"}]
)
```

### LangChain

```python
import neurosleepnet
from langchain_openai import ChatOpenAI

llm = ChatOpenAI()
llm = neurosleepnet.NSN(llm, namespace="my_agent")

response = llm.invoke("Summarise the project status.")
```

Or use the native session history adapter with `RunnableWithMessageHistory`:

```python
from neurosleepnet.integrations.langchain import NeurosleepNetHistory
from langchain_core.runnables.history import RunnableWithMessageHistory

chain_with_memory = RunnableWithMessageHistory(
    runnable=your_chain,
    get_session_history=lambda session_id: NeurosleepNetHistory(namespace=session_id),
)
```

### AutoGen / CrewAI / Custom Agents

```python
from neurosleepnet.integrations.tool import MemoryTool

tool = MemoryTool(namespace="my_agent")

tool.remember("The API rate limit is 1000 requests per hour.", source="system")
results = tool.recall("What are the rate limits?")
tool.sleep()  # Run NREM/REM consolidation
```

Register directly as an AutoGen/OpenAI function tool:

```python
schema = tool.as_autogen_tool()  # Returns OpenAI function calling schema
```

Dispatch-style invocation for frameworks that call tools as callables:

```python
tool("remember", text="Deployment uses docker-compose up -d.")
tool("recall", query="How do I deploy?")
```

### FastAPI REST Microservice

Expose NSN as an HTTP service that any language or service can call.

```bash
uvicorn neurosleepnet.integrations.api:app --host 0.0.0.0 --port 8000
```

Or mount in an existing FastAPI app:

```python
from neurosleepnet.integrations.api import create_app

app = create_app(namespace="production_agent", db_path="/data/agent.db")
```

| Method | Endpoint | Description |
|--------|----------|-------------|
| POST | `/observe` | Store a memory |
| POST | `/observe/batch` | Batch ingest |
| GET | `/search` | Hybrid search (semantic + keyword + graph) |
| GET | `/surface` | Proactive memory surfacing |
| GET | `/graph/{entity}` | Knowledge graph subgraph |
| GET | `/timeline` | Chronological memory log |
| GET | `/reasoning-pack` | Structured SLM context pack |
| POST | `/sleep` | Trigger NREM/REM consolidation |
| DELETE | `/forget/{id}` | Remove a memory |

---

## Configuration

```python
model = neurosleepnet.NSN(
    model,
    namespace="my_agent",           # Namespace for multi-agent data isolation
    db_path="agent.db",             # SQLite database path
    recall_limit=5,                  # Memories injected per call
    auto_observe_inputs=True,        # Automatically store user inputs
    auto_observe_outputs=True,       # Automatically store model outputs
)
```

Environment variables (override config/defaults.yaml):

| Variable | Default | Description |
|----------|---------|-------------|
| `NSN_NAMESPACE` | `default` | Memory namespace |
| `NSN_EMBEDDING_MODEL` | `all-MiniLM-L6-v2` | Sentence transformer model |
| `NSN_DUPLICATE_THRESHOLD` | `0.95` | Cosine similarity threshold for deduplication |

---

## Manual Memory Control

Access the full Memory API directly through the wrapper:

```python
model = neurosleepnet.NSN(your_model, namespace="my_agent")

# Store
model.remember("The server runs on port 8080.", source="system")

# Search
hits = model.recall("What port does the server use?", limit=5)

# Proactive surfacing
relevant = model.memory.surface_relevant("I need to connect to the server.")

# Chronological log
model.timeline(memory_type="procedural", limit=20)

# Knowledge graph
model.memory.get_entity_subgraph("Alice", depth=2)

# Reasoning pack for SLM injection
model.memory.reasoning_pack("Project Alpha")

# Remove a memory
model.memory.forget("memory-id")
model.memory.forget_entity("Alice")  # Remove all memories linked to an entity

# Offline consolidation
model.sleep()
```

---

## Low-Level API

For direct use without the wrapper:

```python
from neurosleepnet import Memory, AsyncMemory

# Synchronous
m = Memory(namespace="my_agent")
m.observe("Alice leads the engineering team.", source="system")
m.search_hybrid("Who leads engineering?", limit=5)
m.trigger_sleep()

# Async
m = AsyncMemory(namespace="my_agent")
await m.observe("Alice leads the engineering team.")
results = await m.search_hybrid("Who leads engineering?")
```

---

## Architecture

```
observe(content)
      │
      ├─ DuplicateDetector   (cosine similarity ≥ 0.95 → reject)
      ├─ ImportanceScorer    (keyword + entity density + length heuristics)
      ├─ MemoryClassifier    (prototype embedding → episodic / semantic / procedural)
      ├─ TrustEngine         (source × recency × consistency → trust score)
      ├─ SQLiteAdapter       (persist with namespace, type, WAL mode)
      ├─ TieredVectorStore   (HNSW + IVFPQ, MaxSim ColBERT scoring, disk-persisted)
      └─ GraphBuilder        (spaCy NER → entity nodes + relationship edges)

search_hybrid(query)
      │
      ├─ Semantic search     (FAISS dense retrieval)
      ├─ Keyword search      (SQLite FTS)
      ├─ Graph search        (entity traversal → linked memories)
      └─ RRF fusion          (weighted Reciprocal Rank Fusion)

trigger_sleep()
      │
      ├─ NREM consolidation  (episodic → compressed semantic via ContextCompressor)
      ├─ REM resolution      (contradiction detection → prune lower-trust memory)
      └─ Decay               (importance ×0.85 if unused, +0.02×access_count if used)
```

### Memory Types

| Type | Description | Example |
|------|-------------|---------|
| `EPISODIC` | Events and experiences tied to a time or context | "Bob deployed the fix at 3pm." |
| `SEMANTIC` | General facts and knowledge | "NeuroSleepNet uses FAISS for vector search." |
| `PROCEDURAL` | Step-by-step instructions and workflows | "To deploy: run docker-compose up -d." |

### Trust Scoring

| Source | Trust Score |
|--------|-------------|
| `system` | 1.0 |
| `user` / `user_input` | 0.9 |
| `llm` / `gpt` / `claude` | 0.8 |
| `agent` / `agent_sensor` / `tool` | 0.75 |
| `web` / `web_scrape` | 0.4 |

---

## Running the Demo

```bash
python demo.py
```

The demo shows the complete NSN workflow: wrapping a model in two lines, ingesting facts, calling the model with auto-injected memory context, hybrid search, reasoning pack generation, sleep consolidation, and the timeline API.

---

## Testing

```bash
python test_e2e.py    # Full end-to-end acceptance suite
python smoke_test.py  # Integration smoke tests (MemoryTool + OpenAI adapter)
```

---

## Requirements

- Python 3.9+
- `sentence-transformers >= 2.2.0` (downloads `all-MiniLM-L6-v2` automatically on first use)
- `faiss-cpu >= 1.7.4`
- `spacy >= 3.7.0` + `en_core_web_sm`
- `pyyaml >= 6.0`
- `numpy >= 1.24.0`

---

## License

MIT
