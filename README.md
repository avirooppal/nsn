# NeuroSleepNet

**Version:** 0.3.0

NeuroSleepNet is a Cognitive Memory Operating System for SLMs and AI agents. It provides long-term memory, knowledge formation, context compression, and offline memory consolidation — all running **100% locally** with zero cloud dependencies.

Wrap any existing LLM or agent with two lines of code and it gains persistent memory automatically.

---

## Installation

```bash
pip install -e .
python -m spacy download en_core_web_sm
```

Optional integrations:

```bash
pip install "neurosleepnet[api]"       # FastAPI REST server
pip install "neurosleepnet[langchain]" # LangChain adapter
pip install "neurosleepnet[all]"       # Everything
```

---

## Quickstart — Wrap any model (2 lines)

```python
import neurosleepnet

# Your existing LLM, pipeline, or callable
model = YourLLM()

# Wrap it — memory is now live
model = neurosleepnet.NSN(model, namespace="my_agent")

# Use it exactly like before — NSN injects memory automatically
response = model("What do we know about Alice?")
```

That is the entire integration. NSN auto-observes inputs, injects recalled context before each call, and stores responses back into long-term memory.

---

## Integrations

### OpenAI / OpenAI-compatible (GPT-4o, Claude, Gemini via API)

```python
import neurosleepnet
from openai import OpenAI

client = OpenAI()
client = neurosleepnet.NSN(client, namespace="my_agent")

# Memory is injected automatically as a system message
response = client.chat.completions.create(
    model="gpt-4o",
    messages=[{"role": "user", "content": "What did Alice say?"}]
)
```

### LangChain

```python
import neurosleepnet
from langchain_openai import ChatOpenAI

llm = ChatOpenAI()
llm = neurosleepnet.NSN(llm, namespace="my_agent")

response = llm.invoke("Summarize what we know about the project.")
```

Or use the native LangChain history adapter:

```python
from neurosleepnet.integrations.langchain import NeurosleepNetHistory
from langchain_core.runnables.history import RunnableWithMessageHistory

chain = RunnableWithMessageHistory(
    runnable=your_chain,
    get_session_history=lambda sid: NeurosleepNetHistory(namespace=sid),
)
```

### AutoGen / CrewAI / Custom Agents

```python
from neurosleepnet.integrations.tool import MemoryTool

tool = MemoryTool(namespace="my_agent")
tool.remember("Alice leads the engineering team.")
results = tool.recall("Who leads engineering?")
tool.sleep()  # Run NREM/REM consolidation
```

Register as an AutoGen function tool:

```python
schema = tool.as_autogen_tool()
```

### FastAPI REST Server

```bash
uvicorn neurosleepnet.integrations.api:app --host 0.0.0.0 --port 8000
```

| Endpoint | Method | Description |
|---|---|---|
| `/observe` | POST | Store a memory |
| `/observe/batch` | POST | Batch ingest |
| `/search` | GET | Hybrid search |
| `/surface` | GET | Proactive surfacing |
| `/graph/{entity}` | GET | Knowledge subgraph |
| `/timeline` | GET | Chronological memories |
| `/reasoning-pack` | GET | SLM context pack |
| `/sleep` | POST | Trigger consolidation |
| `/forget/{id}` | DELETE | Remove a memory |

---

## Core Architecture

### 1. Perception

Filters incoming data before it enters memory.

- **Duplicate Detector:** Rejects identical or near-identical observations (>= 95% semantic similarity).
- **Importance Scorer:** Scores input significance via keyword signals, entity density, and length.
- **Memory Classifier:** Categorizes data into `EPISODIC`, `SEMANTIC`, or `PROCEDURAL` using embedding-based prototype matching.

### 2. Trust Engine

Maintains data reliability over time.

- **Source Scorer:** Evaluates trust based on data origin (`system=1.0`, `user=0.9`, `llm=0.8`, `agent=0.75`, `web=0.4`).
- **Recency Scorer:** Prioritizes fresh data.
- **Consistency Scorer:** Detects contradictions with existing memories using negation analysis and antonym pairs.

### 3. Knowledge Graph

Transforms flat memory logs into a connected semantic web.

- **Entity Extractor:** Identifies entities via spaCy NER with heuristic fallback.
- **Relationship Extractor:** Detects relationships between entity pairs.
- **GraphBuilder:** Links graph nodes back to source memories in SQLite.

### 4. Hybrid Retrieval (Graph-RRF)

Combines three search modalities via Reciprocal Rank Fusion:

- Dense semantic vector search (FAISS `IndexFlatIP`)
- Keyword search (SQLite FTS)
- Knowledge graph traversal

### 5. Sleep Engine

Offline memory consolidation mimicking the human sleep cycle:

- **NREM:** Synthesizes unconsolidated episodic memories into permanent semantic knowledge using `ContextCompressor`.
- **REM:** Detects and resolves contradictions between memories, pruning the lower-trust copy.
- **Decay:** Adjusts importance scores based on access frequency; promotes frequently-accessed episodic memories to semantic.

---

## Advanced Usage

### Manual memory control

```python
import neurosleepnet

model = neurosleepnet.NSN(your_model, namespace="my_agent")

# Manual store
model.remember("The API rate limit is 1000 requests per hour.", source="system")

# Manual search
hits = model.recall("What are the API limits?", limit=5)

# Trigger offline consolidation
model.sleep()

# Chronological memory log
model.timeline(memory_type="procedural", limit=10)

# Direct access to the full Memory API
model.memory.get_entity_subgraph("Alice", depth=2)
model.memory.reasoning_pack("Project Alpha")
model.memory.forget("some-memory-id")
```

### NSN wrapper options

```python
model = neurosleepnet.NSN(
    model,
    namespace="my_agent",        # Namespace for multi-agent isolation
    db_path="agent.db",          # Custom SQLite path
    recall_limit=5,               # Memories injected per call
    auto_observe_inputs=True,     # Store user inputs automatically
    auto_observe_outputs=True,    # Store model outputs automatically
)
```

### Low-level Memory API

```python
from neurosleepnet import Memory

m = Memory(namespace="my_agent")
m.observe("Alice leads the project.", source="system")
m.search_hybrid("Who leads?", limit=5)
m.trigger_sleep()
m.timeline(memory_type="episodic")
m.get_entity_subgraph("Alice", depth=1)
m.reasoning_pack("Project Alpha")
```

### Async support

```python
from neurosleepnet import AsyncMemory

m = AsyncMemory(namespace="async_agent")
await m.observe("Alice is the lead engineer.")
results = await m.search_hybrid("Who leads?")
```

---

## Getting Started

### Prerequisites

- Python 3.9+
- `sentence-transformers` (auto-downloads `all-MiniLM-L6-v2` on first use)
- `faiss-cpu`
- `spacy` + `en_core_web_sm`

### Installation

```bash
git clone https://github.com/avirooppal/nsn.git
cd nsn
pip install -e .
python -m spacy download en_core_web_sm
```

### Run the demo

```bash
python demo.py
```

---

## Testing

```bash
python test_e2e.py   # Full acceptance suite
python smoke_test.py # Integration smoke tests
```

---

## License

MIT
