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

## Empirical Benchmark & Research Evaluation

A research-grade evaluation (`seed=42`, framework v2.0) comparing **NeuroSleepNet (NSN)** against 6 alternative memory architectures across three benchmark categories:

1. **Knowledge Update**: Temporal state tracking across sequential fact updates (e.g., Day 1 → Day 10 → Day 20 port migrations).
2. **Contradiction Resolution**: Source trust weighting and conflict resolution (`source="system"` vs. `source="user"`).
3. **Multi-Hop Traversal**: Relational reasoning across multi-hop entity chains ($A \to B \to C$).

### Systems Under Evaluation

| # | System | Category | Architecture |
|:---|:---|:---|:---|
| 1 | **Vanilla LLM** | No Memory | Linear history scan, stateless fallback |
| 2 | **Rolling Window LLM** | Naive LLM Memory | Sliding 10-item buffer, token-overlap matching |
| 3 | **LLM + BM25** | Keyword Memory | SQLite FTS5 phrase-matching |
| 4 | **LLM + Dense RAG** | Semantic Memory | FAISS cosine vector similarity (`all-MiniLM-L6-v2`) |
| 5 | **LLM + Hybrid RAG** | Hybrid Memory | FAISS + FTS5 + Reciprocal Rank Fusion (RRF) |
| 6 | **LLM + RAG Memory** | Dense + Generator | FAISS vector retrieval + generation context |
| 7 | **NeuroSleepNet (NSN)** | **Biologically-Inspired** | **FAISS + FTS5 + Entity Graph + RRF + Reranker + Sleep Consolidation** |

---

### Head-to-Head Benchmark Results

#### 1. Knowledge Update Benchmark (Temporal State Tracking)

| System | Recall@5 | Hit@5 | MRR | nDCG@5 | Exact Match | P95 Latency (ms) |
|:---|:---:|:---:|:---:|:---:|:---:|:---:|
| **Vanilla LLM** | N/A | N/A | N/A | N/A | 2.50% | 3.8 ms |
| **Rolling Window LLM** | 17.50% | 17.50% | 0.1250 | 0.1508 | 10.00% | 2.1 ms |
| **LLM + BM25** | 0.00% | 0.00% | 0.0000 | 0.0000 | 0.00% | 13.8 ms |
| **LLM + Dense RAG** | 75.00% | 75.00% | 0.4250 | 0.5044 | 25.00% | 888.0 ms |
| **LLM + Hybrid RAG** | 75.00% | 75.00% | 0.4250 | 0.5044 | 25.00% | 882.7 ms |
| **LLM + RAG Memory** | 75.00% | 75.00% | 0.4250 | 0.5044 | 17.50% | 890.0 ms |
| **NeuroSleepNet (NSN)** | **85.71%** | **85.71%** | **0.4929** | **0.5828** | **28.57%** | **853.6 ms** |

* **NSN Advantage:** **+10.71 pp** Recall@5 over Dense/Hybrid RAG, **+16.0%** MRR improvement, and **50% reduction in total retrieval failures** (5 vs. 10).

#### 2. Contradiction Resolution Benchmark (Source Trust & Conflict)

| System | Recall@5 | Hit@5 | MRR | nDCG@5 | P95 Latency (ms) |
|:---|:---:|:---:|:---:|:---:|:---:|
| **Vanilla LLM** | N/A | N/A | N/A | N/A | 0.6 ms |
| **Rolling Window LLM** | 25.00% | 25.00% | 0.1125 | 0.1477 | 4.3 ms |
| **LLM + BM25** | 0.00% | 0.00% | 0.0000 | 0.0000 | 20.0 ms |
| **LLM + Dense RAG** | 0.00% | 0.00% | 0.0000 | 0.0000 | 782.1 ms |
| **LLM + Hybrid RAG** | 0.00% | 0.00% | 0.0000 | 0.0000 | 738.4 ms |
| **LLM + RAG Memory** | 0.00% | 0.00% | 0.0000 | 0.0000 | 2288.6 ms |
| **NeuroSleepNet (NSN)** | **100.00%** | **100.00%** | **0.5000** | **0.6309** | **788.4 ms** |

* **Key Finding:** Standard Dense and Hybrid RAG suffer from *semantic collision* (0.00% Recall@5) because cosine distance cannot differentiate verified facts from unverified conflicting claims. NSN achieves **100.00% Recall@5** via its `TrustManager` and REM sleep consolidation.

#### 3. Multi-Hop Relational Traversal Benchmark

| System | Recall@5 | Hit@5 | MRR | nDCG@5 | Exact Match |
|:---|:---:|:---:|:---:|:---:|:---:|
| **Rolling Window LLM** | 4.27% | 4.27% | 0.1000 | 0.0552 | 20.00% |
| **LLM + BM25** | 0.00% | 0.00% | 0.0000 | 0.0000 | 0.00% |
| **LLM + Dense RAG** | 21.33% | 21.33% | 0.2283 | 0.1628 | 20.00% |
| **LLM + Hybrid RAG** | 21.33% | 21.33% | 0.2283 | 0.1628 | 20.00% |
| **NeuroSleepNet (NSN)** | **42.67%** | **42.67%** | **1.0000** | **0.5521** | **20.00%** |

* **Key Finding:** NSN's integrated Knowledge Graph **doubles multi-hop recall** (42.67% vs. 21.33%) and achieves **MRR = 1.0000** (when gold relational memories are retrieved, they are placed at Rank 1).

---

### Summary of Final Research Verdict

| Benchmark Dimension | NSN Recall@5 | Best Competitor | NSN Advantage |
|:---|:---:|:---:|:---:|
| **Knowledge Update** | **85.71%** | 75.00% (*Dense/Hybrid RAG*) | **+10.71 pp** |
| **Contradiction Resolution** | **100.00%** | 25.00% (*Rolling Window*) | **+75.00 pp** |
| **Multi-Hop Traversal** | **42.67%** | 21.33% (*Dense/Hybrid RAG*) | **+21.33 pp** |

> Complete research paper documentation, mathematical formulations, 2×2 failure matrices, and ablation breakdown are available in [`benchmarks/results/RESEARCH_PAPER_RESULTS.md`](benchmarks/results/RESEARCH_PAPER_RESULTS.md).

---

### Reproduce the Benchmark Suite

```bash
# Ensure dependencies are installed
pip install -e .

# Run the full 7-system head-to-head comparison
$env:PYTHONPATH = "."
$env:PYTHONIOENCODING = "utf-8"
$env:HF_HUB_OFFLINE = "1"
python -m benchmarks.run_head_to_head --samples 20 --chains 10 --seed 42

# Run unit tests verifying metric mathematical integrity
python -m pytest benchmarks/tests/ -v
```

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
