# NeuroSleepNet Architecture

## Cognitive Memory Operating System for SLMs & AI Agents

---

# Vision

NeuroSleepNet is a Cognitive Memory Operating System designed to amplify the intelligence of Small Language Models (SLMs) and AI agents.

Instead of acting as a passive memory database, NeuroSleepNet continuously:

- Observes
- Forms memories
- Builds relationships
- Consolidates knowledge
- Compresses context
- Generates reasoning packs
- Retrieves relevant cognition

The goal is not to store information.

The goal is to make small models behave as if they possess significantly larger context windows, richer long-term memory, and stronger situational awareness.

---

# Core Philosophy

Traditional Memory Systems:

```text
Store
↓
Search
↓
Retrieve
```

NeuroSleepNet:

```text
Observe
↓
Form Memory
↓
Build Knowledge
↓
Sleep
↓
Compress
↓
Generate Reasoning Context
↓
Retrieve
↓
Reason
```

---

# High Level Architecture

```text
                          Agent
                            │
                            ▼

                ┌──────────────────────┐
                │ NeuroSleepNet SDK    │
                └──────────┬───────────┘
                           │
                           ▼

              ┌──────────────────────────┐
              │ Cognitive Kernel         │
              └──────────┬───────────────┘

      ┌──────────────────┼──────────────────┐

      ▼                  ▼                  ▼

 Perception        Memory Engine     Retrieval Engine

      │                  │                  │

      ▼                  ▼                  ▼

 Knowledge Graph   Sleep Engine      Context Builder

      │                  │                  │

      └────────────┬─────┴─────┬────────────┘
                   ▼           ▼

          Reasoning Pack Generator

                   │

                   ▼

            Storage Layer
```

---

# Repository Structure

```text
neurosleepnet/

├── pyproject.toml
├── README.md
├── LICENSE

├── neurosleepnet/

│
├── sdk/
│   ├── memory.py
│   ├── client.py
│   ├── session.py
│   └── namespace.py
│
├── kernel/
│   ├── cognitive_kernel.py
│   ├── orchestrator.py
│   ├── scheduler.py
│   └── lifecycle.py
│
├── perception/
│   ├── ingestion.py
│   ├── extraction.py
│   ├── filters.py
│   ├── deduplication.py
│   ├── importance.py
│   └── classification.py
│
├── memory/
│   ├── working/
│   ├── episodic/
│   ├── semantic/
│   ├── procedural/
│   └── schemas.py
│
├── graph/
│   ├── graph_engine.py
│   ├── nodes.py
│   ├── edges.py
│   ├── entity_resolution.py
│   ├── relationship_extraction.py
│   └── traversal.py
│
├── sleep/
│   ├── consolidation.py
│   ├── synthesis.py
│   ├── promotion.py
│   ├── forgetting.py
│   ├── decay.py
│   └── conflict_resolution.py
│
├── cognition/
│   ├── context_builder.py
│   ├── reasoning_pack.py
│   ├── summarization.py
│   ├── profile_builder.py
│   └── memory_compression.py
│
├── retrieval/
│   ├── vector_search.py
│   ├── keyword_search.py
│   ├── graph_search.py
│   ├── hybrid_search.py
│   ├── reranker.py
│   └── retrieval_pipeline.py
│
├── trust/
│   ├── scoring.py
│   ├── provenance.py
│   ├── consistency.py
│   └── verification.py
│
├── explain/
│   ├── retrieval.py
│   ├── lineage.py
│   └── trust.py
│
├── storage/
│   ├── base.py
│   ├── sqlite.py
│   ├── postgres.py
│   ├── qdrant.py
│   ├── redis.py
│   └── duckdb.py
│
├── embeddings/
│   ├── local.py
│   ├── transformers.py
│   ├── ollama.py
│   └── openai.py
│
├── daemon/
│   ├── server.py
│   ├── api.py
│   ├── websocket.py
│   ├── scheduler.py
│   └── health.py
│
├── mcp/
│   ├── server.py
│   ├── tools.py
│   └── resources.py
│
├── adapters/
│   ├── langchain/
│   ├── llamaindex/
│   ├── crewai/
│   ├── autogen/
│   ├── openai_agents/
│   ├── cursor/
│   └── mcp/
│
├── audit/
│   ├── events.py
│   ├── snapshots.py
│   └── exports.py
│
├── cli/
│   ├── start.py
│   ├── sleep.py
│   ├── graph.py
│   ├── inspect.py
│   └── export.py
│
└── config/
    ├── defaults.yaml
    └── settings.py
```

---

# Memory Hierarchy

## Working Memory

Current session.

TTL:

```text
Minutes → Hours
```

Examples:

- Active tasks
- Current conversation
- Temporary tool outputs

---

## Episodic Memory

Experiences.

Examples:

```text
User deployed a FastAPI application.
```

---

## Semantic Memory

Generalized knowledge.

Examples:

```text
User prefers TypeScript.
```

---

## Procedural Memory

Workflows and habits.

Examples:

```text
User deploys using Docker Compose.
```

---

# Knowledge Graph

Purpose:

Transform isolated memories into connected knowledge.

Example:

```text
User
 ├─ Uses TypeScript
 ├─ Uses React
 ├─ Uses Next.js
 └─ Frontend Developer
```

Node Types:

- Person
- Project
- Goal
- Skill
- Preference
- Tool
- Company
- Concept

Edge Types:

- Uses
- Likes
- Owns
- WorksOn
- DependsOn
- RelatedTo
- LearnedFrom

---

# Sleep Engine

The most important subsystem.

Purpose:

Convert experiences into knowledge.

Pipeline:

```text
Memories
   ↓
Deduplicate
   ↓
Cluster
   ↓
Merge
   ↓
Synthesize
   ↓
Promote
   ↓
Decay
   ↓
Forget
```

Output:

- Reduced noise
- Better retrieval
- Knowledge formation

---

# Context Compression Engine

Purpose:

Compress large memory collections into compact cognitive state.

Example:

Input:

```text
100,000 memories
```

Output:

```text
User Profile
Current Goals
Long-Term Goals
Recent Decisions
Preferences
Project State
Known Constraints
```

Target:

```text
300–1000 tokens
```

---

# Reasoning Pack Generator

Purpose:

Generate optimized context for SLMs.

Instead of:

```text
Raw Memory Retrieval
```

Generate:

```text
User Profile
Task Context
Relevant Knowledge
Relevant Experiences
Constraints
Goals
```

Output:

```json
{
  "profile": {},
  "goals": [],
  "knowledge": [],
  "constraints": [],
  "memories": []
}
```

This is what gets injected into the model.

---

# Retrieval Pipeline

```text
Query
 ↓
Vector Search
 ↓
Keyword Search
 ↓
Graph Search
 ↓
Hybrid Merge
 ↓
Trust Filtering
 ↓
Importance Filtering
 ↓
Recency Boost
 ↓
Reranking
 ↓
Results
```

---

# Trust Engine

Every memory receives:

```text
0.0 → 1.0
```

Trust Formula:

```text
Trust Score =
(Source Reliability × 0.4)
+
(Consistency × 0.3)
+
(Recency × 0.2)
+
(Verification × 0.1)
```

Used by:

- Retrieval
- Sleep
- Consolidation
- Conflict Resolution

---

# State Management

## Hot State

Lives In:

```text
Memory Cache
Redis
```

Contains:

- Active conversations
- Session context
- Current tasks

---

## Warm State

Lives In:

```text
SQLite
Postgres
DuckDB
```

Contains:

- Episodic memories
- Semantic memories
- Profiles

---

## Cold State

Lives In:

```text
Archive Storage
```

Contains:

- Historical memories
- Snapshots
- Exports

---

# Deployment Modes

## Embedded Mode

```python
memory = Memory()
```

Backend:

```text
SQLite
+
Local Embeddings
```

No server.

---

## Daemon Mode

```bash
neurosleepnet start
```

Provides:

- Shared memory
- REST API
- MCP
- Multi-agent support

---

## Cluster Mode

```yaml
backend: postgres
vector_store: qdrant
cache: redis
```

For enterprise deployments.

---

# Public API

```python
memory.store()

memory.retrieve()

memory.reasoning_pack()

memory.graph()

memory.sleep()

memory.audit()

memory.explain()

memory.forget()

memory.export()
```

---

# Data Flow

```text
Agent Event
      ↓
Perception Layer
      ↓
Memory Formation
      ↓
Knowledge Graph
      ↓
Storage
      ↓
Sleep Cycle
      ↓
Knowledge Synthesis
      ↓
Context Compression
      ↓
Reasoning Pack
      ↓
SLM Prompt
```

---

# Success Metric

NeuroSleepNet succeeds when:

- A 7B–14B model behaves closer to a much larger model.
- Retrieval quality improves over time.
- Memory size grows slower than experiences.
- Context windows remain small.
- Reasoning packs become more useful over time.
- Agents require less prompt engineering.
- Long-term agent performance continuously improves.

```

```
