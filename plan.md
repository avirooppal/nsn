# NSN Architectural Rebuild Plan

Source of truth for rebuilding NSN. Content below is drawn only from the
original architectural vision document — nothing added beyond it.

---

## The Core Problem with Current NSN

NSN is a great prototype but it has three structural weaknesses that prevent
it from competing with the best memory systems:

1. **Single-dimensional retrieval** — one embedding model, two search modes.
   Real cognition has many retrieval pathways.
2. **Heuristic-based intelligence** — importance scoring, compression, and
   contradiction detection are all hand-tuned rules. They can't learn from
   the agent's actual behaviour.
3. **No temporal cognition** — the system treats a memory from 2 years ago
   identically to one from 2 minutes ago.

---

## Priority Order (build in this sequence)

| Priority | Change                           | Impact                      | Effort |
| -------- | -------------------------------- | --------------------------- | ------ |
| 1        | HNSW index (replace IndexFlatIP) | Scales to millions          | Low    |
| 2        | Ebbinghaus decay + temporal tags | Real temporal cognition     | Medium |
| 3        | Causal graph edges + DuckDB      | Actual reasoning capability | Medium |
| 4        | Belief versioning in REM         | No lossy deletion           | Low    |
| 5        | Generative NREM via local LLM    | Real synthesis              | High   |
| 6        | ML importance scorer             | Adapts to user              | Medium |
| 7        | Encryption at rest               | Production-safe             | Low    |
| 8        | Memory federation                | Multi-agent                 | High   |

The first four are low-to-medium effort and would immediately put NSN ahead
of every existing open-source memory library. Items 5–8 are what would make
it genuinely world-class.

---

## 1. Replace IndexFlatIP with a Tiered Index

**Current problem:** FAISS `IndexFlatIP` is O(n) per query. At 100k
memories, search takes seconds. At 1M memories, it's unusable.

**What to do:**

```
Working Memory   →  In-process OrderedDict (50 items, LRU)
Recent Buffer    →  FAISS IndexHNSW (fast ANN, O(log n))
Long-term Store  →  FAISS IndexIVFPQ (compressed, O(√n))
                    + full-text FTS5 in SQLite
```

`HNSW` gives approximate nearest-neighbour search in milliseconds at any
scale. `IVFPQ` compresses vectors 8–16× so a million memories fits in under
1GB of RAM.

Additionally, use **late interaction retrieval (ColBERT-style)**: instead
of comparing single sentence vectors, compare every token against every
token. This is dramatically more accurate for longer, nuanced memories.

---

## 2. Ebbinghaus Forgetting Curves (replace flat decay)

**Current problem:** `apply_decay()` multiplies importance by 0.85 per
sleep cycle for unused memories. This is arbitrary and linear. Human memory
follows an exponential decay with stability that grows on each successful
recall.

**What to do:**

```
Memory Strength = e^(-elapsed_days / stability)
stability       = initial_stability × 2.5^(recall_count)
```

This is the algorithm behind Anki and SM-2. Memories you access frequently
become nearly permanent. Memories you never touch fade asymptotically to a
minimum retention floor — they don't vanish, they just recede.

Add a **temporal tag extractor**: parse "yesterday", "last quarter", "in
2023" from memory content and convert to absolute timestamps. The graph
then has `BEFORE`/`AFTER`/`DURING` edges between memories, enabling actual
temporal reasoning.

---

## 3. Causal Knowledge Graph (replace flat entity-relation pairs)

**Current problem:** the graph stores `(Alice) -[IS]-> (Alpha)`. That is
not useful for reasoning. A reasoning engine needs causality.

**What to do:**

Replace the two flat SQLite tables with a proper graph schema that
supports:

| Edge type         | Example                             |
| ----------------- | ----------------------------------- |
| `CAUSED`          | deployment → outage                 |
| `CONTRADICTS`     | "server is fast" ↔ "server is slow" |
| `SUPERSEDES`      | updated belief → old belief         |
| `DEPENDS_ON`      | service B → service A               |
| `OCCURRED_BEFORE` | bug fix → release                   |
| `ATTRIBUTED_TO`   | decision → Alice                    |

Add **confidence weights** to every edge — not binary true/false but a
probability. New evidence updates the weight via Bayesian update rather
than replacing the edge entirely.

For scale, migrate the graph to **DuckDB** (embedded, column-store,
zero-dependency like SQLite but 10–100× faster for analytical graph
queries).

---

## 4. Uncertainty and Competing Beliefs (belief versioning)

**Current problem:** REM deletes the contradicted memory. This is lossy —
the old belief is gone forever, which means the system can't reason about
belief change over time.

**What to do:**

Keep contradicted memories but mark them as `SUPERSEDED`, linked to the
new belief:

```
state=ACTIVE    "The server runs on port 8080."      trust=0.95
      │
      └── SUPERSEDES
      │
state=ARCHIVED  "The server runs on port 3000."      trust=0.82
```

The system now has a **belief history** — it can answer "what did we think
before?" and "how confident are we in this belief given that it changed
once before?"

Every memory carries not just a `trust_score` but a full
`TrustDistribution`: mean and variance. High-variance memories are flagged
as uncertain and trigger lower confidence in downstream reasoning.

---

## 5. Generative NREM (replace extractive compression)

**Current problem:** `ContextCompressor` picks the best sentences from
episodic memories and concatenates them. This is extractive summarisation
— the output is always a subset of the inputs, never a synthesis.

**What to do:**

Run a small local LLM (Phi-3-mini, Gemma-2-2B, or Qwen-2.5-1.5B via
`llama-cpp-python`) during NREM consolidation to _generate_ the semantic
abstraction:

```python
prompt = f"""
The following {len(episodes)} events were observed today.
Synthesise them into one or two factual, generalised statements
that capture the core knowledge. Do not repeat specifics.

Events:
{formatted_episodes}

Synthesis:
"""
synthesis = local_llm(prompt)
```

The result is an actual semantic abstraction, not a copy-paste. This is
how human declarative memory works: you don't remember the exact words of
a conversation — you remember the meaning.

---

## 6. Learned Importance (replace keyword heuristics)

**Current problem:** `ImportanceScorer` is a hard-coded sum of keyword
bonuses. It has no knowledge of what this specific agent cares about.

**What to do:**

Train a lightweight importance predictor on the agent's own access log:

```
Features: memory_type, entity_count, word_count, source_trust,
          days_since_created, times_retrieved, led_to_action (bool)
Label:    importance (0–1, derived from retrieval frequency)
```

A small gradient-boosted tree (30KB model, `sklearn` GBM) can learn this
in milliseconds on every sleep cycle. Over time, the system learns that
this agent considers "deployment errors" more important than "meeting
notes" — automatically, without configuration.

Add **reinforcement from outcomes**: if a recalled memory directly
preceded a successful agent action (detected via the event hook system NSN
already has), that memory's importance gets a strong boost.

---

## 7. Privacy Layer

**What to do:**

- **Encryption at rest** — `pysqlcipher3` for SQLite (SQLCipher), AES-256.
  The entire database is encrypted with a user-supplied key. Zero access
  without the key.
- **Selective retention policies** — `memory.set_retention("Alice",
days=90)` — all memories about entity "Alice" auto-expire after 90 days.
- **Audit log** — every memory access is logged with timestamp, accessor
  namespace, and query that triggered it.
- **Derived memory tracking** — when NREM synthesises a new semantic
  memory from episodic sources, the link is preserved. Deleting a source
  memory marks all derived memories as `PROVENANCE_INCOMPLETE`.

---

## 8. Multi-Agent Memory Federation

**Current problem:** namespaces are isolated silos with no communication.

**What to do:**

Add a `MemoryFederation` protocol:

```python
# Agent A grants Agent B read access to its "deployments" namespace
federation = MemoryFederation()
federation.share(
    from_namespace="agent_a",
    to_namespace="agent_b",
    filter={"memory_type": "procedural"},  # only share procedures
    access="read"
)
```

During retrieval, an agent can optionally query federated namespaces:

```python
results = memory.search_hybrid(query, include_federated=True)
```

Federated results are clearly tagged with their origin namespace and carry
their original trust scores. The receiving agent can choose to adopt a
memory into its own namespace (with trust decay applied for the transfer).

---

## What the Full Stack Would Look Like

```
┌─────────────────────────────────────────────────────────┐
│                    NSN Wrapper (NSN)                    │
│   Auto-observe → Recall → Inject → Store → Sleep        │
├─────────────────────────────────────────────────────────┤
│                 Intelligence Orchestrator               │
│   Perception | Trust | Classifier | Importance (ML)     │
├──────────────┬──────────────────────┬───────────────────┤
│ Working Mem  │   Episodic Buffer    │   Long-term Store  │
│ (dict, LRU)  │  (HNSW + SQLite)    │ (IVFPQ + DuckDB)  │
├──────────────┴──────────────────────┴───────────────────┤
│                  Causal Knowledge Graph                 │
│     DuckDB graph schema + confidence-weighted edges     │
├─────────────────────────────────────────────────────────┤
│                 Sleep Engine (v2)                       │
│   NREM: LLM synthesis | REM: belief versioning | Decay  │
│   Ebbinghaus curves + spaced repetition scheduling      │
├─────────────────────────────────────────────────────────┤
│              Privacy & Federation Layer                 │
│   SQLCipher encryption | Retention policies | Audit log │
└─────────────────────────────────────────────────────────┘
```
