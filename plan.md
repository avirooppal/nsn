# NeuroSleepNet — Implementation Plan v2.0
## Ground Truth Build Blueprint

**Repo Root:** `c:\Users\aviroop\Desktop\nsn\`  
**Package:** `neurosleepnet/`  
**Goal:** Upgrade NeuroSleepNet from a disconnected skeleton into a fully wired, State-of-the-Art cognitive memory OS for SLMs and AI agents. The user-facing API must remain: `from neurosleepnet import Memory; m = Memory(); m.observe("...")` — all intelligence fires automatically.

---

## Current State: What is Broken

The `sdk/memory.py` `Memory` class has `store()`, `get()`, `list()`, `search()`, `search_keyword()`, `search_hybrid()`. That is ALL it does. Every other subsystem is dead code:

- `perception/` — imported nowhere, never called
- `trust/` — imported nowhere, never called
- `graph/` — imported nowhere, never called
- `sleep/engine.py` — standalone class, not connected to Memory
- `compression/` — standalone classes, not connected to Memory

Additionally:
- `storage/local_vector.py` loads ALL embeddings from SQLite into RAM on every search (O(n) — breaks at scale)
- `perception/classifier.py` uses keyword matching (`'yesterday'`, `'how to'`) — extremely brittle
- `graph/extractor.py` detects entities by capital letters only — misses most real entities
- `trust/consistency.py` detects contradictions via negation words (`'not'`, `'never'`) — misses semantic antonyms
- `sleep/engine.py` "consolidates" memories by joining them with ` | ` — not knowledge synthesis
- RRF constant `k=60` is hardcoded, fusion weights are equal despite semantic >> keyword in performance
- `Memory()` init takes 8+ seconds (model loads synchronously on import)
- No namespace isolation — all agents share one flat table
- No access tracking — memories never evolve in importance

---

## Build Order (Strict — Do Not Reorder)

```
Phase A: Storage & Schema Foundation  (Steps 1–4)
Phase B: Intelligence Component Upgrades  (Steps 5–9)  [depends on A]
Phase C: SDK Wiring — Most Critical  (Steps 10–11)  [depends on A+B]
Phase D: Retrieval Upgrades  (Steps 12–13)  [depends on C]
Phase E: Sleep Engine Upgrades  (Step 14)  [depends on C]
Phase F: Developer Experience  (Steps 15–19)  [depends on C]
Phase G: Advanced Features  (Steps 20–23)  [depends on C+D]
```

---

## PHASE A — Storage & Schema Foundation

### Step 1 — Extend SQLite Schema

**File:** `neurosleepnet/storage/sqlite.py`

**Changes:**

1. Add 4 new columns to the `CREATE TABLE IF NOT EXISTS memories` SQL statement:
   ```sql
   namespace TEXT DEFAULT 'default',
   memory_type TEXT DEFAULT 'semantic',
   access_count INTEGER DEFAULT 0,
   last_accessed_at TEXT
   ```

2. Add 4 new `ALTER TABLE` guards (same try/except pattern already in the file) for backwards compatibility with existing databases.

3. Add these indexes at the end of `_initialize_db`:
   ```sql
   CREATE INDEX IF NOT EXISTS idx_memories_namespace ON memories(namespace);
   CREATE INDEX IF NOT EXISTS idx_memories_type ON memories(memory_type);
   ```

4. Enable WAL mode at the end of `_initialize_db`:
   ```python
   cursor.execute("PRAGMA journal_mode=WAL")
   cursor.execute("PRAGMA synchronous=NORMAL")
   ```

5. Update the `store()` method signature to accept new fields:
   ```python
   def store(self, memory_id, content, created_at, metadata="{}", importance=0.0,
             trust_score=0.5, embedding="[]", namespace="default", memory_type="semantic"):
   ```
   Update the INSERT statement to include the new columns.

6. Update `get()` and `list()` SELECT statements to also fetch `namespace`, `memory_type`, `access_count`, `last_accessed_at` and include them in returned dicts.

7. Add new `increment_access(memory_id: str)` method:
   ```python
   UPDATE memories SET access_count = access_count + 1, last_accessed_at = <utcnow> WHERE id = ?
   ```

8. Add new `list_namespace(namespace: str)` method that adds `WHERE namespace = ?` to the list query.

9. Update `search_keyword()` to accept optional `namespace` param and add `AND namespace = ?` when provided.

**Acceptance:** Old databases load via ALTER TABLE guards. New fields appear in all dict returns.

---

### Step 2 — Update MemoryRecord Dataclass

**File:** `neurosleepnet/memory/schemas.py`

Add 4 new fields to `MemoryRecord`:
```python
namespace: str = "default"
memory_type: str = "semantic"
access_count: int = 0
last_accessed_at: str = None
```

Fix `from_dict` to be resilient to missing keys (old data):
```python
@classmethod
def from_dict(cls, data: dict):
    from dataclasses import fields
    valid_keys = {f.name for f in fields(cls)}
    filtered = {k: v for k, v in data.items() if k in valid_keys}
    return cls(**filtered)
```

**Acceptance:** `MemoryRecord(content="test")` has `namespace="default"`, `memory_type="semantic"`, `access_count=0`.

---

### Step 3 — Replace LocalVectorStore with FAISS-Based Indexed Store

**File:** `neurosleepnet/storage/local_vector.py` — FULL REWRITE

**New dependency:** Add `faiss-cpu>=1.7.4` to `pyproject.toml` dependencies list.

**New class `FAISSVectorStore`** (replaces `LocalVectorStore`, inherits `VectorStore`):

- `__init__(self, storage, index_path="neurosleepnet.faiss", ids_path="neurosleepnet.faiss_ids")`
  - `self.dimension = 384` (all-MiniLM-L6-v2 output size)
  - Try loading index from disk; if not found, call `_build_from_storage()` then `_persist()`
  - Use `faiss.IndexFlatIP` (inner product = cosine sim for L2-normalized vectors)
  - Maintain `self._id_map: list[str]` to map FAISS integer indices → memory UUIDs

- `_build_from_storage()`: Iterate all records with embeddings from `storage.list()`, L2-normalize each, add to FAISS index, append id to `_id_map`

- `_persist()`: `faiss.write_index()` to disk; write `_id_map` as JSON to ids_path

- `add(self, memory_id: str, embedding: List[float])`: L2-normalize, `index.add()`, append to `_id_map`, call `_persist()`

- `search(self, query_embedding, limit)`: L2-normalize query, call `index.search(q, k)`, map result indices via `_id_map` to ids, fetch records via `storage.get()`, attach score, return list

**Acceptance:** After 100 stores, search returns in <5ms. After process restart, index loads from disk with correct results.

---

### Step 4 — Lazy Model Loading in LocalEmbeddingProvider

**File:** `neurosleepnet/embeddings/local.py` — REWRITE

- Do NOT import or load `SentenceTransformer` in `__init__`
- Add module-level `_MODEL_CACHE = {}` dict
- Add `@property model` that loads `SentenceTransformer(self.model_name)` on first access and caches in `_MODEL_CACHE` by model name
- All `embed()` and `embed_batch()` calls access via `self.model`

**Acceptance:** `Memory()` completes in <0.5s. First `observe()` triggers the actual model load. Two `Memory()` instances in same process share one model object.

---

## PHASE B — Intelligence Component Upgrades

### Step 5 — Embedding-Based Zero-Shot Memory Classifier

**File:** `neurosleepnet/perception/classifier.py` — FULL REWRITE

**Strategy:** Instead of keyword matching, compute cosine similarity between the input embedding and pre-computed "prototype embeddings" for each memory type.

Define 3 prototype sentence lists (3 sentences per type):
```python
PROTOTYPES = {
    "episodic": [
        "This is something that happened to me, a personal event or experience.",
        "The user performed an action or something occurred at a specific time.",
        "This is a log of what happened during this session.",
    ],
    "semantic": [
        "This is a general fact, piece of knowledge, or objective truth about the world.",
        "This is a definition or established concept that is universally true.",
        "This is factual information extracted from a document or knowledge base.",
    ],
    "procedural": [
        "This is a step-by-step guide, instruction, or workflow on how to do something.",
        "This describes a repeatable process, habit, or procedure.",
        "Follow these steps in order to complete the task.",
    ],
}
```

`MemoryClassifier.__init__(self, embedder=None)`:
- Store embedder reference
- `_prototype_embeddings = None` (lazy build)

`_build_prototypes()`:
- For each type, call `embedder.embed_batch(sentences)`, take mean vector, L2-normalize

`classify(observation) -> str`:
- If no embedder, call `_keyword_fallback()`
- If prototypes not built, build them
- Embed input, L2-normalize, compute dot product against each prototype
- Return type with highest score

Keep `_keyword_fallback()` with the original keyword logic as a fallback when no embedder.

**Acceptance:** With embedder: "I deployed the system this morning" → `episodic`. "Water is H2O" → `semantic`. "To deploy: run docker compose up" → `procedural`.

---

### Step 6 — Upgraded ImportanceScorer

**File:** `neurosleepnet/perception/importance.py` — REWRITE

Keep existing critical/goal keyword signals. Add:
- **Information density:** `unique_words / total_words * 0.1`
- **Named entity density:** count capitalized multi-word sequences via regex, add `min(count * 0.05, 0.15)`
- **Length tiers:** content >200 chars adds 0.15, >50 chars adds 0.08

Return `round(min(1.0, base_score), 4)`.

**Acceptance:** Error messages > 0.6. Short generic strings < 0.25. Multi-entity content scores higher.

---

### Step 7 — spaCy NER Entity Extractor

**File:** `neurosleepnet/graph/extractor.py` — REWRITE

**New dependency:** Add `spacy>=3.7.0` to `pyproject.toml`. Model `en_core_web_sm` installed separately via `python -m spacy download en_core_web_sm`.

**Strategy:** Try loading spaCy at module level; use it if available, fall back to original capitalization heuristic if not.

```python
_SPACY_NLP = None
try:
    import spacy
    _SPACY_NLP = spacy.load("en_core_web_sm")
except (ImportError, OSError):
    pass
```

Map spaCy label types to graph node labels:
```python
SPACY_LABEL_MAP = {
    "PERSON": "Person", "ORG": "Organization", "GPE": "Location",
    "LOC": "Location", "PRODUCT": "Product", "WORK_OF_ART": "Concept",
    "EVENT": "Event", "NORP": "Group",
}
```

`EntityExtractor.extract(text)` now returns `list[dict]` where each dict has `{"name": str, "label": str}`.

Update `RelationshipExtractor.extract()` to use `entity["name"]` instead of `entity` (string) everywhere it accesses entities.

Update `GraphBuilder.process_memory()` to pass `entity["label"]` to `GraphNode(label=..., name=...)` instead of hardcoded `"Entity"`.

**Acceptance:** `extract("Apple was founded by Steve Jobs in California.")` returns dicts including `{"name": "Apple", "label": "Organization"}` and `{"name": "Steve Jobs", "label": "Person"}`.

---

### Step 8 — Semantic Contradiction Detection in ConsistencyScorer

**File:** `neurosleepnet/trust/consistency.py` — FULL REWRITE

Add antonym pair list:
```python
ANTONYM_PAIRS = [
    ('success', 'failure'), ('succeed', 'fail'), ('win', 'lose'),
    ('safe', 'dangerous'), ('correct', 'incorrect'), ('true', 'false'),
    ('healthy', 'sick'), ('increase', 'decrease'), ('start', 'stop'),
    ('approve', 'reject'), ('create', 'destroy'), ('love', 'hate'),
    ('hot', 'cold'), ('fast', 'slow'), ('open', 'closed'),
]
```

`score(content)` logic:
1. Search for top-1 existing similar memory (already done in current code)
2. If similarity < 0.6: return 0.8 (not similar enough to conflict)
3. Check negation asymmetry (existing logic) → return 0.15 if asymmetric
4. Check antonym pairs: if one content uses word_a and the other uses word_b from the same pair → return 0.2
5. If similarity > 0.85 and no conflict signal → return 1.0 (consistent reinforcement)
6. Default → return 0.8

**Acceptance:** "Alice succeeded" vs existing "Alice failed" → score < 0.3. "Water is H2O" vs existing "Water is H2O" → score > 0.9.

---

### Step 9 — NREM Synthesis via ContextCompressor

**File:** `neurosleepnet/sleep/engine.py`

In `nrem_consolidation()`, replace the naive `" | ".join(contents)` with:
1. Import `ContextCompressor` from `neurosleepnet.compression.compressor`
2. Import `re` and `collections.Counter`
3. Build a `synthesis_query` from the 5 most common non-trivial words across all episodic memories
4. Call `ContextCompressor(max_tokens=150).compress(episodic_mems, query=synthesis_query)`
5. Use compressed result as the content of the new SEMANTIC memory
6. Also store `"source_ids": [m['id'] for m in episodic_mems]` in the metadata
7. Pass `memory_type="semantic"` to the storage `store()` call

**Acceptance:** Consolidating 5 episodic memories produces a coherent English sentence, not a pipe-separated dump.

---

## PHASE C — SDK Wiring (Most Critical Phase)

### Step 10 — Rewrite sdk/memory.py as the Intelligence Orchestrator

**File:** `neurosleepnet/sdk/memory.py` — MAJOR REWRITE

This file must import and wire all components. The public API must remain backward compatible (all existing methods stay) and add new ones.

**New imports at top:**
```python
from neurosleepnet.config.settings import Settings
from neurosleepnet.storage.sqlite import SQLiteAdapter
from neurosleepnet.memory.schemas import MemoryRecord
from neurosleepnet.embeddings.local import LocalEmbeddingProvider
from neurosleepnet.storage.local_vector import FAISSVectorStore
from neurosleepnet.perception.schemas import Observation
from neurosleepnet.perception.classifier import MemoryClassifier
from neurosleepnet.perception.detector import DuplicateDetector
from neurosleepnet.perception.importance import ImportanceScorer
from neurosleepnet.trust.engine import TrustEngine
from neurosleepnet.graph.builder import GraphBuilder
from collections import OrderedDict
from dataclasses import dataclass
import json, datetime
```

**New dataclass at module level:**
```python
@dataclass
class ObserveResult:
    stored: bool
    memory_id: str = None
    memory_type: str = None
    importance: float = None
    trust_score: float = None
    is_duplicate: bool = False
    reason: str = ""
```

**New `Memory.__init__(self, namespace="default", db_path="neurosleepnet.db")`:**
```python
self.namespace = namespace
self.settings = Settings()
self.storage = SQLiteAdapter(db_path=db_path)
self.embedder = LocalEmbeddingProvider()
self.vector_store = FAISSVectorStore(
    self.storage,
    index_path=db_path.replace(".db", ".faiss"),
    ids_path=db_path.replace(".db", ".faiss_ids")
)
self.classifier = MemoryClassifier(embedder=self.embedder)
self.importance_scorer = ImportanceScorer()
self.duplicate_detector = DuplicateDetector(self)
self.trust_engine = TrustEngine(self)
self.graph_builder = GraphBuilder(self.storage)
self._working_memory = OrderedDict()
self._working_memory_size = 50
self._hooks = {}
```

**New primary `observe(self, content, source="agent", metadata=None) -> ObserveResult`:**
Runs the full pipeline in this exact order:
1. Create `Observation(content=content, source=source, metadata=metadata or {})`
2. Call `self.duplicate_detector.is_duplicate(obs)` → if True, emit `"duplicate_detected"` hook, return `ObserveResult(stored=False, is_duplicate=True)`
3. Call `self.importance_scorer.score(obs)` → `importance`
4. Call `self.classifier.classify(obs)` → `memory_type`
5. Call `self.trust_engine.calculate(obs)` → `trust_profile`
6. Call `self.embedder.embed(content)` → `embedding`
7. Create `MemoryRecord(content, metadata={...obs.metadata, source, type=memory_type.upper()}, importance, trust_score=trust_profile.final_score, embedding, namespace=self.namespace, memory_type=memory_type)`
8. Call `self.storage.store(...)` with all fields including `namespace` and `memory_type`
9. Call `self.vector_store.add(record.id, embedding)`
10. Call `self.graph_builder.process_memory(record.to_dict())`
11. Emit `"stored"` hook with `{memory_id, type, importance}`
12. Return `ObserveResult(stored=True, memory_id, memory_type, importance, trust_score, is_duplicate=False)`

**Updated `store()` (backward-compat low-level path, no intelligence):**
Same as current but also calls `self.vector_store.add(record.id, embedding)` and passes `namespace=self.namespace`.

**Updated `search()`:**
Same as current but calls `self.storage.increment_access(r['id'])` for each result before returning, and calls `self._cache_results(results)`.

**Updated `search_keyword()`:**
Pass `namespace=self.namespace` to `self.storage.search_keyword()`. Call `increment_access` and `_cache_results` on results.

**Updated `search_hybrid()`:**
Add `semantic_weight=1.5`, `keyword_weight=0.8` parameters. Replace equal-weight RRF with weighted RRF: `score += weight / (k + rank + 1)`. Call `_cache_results` on final results.

**New working memory methods:**
```python
def _cache_results(self, results: list):
    for r in results:
        self._working_memory[r['id']] = r
        if len(self._working_memory) > self._working_memory_size:
            self._working_memory.popitem(last=False)

def get_working_memory(self) -> list:
    return list(self._working_memory.values())
```

**Updated `get()`:**
Check `self._working_memory` first before hitting SQLite.

**New event hook methods:**
```python
def on(self, event: str, callback) -> "Memory":
    self._hooks.setdefault(event, []).append(callback)
    return self

def _emit(self, event: str, data: dict):
    for cb in self._hooks.get(event, []):
        try: cb(data)
        except: pass
```

**Acceptance:**
```python
m = Memory()
r = m.observe("Alice built NeuroSleepNet.")
assert r.stored == True and r.memory_type in ["episodic", "semantic", "procedural"]

r2 = m.observe("Alice built NeuroSleepNet.")
assert r2.is_duplicate == True and r2.stored == False
```

---

### Step 11 — Add Batch Ingestion API

**File:** `neurosleepnet/sdk/memory.py` (add method)

`ingest_batch(self, items: list, source="batch") -> list[ObserveResult]`:

- Parse items: each can be `str` OR `dict` with keys `content`, `source`, `metadata`
- Collect all `content` strings into a list
- Call `self.embedder.embed_batch(all_contents)` — single model forward pass
- For each item + its pre-computed embedding:
  - Run duplicate check, importance, classification, trust (same as observe)
  - Store if not duplicate, add to FAISS index, build graph
  - Collect `ObserveResult` per item
- Return list of results

**Acceptance:** `memory.ingest_batch(["a", "b", "c"])` uses exactly 1 call to `embed_batch`, not 3 calls to `embed`.

---

## PHASE D — Retrieval Upgrades

### Step 12 — Graph Search + 3-Modality Hybrid Search

**File:** `neurosleepnet/sdk/memory.py` (add/update methods)

**New `search_graph(self, query, limit=5) -> list`:**
1. Use `EntityExtractor` to extract entities from the query
2. For each entity (up to 3), call `self.storage.query_graph(entity["name"])`
3. From the graph result, retrieve memories via `source_memory` in node/edge properties
4. Attach `score=0.7` for direct node match, `score=0.65` for traversal
5. Deduplicate by memory id, return up to `limit`

**Updated `search_hybrid(self, query, limit=5, semantic_weight=1.5, keyword_weight=0.8, graph_weight=1.0) -> list`:**
1. Get `semantic_results`, `keyword_results`, `graph_results`
2. Run weighted RRF across all 3 lists: `score += weight / (60 + rank + 1)`
3. Sort by total RRF score, return top `limit`
4. Each result gets `hybrid_score` field

**Acceptance:** After observing "Alice built NeuroSleepNet", `search_graph("Alice")` returns that memory. `search_hybrid("Alice")` results include `hybrid_score` field.

---

### Step 13 — Working Memory LRU Cache in get()

Already wired in Step 10. Verify `get(memory_id)` checks `_working_memory` before SQLite. No additional changes needed.

---

## PHASE E — Sleep Engine Upgrades

### Step 14 — Access-Frequency Decay in SleepEngine

**File:** `neurosleepnet/sleep/engine.py`

Add new `apply_decay(self, min_importance=0.05)` method:
- Iterate all memories via `self.memory.storage.list()`
- If `access_count == 0`: multiply importance by 0.85, floor at `min_importance`
- If `access_count > 0`: add `min(access_count * 0.02, 0.2)` to importance, cap at 1.0
- If `memory_type == "episodic"` and `access_count >= 3`: promote to `"semantic"`, add `"promoted_from": "episodic"` to metadata
- Re-store updated records via `self.memory.storage.store()`

Update `trigger()` to call `self.apply_decay()` after `rem_consolidation()`.

**Acceptance:** After `trigger_sleep()`, a zero-access memory has lower importance. An episodic memory with `access_count=5` becomes `memory_type="semantic"`.

---

## PHASE F — Developer Experience

### Step 15 — Namespace Propagation

**File:** `neurosleepnet/storage/sqlite.py`

Ensure `search_keyword(query, limit, namespace=None)` adds `AND namespace = ?` when namespace is provided.

**File:** `neurosleepnet/sdk/memory.py`

Ensure `search_keyword()` passes `namespace=self.namespace` to `self.storage.search_keyword()`. Ensure `list()` uses `self.storage.list_namespace(self.namespace)` instead of `self.storage.list()`.

**Acceptance:** `Memory(namespace="agent_1").observe("X")` and `Memory(namespace="agent_2").observe("Y")` — each instance's `list()` returns only its own memories.

---

### Step 16 — Public Convenience Methods on Memory

**File:** `neurosleepnet/sdk/memory.py` (add methods)

```python
def trigger_sleep(self):
    """Runs NREM + REM + Decay consolidation cycle."""
    from neurosleepnet.sleep.engine import SleepEngine
    return SleepEngine(self).trigger()

def reasoning_pack(self, topic: str) -> str:
    """Returns JSON reasoning pack for an SLM around a topic."""
    from neurosleepnet.compression.pack import ReasoningPackGenerator
    return ReasoningPackGenerator(self.storage, memory=self).generate_pack(topic)
```

---

### Step 17 — forget() and forget_entity() Methods

**File:** `neurosleepnet/sdk/memory.py` (add methods)

`forget(self, memory_id) -> bool`:
- Call `self.storage.delete(memory_id)`
- Remove from `_working_memory`
- Rebuild FAISS index: call `self.vector_store._build_from_storage()` then `_persist()`
- Emit `"forgotten"` hook
- Return True

`forget_entity(self, entity_name) -> int`:
- Query graph for entity
- Collect all `source_memory` IDs from node and edge properties
- Call `self.forget(id)` for each
- Return count deleted

---

### Step 18 — AsyncMemory Wrapper

**File:** `neurosleepnet/sdk/async_memory.py` — NEW FILE

```python
import asyncio
from .memory import Memory, ObserveResult

class AsyncMemory:
    """Async wrapper for use with async AI agents (LangChain, AutoGen, CrewAI)."""
    def __init__(self, **kwargs):
        self._memory = Memory(**kwargs)

    async def observe(self, content, source="agent", metadata=None) -> ObserveResult:
        loop = asyncio.get_event_loop()
        return await loop.run_in_executor(None, lambda: self._memory.observe(content, source, metadata))

    async def search_hybrid(self, query, limit=5) -> list:
        loop = asyncio.get_event_loop()
        return await loop.run_in_executor(None, lambda: self._memory.search_hybrid(query, limit))

    async def search(self, query, limit=5) -> list:
        loop = asyncio.get_event_loop()
        return await loop.run_in_executor(None, lambda: self._memory.search(query, limit))

    async def trigger_sleep(self):
        loop = asyncio.get_event_loop()
        return await loop.run_in_executor(None, self._memory.trigger_sleep)

    async def reasoning_pack(self, topic) -> str:
        loop = asyncio.get_event_loop()
        return await loop.run_in_executor(None, lambda: self._memory.reasoning_pack(topic))

    def __getattr__(self, name):
        return getattr(self._memory, name)
```

**File:** `neurosleepnet/__init__.py` — update to:
```python
from .sdk.memory import Memory
from .sdk.async_memory import AsyncMemory

__version__ = "0.2.0"
__all__ = ["Memory", "AsyncMemory"]
```

---

### Step 19 — Update pyproject.toml

**File:** `pyproject.toml`

Update dependencies list to:
```toml
dependencies = [
    "pyyaml>=6.0",
    "sentence-transformers>=2.2.0",
    "faiss-cpu>=1.7.4",
    "numpy>=1.24.0",
    "spacy>=3.7.0",
]
```

Update version to `"0.2.0"`.

After install, builder must run: `python -m spacy download en_core_web_sm`

---

## PHASE G — Advanced Features

### Step 20 — Memory Timeline API

**File:** `neurosleepnet/sdk/memory.py` (add method)

`timeline(self, start_iso=None, end_iso=None, memory_type=None) -> list`:
1. Get all memories via `self.storage.list_namespace(self.namespace)`
2. Filter by `created_at` if `start_iso` or `end_iso` provided
3. Filter by `memory_type` if provided
4. Sort by `created_at` ascending
5. Return list of dicts: `{id, content, created_at, type, importance}`

---

### Step 21 — Proactive Memory Surfacing

**File:** `neurosleepnet/sdk/memory.py` (add method)

`surface_relevant(self, context, threshold=0.75) -> list`:
1. Call `self.search_hybrid(context, limit=10)`
2. Filter results where `hybrid_score >= threshold`
3. Sort by `relevance_score * importance` descending
4. Return list of dicts: `{content, memory_type, relevance_score, importance}`

---

### Step 22 — Expand Config

**File:** `neurosleepnet/config/defaults.yaml`

```yaml
backend: sqlite
vector_store: faiss
cache: memory
namespace: default
duplicate_threshold: 0.95
importance_decay_rate: 0.85
max_working_memory: 50
embedding_model: all-MiniLM-L6-v2
semantic_weight: 1.5
keyword_weight: 0.8
graph_weight: 1.0
```

**File:** `neurosleepnet/config/settings.py`

Extend `_load_defaults` to parse all new yaml keys. Extend `_load_env` to support `NSN_NAMESPACE`, `NSN_EMBEDDING_MODEL`, `NSN_DUPLICATE_THRESHOLD` env var overrides.

---

### Step 23 — Upgrade ReasoningPackGenerator

**File:** `neurosleepnet/compression/pack.py`

Update `__init__` to accept optional `memory=None` parameter.

In `generate_pack(topic)`:
- If `self.memory` is not None: use `self.memory.search_hybrid(topic, limit=10)` for retrieval
- Else: fall back to `self.storage.search_keyword(topic, limit=10)`
- Add `"key_facts"` field: top 3 memories with `importance >= 0.7`
- Upgrade system prompt to: `"You are a reasoning engine with access to long-term memory. Use the provided context, key facts, and logical rules to answer precisely. Prefer information from key_facts when answering direct questions."`

---

## Final End-to-End Acceptance Test

This exact code must run without errors after all steps are complete:

```python
from neurosleepnet import Memory

memory = Memory(namespace="my_agent")

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
import json
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
surfaced = memory.surface_relevant("What happened with the deployment?")
assert len(surfaced) >= 1

# Event hook
events = []
memory.on("stored", lambda e: events.append(e))
memory.observe("New fact after hook setup.", source="test")
assert len(events) == 1

# Namespace isolation
m2 = Memory(namespace="other_agent")
m2.observe("Agent 2 fact only.")
assert len(memory.list()) != len(m2.list()), "Namespaces are leaking"

# Async
import asyncio
from neurosleepnet import AsyncMemory
async def test_async():
    am = AsyncMemory(namespace="async_test")
    r = await am.observe("async test memory")
    assert r.stored == True
asyncio.run(test_async())

print("ALL ACCEPTANCE TESTS PASSED.")
```

---

## File Change Summary

| File | Action |
|------|--------|
| `neurosleepnet/storage/sqlite.py` | MODIFY — new columns, WAL, increment_access, list_namespace |
| `neurosleepnet/memory/schemas.py` | MODIFY — 4 new fields, resilient from_dict |
| `neurosleepnet/storage/local_vector.py` | REWRITE — FAISSVectorStore |
| `neurosleepnet/embeddings/local.py` | REWRITE — lazy loading, module cache |
| `neurosleepnet/perception/classifier.py` | REWRITE — embedding zero-shot classification |
| `neurosleepnet/perception/importance.py` | REWRITE — density signals |
| `neurosleepnet/graph/extractor.py` | REWRITE — spaCy NER + fallback |
| `neurosleepnet/trust/consistency.py` | REWRITE — antonym pair detection |
| `neurosleepnet/sleep/engine.py` | MODIFY — compressor synthesis, apply_decay |
| `neurosleepnet/sdk/memory.py` | MAJOR REWRITE — full pipeline wiring |
| `neurosleepnet/sdk/async_memory.py` | NEW — AsyncMemory wrapper |
| `neurosleepnet/compression/pack.py` | MODIFY — hybrid search, key_facts, memory param |
| `neurosleepnet/__init__.py` | MODIFY — export AsyncMemory, bump version |
| `neurosleepnet/config/defaults.yaml` | MODIFY — new config keys |
| `neurosleepnet/config/settings.py` | MODIFY — parse new keys |
| `pyproject.toml` | MODIFY — faiss-cpu, spacy, version 0.2.0 |
