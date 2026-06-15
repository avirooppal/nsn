# BUILD_PLAN.md

# NeuroSleepNet Development Roadmap

## Purpose

This document defines the implementation roadmap for NeuroSleepNet.

The roadmap is designed for:

- Human engineers
- Coding agents
- Autonomous engineering systems

Each task must satisfy:

- Single responsibility
- Clear start state
- Clear end state
- Independent testing
- Mergeable without future tasks

No task should depend on unfinished future systems.

---

# Product Goal

NeuroSleepNet is a Cognitive Memory Operating System for Small Language Models (SLMs) and AI Agents.

Primary Goal:

```text
Increase the effective intelligence of SLMs through:

- Long-term memory
- Knowledge formation
- Context compression
- Reasoning pack generation
```

Developer Experience Goal:

```python
from neurosleepnet import Memory

memory = Memory()

memory.store(
    "User prefers TypeScript"
)

memory.reasoning_pack()
```

No configuration.

No cloud account.

No vector database setup.

No Docker.

Works locally.

---

# PHASE 0

# Repository Foundation

Goal:

```text
Create installable package.
```

---

## TASK-0001

Title:

Create repository structure.

Output:

```text
neurosleepnet/
tests/
README.md
pyproject.toml
```

Acceptance:

```bash
pip install -e .
```

succeeds.

---

## TASK-0002

Create package entrypoint.

Output:

```python
import neurosleepnet
```

Acceptance:

Import succeeds.

---

## TASK-0003

Create Memory class.

Output:

```python
Memory()
```

Acceptance:

Instantiation succeeds.

---

# PHASE 1

# Configuration System

Goal:

Centralized configuration.

---

## TASK-0010

Create Settings object.

Acceptance:

```python
Settings()
```

returns defaults.

---

## TASK-0011

Create defaults.yaml.

Acceptance:

Settings load YAML.

---

## TASK-0012

Environment variable overrides.

Acceptance:

Environment variables override config.

---

# PHASE 2

# Storage Layer

Goal:

Persistent memory storage.

---

## TASK-0020

Create StorageAdapter interface.

Methods:

```python
store()
get()
delete()
list()
```

Acceptance:

Interface imports.

---

## TASK-0021

Implement SQLite adapter.

Acceptance:

SQLite database created.

---

## TASK-0022

Create memories table.

Acceptance:

Table exists.

---

## TASK-0023

Store memory record.

Acceptance:

Insert succeeds.

---

## TASK-0024

Get memory record.

Acceptance:

Returns stored data.

---

## TASK-0025

Delete memory record.

Acceptance:

Record removed.

---

## TASK-0026

List memory records.

Acceptance:

Returns records.

---

# PHASE 3

# Memory Schema

Goal:

Standard memory format.

---

## TASK-0030

Create MemoryRecord model.

Fields:

- id
- content
- created_at

Acceptance:

Serialization works.

---

## TASK-0031

Add metadata.

Acceptance:

Metadata persists.

---

## TASK-0032

Add importance field.

Acceptance:

Persists.

---

## TASK-0033

Add trust score.

Acceptance:

Persists.

---

# PHASE 4

# Public SDK

Goal:

Minimal usable API.

---

## TASK-0040

Implement memory.store().

Acceptance:

Stores memory.

---

## TASK-0041

Implement memory.get().

Acceptance:

Returns memory.

---

## TASK-0042

Implement memory.list().

Acceptance:

Returns memories.

---

# PHASE 5

# Embeddings

Goal:

Semantic representation.

---

## TASK-0050

EmbeddingProvider interface.

Acceptance:

Imports.

---

## TASK-0051

Local embedding provider.

Acceptance:

Embeddings generated.

---

## TASK-0052

Store embeddings.

Acceptance:

Vectors persist.

---

## TASK-0053

Load embeddings.

Acceptance:

Vectors load.

---

# PHASE 6

# Retrieval

Goal:

Useful memory search.

---

## TASK-0060

Similarity function.

Acceptance:

Similarity score generated.

---

## TASK-0061

Top-k retrieval.

Acceptance:

Correct nearest neighbor.

---

## TASK-0062

memory.retrieve().

Acceptance:

Relevant memory returned.

---

## TASK-0063

Keyword retrieval.

Acceptance:

Exact match works.

---

## TASK-0064

Hybrid retrieval.

Acceptance:

Keyword + semantic search combined.

---

# PHASE 7

# Perception Layer

Goal:

Control memory creation.

---

## TASK-0070

Observation schema.

Acceptance:

Creates successfully.

---

## TASK-0071

Duplicate detector.

Acceptance:

Duplicates identified.

---

## TASK-0072

Importance scoring.

Acceptance:

Score assigned.

---

## TASK-0073

Memory classification.

Outputs:

- episodic
- semantic
- procedural

Acceptance:

Classification returned.

---

# PHASE 8

# Trust Engine

Goal:

Memory quality.

---

## TASK-0080

Trust schema.

Acceptance:

Created.

---

## TASK-0081

Source scoring.

Acceptance:

Score generated.

---

## TASK-0082

Recency scoring.

Acceptance:

Recent memories score higher.

---

## TASK-0083

Consistency scoring.

Acceptance:

Conflicts detected.

---

## TASK-0084

Trust calculation.

Acceptance:

Final trust score generated.

---

# PHASE 9

# Knowledge Graph

Goal:

Relationship-aware memory.

---

## TASK-0090

Graph node model.

Acceptance:

Node created.

---

## TASK-0091

Graph edge model.

Acceptance:

Edge created.

---

## TASK-0092

Graph persistence.

Acceptance:

Graph stored.

---

## TASK-0093

Entity extraction.

Acceptance:

Entities extracted.

---

## TASK-0094

Relationship extraction.

Acceptance:

Edges created.

---

## TASK-0095

Graph traversal.

Acceptance:

Related nodes returned.

---

# PHASE 10

# Sleep Engine

Goal:

Memory evolution.

---

## TASK-0100

Sleep job model.

Acceptance:

Created.

---

## TASK-0101

Memory clustering.

Acceptance:

Related memories grouped.

---

## TASK-0102

Duplicate consolidation.

Acceptance:

Duplicates merged.

---

## TASK-0103

Decay algorithm.

Acceptance:

Importance decreases over time.

---

## TASK-0104

Forgetting policy.

Acceptance:

Low-value memories removed.

---

## TASK-0105

Promotion policy.

Acceptance:

Episodic promoted to semantic.

---

## TASK-0106

Conflict resolution.

Acceptance:

Conflicting memories resolved.

---

# PHASE 11

# Context Compression

Goal:

Turn large memory collections into compact cognition.

---

## TASK-0110

Profile builder.

Acceptance:

User profile generated.

---

## TASK-0111

Goal extractor.

Acceptance:

Goals identified.

---

## TASK-0112

Preference extractor.

Acceptance:

Preferences identified.

---

## TASK-0113

Project state extractor.

Acceptance:

Project summary generated.

---

## TASK-0114

Memory compression engine.

Acceptance:

Memory collection reduced.

---

## TASK-0115

Compression benchmarking.

Acceptance:

Compression ratio measured.

---

# PHASE 12

# Reasoning Pack Generator

Goal:

Generate SLM-ready cognition.

---

## TASK-0120

ReasoningPack schema.

Acceptance:

Created.

---

## TASK-0121

Profile section generation.

Acceptance:

Populated.

---

## TASK-0122

Knowledge section generation.

Acceptance:

Populated.

---

## TASK-0123

Constraint section generation.

Acceptance:

Populated.

---

## TASK-0124

Goal section generation.

Acceptance:

Populated.

---

## TASK-0125

Reasoning pack assembly.

Acceptance:

Complete pack generated.

---

## TASK-0126

memory.reasoning_pack()

Acceptance:

Returns pack.

---

# PHASE 13

# Explainability

Goal:

Transparent memory decisions.

---

## TASK-0130

Explanation schema.

Acceptance:

Created.

---

## TASK-0131

Retrieval explanations.

Acceptance:

Reason returned.

---

## TASK-0132

Trust explanations.

Acceptance:

Reason returned.

---

## TASK-0133

Memory lineage.

Acceptance:

History visible.

---

# PHASE 14

# Daemon Mode

Goal:

Shared memory service.

---

## TASK-0140

FastAPI bootstrap.

Acceptance:

Server starts.

---

## TASK-0141

Store endpoint.

Acceptance:

Stores memory.

---

## TASK-0142

Retrieve endpoint.

Acceptance:

Retrieves memory.

---

## TASK-0143

Reasoning pack endpoint.

Acceptance:

Returns pack.

---

## TASK-0144

Sleep endpoint.

Acceptance:

Triggers sleep cycle.

---

# PHASE 15

# MCP Integration

Goal:

Native agent interoperability.

---

## TASK-0150

MCP bootstrap.

Acceptance:

Server starts.

---

## TASK-0151

Store memory tool.

Acceptance:

Tool callable.

---

## TASK-0152

Retrieve memory tool.

Acceptance:

Tool callable.

---

## TASK-0153

Reasoning pack tool.

Acceptance:

Tool callable.

---

# PHASE 16

# Plug-and-Play Experience

Goal:

One-command setup.

---

## TASK-0160

Default SQLite configuration.

Acceptance:

No setup required.

---

## TASK-0161

Bundled local embeddings.

Acceptance:

Works offline.

---

## TASK-0162

Zero-config initialization.

Acceptance:

```python
memory = Memory()
```

works.

---

## TASK-0163

Auto-database creation.

Acceptance:

Database created automatically.

---

## TASK-0164

Auto-sleep scheduling.

Acceptance:

Runs without user setup.

---

# PHASE 17

# Benchmarking

Goal:

Prove SLM amplification.

---

## TASK-0170

Create benchmark dataset.

Acceptance:

Loads successfully.

---

## TASK-0171

Retrieval benchmark.

Acceptance:

Metrics generated.

---

## TASK-0172

Compression benchmark.

Acceptance:

Metrics generated.

---

## TASK-0173

Reasoning-pack benchmark.

Acceptance:

Metrics generated.

---

## TASK-0174

SLM comparison benchmark.

Compare:

- Raw SLM
- SLM + NeuroSleepNet

Acceptance:

Improvement measurable.

---

# PHASE 18

# Release Candidate

---

## TASK-0180

Embedded mode validation.

Acceptance:

Works offline.

---

## TASK-0181

Daemon mode validation.

Acceptance:

Works.

---

## TASK-0182

MCP validation.

Acceptance:

Cursor and Claude connect.

---

## TASK-0183

100k memory stress test.

Acceptance:

Passes.

---

## TASK-0184

Documentation completion.

Acceptance:

Examples run successfully.

---

# Definition of Done

NeuroSleepNet is complete when:

```python
memory = Memory()

memory.store(
    "User prefers TypeScript"
)

memory.reasoning_pack()
```

returns a structured cognitive state that measurably improves the performance of a small language model compared to retrieval-only memory systems.

The system must:

- Work offline
- Require zero configuration
- Be self-hostable
- Be model agnostic
- Be storage agnostic
- Be MCP compatible
- Improve SLM performance through memory and context compression

```

```
