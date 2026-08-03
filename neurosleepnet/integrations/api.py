"""
NeuroSleepNet — FastAPI REST Microservice

Exposes all NeuroSleepNet capabilities as a REST API so any agent,
service, or language can integrate via HTTP.

Usage::

    # Start the server:
    uvicorn neurosleepnet.integrations.api:app --host 0.0.0.0 --port 8000

    # Or programmatically:
    from neurosleepnet.integrations.api import create_app
    app = create_app(namespace="my_agent", db_path="agent.db")

Requires: pip install neurosleepnet[api]
"""

from __future__ import annotations

from typing import Any, Optional

try:
    from fastapi import FastAPI, HTTPException, Query
    from fastapi.responses import JSONResponse
    from pydantic import BaseModel
    _FASTAPI_AVAILABLE = True
except ImportError:
    _FASTAPI_AVAILABLE = False
    FastAPI = None
    BaseModel = object


def _require_fastapi():
    if not _FASTAPI_AVAILABLE:
        raise ImportError(
            "fastapi and pydantic are required for the REST API integration. "
            "Install with: pip install neurosleepnet[api]"
        )


# ---------------------------------------------------------------------------
# Pydantic request / response models
# ---------------------------------------------------------------------------

if _FASTAPI_AVAILABLE:
    class ObserveRequest(BaseModel):
        content: str
        source: str = "agent"
        metadata: Optional[dict[str, Any]] = None

    class BatchObserveRequest(BaseModel):
        items: list[dict[str, Any] | str]
        source: str = "batch"

    class ForgetRequest(BaseModel):
        memory_id: str

    class SleepResponse(BaseModel):
        success: bool
        message: str

    class HealthResponse(BaseModel):
        status: str
        version: str
        namespace: str


def create_app(
    namespace: str = "default",
    db_path: str = "neurosleepnet.db",
    title: str = "NeuroSleepNet API",
    version: str = "0.3.0",
) -> "FastAPI":
    """
    Factory that creates and configures a FastAPI application exposing NSN.

    Args:
        namespace: The memory namespace this API instance operates in.
        db_path: Path to the SQLite database file.
        title: OpenAPI title.
        version: API version string.

    Returns:
        A configured FastAPI application ready to be served.
    """
    _require_fastapi()

    from neurosleepnet.sdk.memory import Memory

    memory = Memory(namespace=namespace, db_path=db_path)

    app = FastAPI(
        title=title,
        version=version,
        description=(
            "NeuroSleepNet — Cognitive Memory OS for SLMs and AI Agents. "
            "Provides long-term memory with hybrid search, graph extraction, "
            "and offline consolidation."
        ),
    )

    # -----------------------------------------------------------------------
    # Health
    # -----------------------------------------------------------------------

    @app.get("/health", response_model=HealthResponse, tags=["System"])
    def health():
        """Returns API health status and configuration."""
        return {"status": "ok", "version": version, "namespace": namespace}

    # -----------------------------------------------------------------------
    # Observe (store)
    # -----------------------------------------------------------------------

    @app.post("/observe", tags=["Memory"])
    def observe(req: ObserveRequest):
        """
        Store a new observation in long-term memory.

        The system automatically classifies it (episodic/semantic/procedural),
        scores its importance, checks for duplicates, and indexes it.
        """
        result = memory.observe(req.content, source=req.source, metadata=req.metadata)
        return {
            "stored": result.stored,
            "memory_id": result.memory_id,
            "memory_type": result.memory_type,
            "importance": result.importance,
            "trust_score": result.trust_score,
            "is_duplicate": result.is_duplicate,
        }

    @app.post("/observe/batch", tags=["Memory"])
    def observe_batch(req: BatchObserveRequest):
        """Batch ingest multiple observations using a single embedding pass."""
        results = memory.ingest_batch(req.items, source=req.source)
        return [
            {
                "stored": r.stored,
                "memory_id": r.memory_id,
                "memory_type": r.memory_type,
                "importance": r.importance,
                "is_duplicate": r.is_duplicate,
            }
            for r in results
        ]

    # -----------------------------------------------------------------------
    # Search
    # -----------------------------------------------------------------------

    @app.get("/search", tags=["Retrieval"])
    def search(
        q: str = Query(..., description="Natural language query"),
        limit: int = Query(5, ge=1, le=50),
        mode: str = Query("hybrid", enum=["hybrid", "semantic", "keyword"]),
    ):
        """
        Search long-term memory.

        - mode=hybrid: Combines semantic, keyword, and graph search via RRF (recommended)
        - mode=semantic: Dense vector search via FAISS
        - mode=keyword: SQLite full-text keyword match
        """
        if mode == "hybrid":
            results = memory.search_hybrid(q, limit=limit)
        elif mode == "semantic":
            results = memory.search(q, limit=limit)
        else:
            results = memory.search_keyword(q, limit=limit)

        return {"query": q, "mode": mode, "count": len(results), "results": results}

    @app.get("/surface", tags=["Retrieval"])
    def surface(
        context: str = Query(..., description="Context string for proactive surfacing"),
    ):
        """Proactively surface memories relevant to a context."""
        results = memory.surface_relevant(context)
        return {"context": context, "surfaced": results}

    # -----------------------------------------------------------------------
    # Graph
    # -----------------------------------------------------------------------

    @app.get("/graph/{entity}", tags=["Graph"])
    def entity_subgraph(
        entity: str,
        depth: int = Query(1, ge=1, le=3),
    ):
        """Return the knowledge graph subgraph centered on an entity."""
        subgraph = memory.get_entity_subgraph(entity, depth=depth)
        return {"entity": entity, "depth": depth, **subgraph}

    # -----------------------------------------------------------------------
    # Timeline
    # -----------------------------------------------------------------------

    @app.get("/timeline", tags=["Memory"])
    def timeline(
        memory_type: Optional[str] = Query(None, enum=["episodic", "semantic", "procedural"]),
        limit: int = Query(20, ge=1, le=100),
        ascending: bool = Query(False),
    ):
        """Return chronologically ordered memory entries."""
        entries = memory.timeline(limit=limit, memory_type=memory_type, ascending=ascending)
        return {"count": len(entries), "entries": entries}

    # -----------------------------------------------------------------------
    # Reasoning Pack
    # -----------------------------------------------------------------------

    @app.get("/reasoning-pack", tags=["Reasoning"])
    def reasoning_pack(topic: str = Query(..., description="Topic to build reasoning pack for")):
        """
        Generate a JSON reasoning pack for SLM prompt injection.

        Returns structured context, key facts, and graph-derived logical rules
        for a given topic.
        """
        import json as _json
        pack_str = memory.reasoning_pack(topic)
        return _json.loads(pack_str)

    # -----------------------------------------------------------------------
    # Sleep
    # -----------------------------------------------------------------------

    @app.post("/sleep", response_model=SleepResponse, tags=["Consolidation"])
    def trigger_sleep():
        """
        Trigger the offline memory consolidation cycle (NREM + REM + Decay).

        - NREM: Synthesizes episodic memories into semantic knowledge
        - REM: Resolves contradictions and prunes conflicting memories
        - Decay: Adjusts importance scores based on access frequency
        """
        success = memory.trigger_sleep()
        return {
            "success": success,
            "message": "Sleep cycle complete." if success else "Sleep cycle already running.",
        }

    # -----------------------------------------------------------------------
    # Forget
    # -----------------------------------------------------------------------

    @app.delete("/forget/{memory_id}", tags=["Memory"])
    def forget(memory_id: str):
        """Delete a specific memory by ID and rebuild the FAISS index."""
        success = memory.forget(memory_id)
        if not success:
            raise HTTPException(status_code=404, detail=f"Memory {memory_id} not found.")
        return {"deleted": memory_id}

    @app.delete("/forget/entity/{entity_name}", tags=["Memory"])
    def forget_entity(entity_name: str):
        """Delete all memories linked to a named entity in the knowledge graph."""
        count = memory.forget_entity(entity_name)
        return {"entity": entity_name, "deleted_count": count}

    return app


# Allow direct execution: uvicorn neurosleepnet.integrations.api:app
app = create_app()
