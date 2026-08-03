"""
NeuroSleepNet — Generic Agent Tool Interface

Provides a simple, framework-agnostic tool interface for any AI agent
system (AutoGen, CrewAI, custom agents, etc.).

Usage (3 lines)::

    from neurosleepnet.integrations.tool import MemoryTool
    tool = MemoryTool(namespace="my_agent")
    tool.remember("Alice leads the engineering team.")
    results = tool.recall("Who leads engineering?")
"""

from __future__ import annotations

from typing import Any

from neurosleepnet.sdk.memory import Memory, ObserveResult


class MemoryTool:
    """
    Framework-agnostic memory tool wrapper.

    Designed to be passed as a callable tool to any agent framework that
    supports tool-use APIs (AutoGen, CrewAI, LlamaIndex, custom agents).

    Example::

        from neurosleepnet.integrations.tool import MemoryTool

        tool = MemoryTool(namespace="agent_1")
        tool.remember("The API key expires on Dec 31.")
        hits = tool.recall("When does the API key expire?")
        print(hits[0]["content"])
    """

    name: str = "memory"
    description: str = (
        "Long-term cognitive memory. Use remember() to store facts and "
        "recall() to retrieve relevant context."
    )

    def __init__(self, namespace: str = "default", db_path: str = "neurosleepnet.db", **kwargs):
        self._memory = Memory(namespace=namespace, db_path=db_path, **kwargs)

    # ------------------------------------------------------------------
    # Core tool methods
    # ------------------------------------------------------------------

    def remember(self, text: str, source: str = "agent", metadata: dict | None = None) -> ObserveResult:
        """
        Store a fact, event, or instruction in long-term memory.

        Args:
            text: The content to remember.
            source: The originating source (e.g. 'user', 'llm', 'tool').
            metadata: Optional dict of extra context.

        Returns:
            ObserveResult with .stored, .memory_type, .importance, .is_duplicate.
        """
        return self._memory.observe(text, source=source, metadata=metadata)

    def recall(self, query: str, limit: int = 5) -> list[dict[str, Any]]:
        """
        Retrieve relevant memories using hybrid search.

        Args:
            query: Natural language question or context string.
            limit: Maximum number of results.

        Returns:
            List of memory dicts with 'content', 'memory_type', 'importance', 'hybrid_score'.
        """
        return self._memory.search_hybrid(query, limit=limit)

    def forget(self, memory_id: str) -> bool:
        """Remove a specific memory by its ID."""
        return self._memory.forget(memory_id)

    def forget_entity(self, entity_name: str) -> int:
        """Remove all memories linked to a named entity. Returns count deleted."""
        return self._memory.forget_entity(entity_name)

    def surface(self, context: str) -> list[dict[str, Any]]:
        """Proactively surface memories relevant to a given context."""
        return self._memory.surface_relevant(context)

    def sleep(self) -> bool:
        """Trigger the offline NREM/REM/Decay consolidation cycle."""
        return self._memory.trigger_sleep()

    def reasoning_context(self, topic: str) -> str:
        """Return a JSON reasoning pack for injection into an SLM prompt."""
        return self._memory.reasoning_pack(topic)

    def timeline(self, memory_type: str | None = None, limit: int = 20) -> list[dict]:
        """Return chronological memory entries, optionally filtered by type."""
        return self._memory.timeline(memory_type=memory_type, limit=limit)

    # ------------------------------------------------------------------
    # Compatibility: allow calling the tool directly like a function
    # ------------------------------------------------------------------

    def __call__(self, action: str, **kwargs) -> Any:
        """
        Dispatch-style invocation for frameworks that call tools as callables.

        Supported actions: 'remember', 'recall', 'forget', 'surface', 'sleep'.

        Example::

            tool("remember", text="Alice is the project lead.")
            tool("recall", query="Who leads the project?")
        """
        dispatch = {
            "remember": self.remember,
            "recall": self.recall,
            "forget": self.forget,
            "forget_entity": self.forget_entity,
            "surface": self.surface,
            "sleep": self.sleep,
            "timeline": self.timeline,
            "reasoning_context": self.reasoning_context,
        }
        fn = dispatch.get(action)
        if fn is None:
            raise ValueError(f"Unknown action '{action}'. Valid: {list(dispatch)}")
        return fn(**kwargs)

    # ------------------------------------------------------------------
    # AutoGen / CrewAI tool descriptor helpers
    # ------------------------------------------------------------------

    def as_autogen_tool(self) -> dict:
        """
        Returns an OpenAI-compatible function schema for AutoGen tool registration.
        """
        return {
            "type": "function",
            "function": {
                "name": "memory",
                "description": self.description,
                "parameters": {
                    "type": "object",
                    "properties": {
                        "action": {
                            "type": "string",
                            "enum": ["remember", "recall", "forget", "surface", "sleep"],
                            "description": "The memory operation to perform.",
                        },
                        "text": {
                            "type": "string",
                            "description": "Content to remember (required for 'remember').",
                        },
                        "query": {
                            "type": "string",
                            "description": "Search query (required for 'recall' and 'surface').",
                        },
                        "memory_id": {
                            "type": "string",
                            "description": "Memory ID to delete (required for 'forget').",
                        },
                    },
                    "required": ["action"],
                },
            },
        }

    def __repr__(self) -> str:  # pragma: no cover
        return f"MemoryTool(namespace={self._memory.namespace!r})"
