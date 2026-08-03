"""
NeuroSleepNet — OpenAI-Compatible Context Injector

Injects long-term memory context directly into OpenAI-format message lists
for use with any OpenAI-compatible API (GPT-4o, Claude via bedrock, Gemini, etc.)

Usage (3 lines)::

    from neurosleepnet.integrations.openai_adapter import MemoryInjector
    injector = MemoryInjector(namespace="my_agent")
    messages = injector.inject(query, messages)
    response = client.chat.completions.create(model="gpt-4o", messages=messages)
"""

from __future__ import annotations

import json
from typing import Any

from neurosleepnet.sdk.memory import Memory, ObserveResult


class MemoryInjector:
    """
    Injects NeuroSleepNet memory context into OpenAI chat message arrays.

    Works with any OpenAI-compatible API: OpenAI, Azure OpenAI, Anthropic
    (via the Messages API format), Together.ai, Groq, Ollama, etc.

    Example::

        from openai import OpenAI
        from neurosleepnet.integrations.openai_adapter import MemoryInjector

        client = OpenAI()
        injector = MemoryInjector(namespace="my_agent")

        # Remember facts autonomously
        injector.remember("The project deadline is March 15, 2026.")

        # Inject memory into any conversation
        messages = [{"role": "user", "content": "When is the deadline?"}]
        enriched = injector.inject("When is the deadline?", messages)
        response = client.chat.completions.create(model="gpt-4o", messages=enriched)
    """

    SYSTEM_PREFIX = (
        "You have access to long-term memory. "
        "Use the following recalled context to answer the user's question accurately.\n\n"
        "=== RECALLED MEMORY ===\n"
        "{context}\n"
        "=== END MEMORY ===\n\n"
        "Answer based on the above context. "
        "If the context is not relevant, rely on your general knowledge."
    )

    def __init__(
        self,
        namespace: str = "default",
        db_path: str = "neurosleepnet.db",
        recall_limit: int = 5,
        **kwargs,
    ):
        self._memory = Memory(namespace=namespace, db_path=db_path, **kwargs)
        self.recall_limit = recall_limit

    def remember(self, text: str, source: str = "agent", metadata: dict | None = None) -> ObserveResult:
        """Store a fact in long-term memory."""
        return self._memory.observe(text, source=source, metadata=metadata)

    def recall(self, query: str, limit: int | None = None) -> list[dict[str, Any]]:
        """Retrieve relevant memories for a query."""
        return self._memory.search_hybrid(query, limit=limit or self.recall_limit)

    def inject(
        self,
        query: str,
        messages: list[dict[str, str]],
        limit: int | None = None,
        reasoning_pack: bool = False,
    ) -> list[dict[str, str]]:
        """
        Inject memory context as a system message into an OpenAI messages array.

        Args:
            query: The current user query to search memory for.
            messages: The existing messages list (will not be mutated).
            limit: Override number of memories to retrieve.
            reasoning_pack: If True, include the full JSON reasoning pack instead
                            of simple bullet points.

        Returns:
            A new messages list with a memory context system message prepended.
        """
        results = self.recall(query, limit=limit)

        if not results:
            return messages

        if reasoning_pack:
            pack_json = self._memory.reasoning_pack(query)
            pack = json.loads(pack_json)
            context_lines = pack.get("context", [])
            key_facts = pack.get("key_facts", [])
            context_str = ""
            if key_facts:
                context_str += "Key Facts:\n" + "\n".join(f"- {f}" for f in key_facts) + "\n\n"
            if context_lines:
                context_str += "Additional Context:\n" + "\n".join(f"- {c}" for c in context_lines)
        else:
            context_str = "\n".join(
                f"[{r.get('memory_type', 'memory').upper()}] {r['content']}"
                for r in results
            )

        memory_system_message = {
            "role": "system",
            "content": self.SYSTEM_PREFIX.format(context=context_str),
        }

        # Prepend the memory system message; preserve any existing system messages
        return [memory_system_message] + list(messages)

    def observe_and_inject(
        self,
        user_message: str,
        messages: list[dict[str, str]],
        source: str = "user",
    ) -> list[dict[str, str]]:
        """
        Convenience: observe the user message then inject relevant memories.
        Combines remember() + inject() in one call.
        """
        self.remember(user_message, source=source)
        return self.inject(user_message, messages)

    def __repr__(self) -> str:  # pragma: no cover
        return f"MemoryInjector(namespace={self._memory.namespace!r})"
