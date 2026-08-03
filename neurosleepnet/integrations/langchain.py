"""
NeuroSleepNet — LangChain Memory Integration

Implements LangChain's BaseChatMessageHistory interface so NeuroSleepNet
can be used as a drop-in memory backend for any LangChain chain or agent.

Usage (3 lines)::

    from neurosleepnet.integrations.langchain import NeurosleepNetHistory
    from langchain.chains import ConversationChain
    chain = ConversationChain(llm=llm, memory=NeurosleepNetHistory(namespace="agent"))

Requires: pip install langchain-core
"""

from __future__ import annotations

from typing import List, Sequence

from neurosleepnet.sdk.memory import Memory

try:
    from langchain_core.chat_history import BaseChatMessageHistory
    from langchain_core.messages import BaseMessage, HumanMessage, AIMessage, SystemMessage
    _LANGCHAIN_AVAILABLE = True
except ImportError:
    _LANGCHAIN_AVAILABLE = False
    BaseChatMessageHistory = object
    BaseMessage = object


def _require_langchain():
    if not _LANGCHAIN_AVAILABLE:
        raise ImportError(
            "langchain-core is required for the LangChain integration. "
            "Install it with: pip install langchain-core"
        )


class NeurosleepNetHistory(BaseChatMessageHistory):
    """
    LangChain-compatible chat message history backed by NeuroSleepNet.

    All messages are stored in NSN's cognitive memory pipeline — they are
    automatically classified, deduplicated, importance-scored, and become
    searchable via hybrid retrieval.

    Example with ConversationChain::

        from langchain_openai import ChatOpenAI
        from langchain.chains import ConversationChain
        from langchain.memory import ConversationBufferMemory
        from neurosleepnet.integrations.langchain import NeurosleepNetHistory

        history = NeurosleepNetHistory(namespace="session_123")
        memory = ConversationBufferMemory(chat_memory=history)
        chain = ConversationChain(llm=ChatOpenAI(), memory=memory)

    Example with RunnableWithMessageHistory::

        from langchain_core.runnables.history import RunnableWithMessageHistory
        from neurosleepnet.integrations.langchain import NeurosleepNetHistory

        chain_with_memory = RunnableWithMessageHistory(
            runnable=your_chain,
            get_session_history=lambda sid: NeurosleepNetHistory(namespace=sid),
        )
    """

    def __init__(
        self,
        namespace: str = "default",
        db_path: str = "neurosleepnet.db",
        **kwargs,
    ):
        _require_langchain()
        self._memory = Memory(namespace=namespace, db_path=db_path, **kwargs)
        self._messages: list = []
        self._loaded = False

    @property
    def messages(self) -> List[BaseMessage]:
        """Load message history from memory on first access."""
        _require_langchain()
        if not self._loaded:
            self._messages = self._load_from_memory()
            self._loaded = True
        return self._messages

    def _load_from_memory(self) -> List[BaseMessage]:
        """Reconstruct message history from NSN's timeline."""
        _require_langchain()
        timeline = self._memory.timeline(limit=50, ascending=True)
        messages = []
        for entry in timeline:
            content = entry["content"]
            meta = {}
            # Use memory_type metadata to recover original role
            if "[HUMAN]" in content:
                messages.append(HumanMessage(content=content.replace("[HUMAN] ", "")))
            elif "[AI]" in content:
                messages.append(AIMessage(content=content.replace("[AI] ", "")))
            elif "[SYSTEM]" in content:
                messages.append(SystemMessage(content=content.replace("[SYSTEM] ", "")))
            else:
                messages.append(HumanMessage(content=content))
        return messages

    def add_message(self, message: BaseMessage) -> None:
        """Store a LangChain message in NeuroSleepNet."""
        _require_langchain()
        role = type(message).__name__.replace("Message", "").upper()
        tagged_content = f"[{role}] {message.content}"
        self._memory.observe(
            tagged_content,
            source="langchain",
            metadata={"role": role, "langchain_type": type(message).__name__},
        )
        self._messages.append(message)

    def add_messages(self, messages: Sequence[BaseMessage]) -> None:
        for msg in messages:
            self.add_message(msg)

    def clear(self) -> None:
        """Clear in-memory message cache (does not delete from NSN storage)."""
        self._messages = []
        self._loaded = False

    def search_relevant(self, query: str, limit: int = 5) -> List[str]:
        """
        Retrieve relevant past messages for a query using hybrid search.
        Useful for building condensed context windows.

        Returns a list of content strings.
        """
        results = self._memory.search_hybrid(query, limit=limit)
        return [r["content"] for r in results]

    def __repr__(self) -> str:  # pragma: no cover
        return f"NeurosleepNetHistory(namespace={self._memory.namespace!r})"
