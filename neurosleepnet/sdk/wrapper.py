"""
NeuroSleepNet — Universal Model Wrapper (NSN)

Wraps any LLM, agent, or callable with persistent cognitive memory.
The wrapped model is a transparent proxy — all original attributes and
methods are preserved. NSN intercepts calls to inject recalled memory
and auto-stores the conversation into long-term memory.

Supported model types (auto-detected):
    - OpenAI / OpenAI-compatible clients (client.chat.completions.create)
    - LangChain LLMs / ChatModels (.invoke, .predict)
    - HuggingFace pipelines (__call__ with text input)
    - Any generic callable

Usage::

    import neurosleepnet
    from openai import OpenAI

    client = OpenAI()
    client = neurosleepnet.NSN(client, namespace="my_agent")

    # Use exactly like OpenAI — memory is injected automatically
    response = client.chat.completions.create(
        model="gpt-4o",
        messages=[{"role": "user", "content": "What did Alice say?"}]
    )
"""

from __future__ import annotations

import logging
from typing import Any

from neurosleepnet.sdk.memory import Memory, ObserveResult

logger = logging.getLogger("neurosleepnet.wrapper")


# ---------------------------------------------------------------------------
# Proxy helpers
# ---------------------------------------------------------------------------

class _CompletionsProxy:
    """Proxies client.chat.completions to intercept create() calls."""

    def __init__(self, real_completions, memory: Memory, recall_limit: int):
        self._real = real_completions
        self._memory = memory
        self._recall_limit = recall_limit

    def create(self, *, messages: list, **kwargs) -> Any:
        # 1. Pull the latest user message as the search query
        user_query = next(
            (m["content"] for m in reversed(messages) if m.get("role") == "user"),
            "",
        )

        # 2. Inject recalled memory as a system message
        enriched = _inject_memory_message(self._memory, user_query, messages, self._recall_limit)

        # 3. Forward to the real completions endpoint
        response = self._real.create(messages=enriched, **kwargs)

        # 4. Auto-store the assistant reply
        try:
            reply = response.choices[0].message.content
            self._memory.observe(reply, source="llm")
        except Exception:
            pass

        return response

    def __getattr__(self, name: str) -> Any:
        return getattr(self._real, name)


class _ChatProxy:
    """Proxies client.chat to expose .completions."""

    def __init__(self, real_chat, memory: Memory, recall_limit: int):
        self._real = real_chat
        self.completions = _CompletionsProxy(real_chat.completions, memory, recall_limit)

    def __getattr__(self, name: str) -> Any:
        return getattr(self._real, name)


# ---------------------------------------------------------------------------
# Memory injection helper (shared across all paths)
# ---------------------------------------------------------------------------

def _inject_memory_message(memory: Memory, query: str, messages: list, limit: int) -> list:
    """Prepend a memory context system message to the messages list."""
    if not query:
        return messages

    results = memory.search_hybrid(query, limit=limit)
    if not results:
        return messages

    context_lines = "\n".join(
        f"[{r.get('memory_type', 'memory').upper()}] {r['content']}"
        for r in results
    )
    system_msg = {
        "role": "system",
        "content": (
            "Long-term memory context (use this to inform your response):\n\n"
            f"{context_lines}"
        ),
    }

    # Keep any existing system messages, prepend the memory system message
    return [system_msg] + list(messages)


# ---------------------------------------------------------------------------
# Main NSN Wrapper
# ---------------------------------------------------------------------------

class NSN:
    """
    Universal NeuroSleepNet model wrapper.

    Wraps any LLM, pipeline, or callable with persistent cognitive memory.
    The wrapped object is a fully transparent proxy — all original attributes
    and method calls are forwarded to the underlying model unchanged, except
    for call interception points where memory is injected.

    Args:
        model: Any LLM client, pipeline, or callable to wrap.
        namespace: Memory namespace (isolates this agent's memories).
        db_path: Path to the SQLite memory database.
        recall_limit: Number of memories to inject per call.
        auto_observe_inputs: Whether to store user inputs in memory.
        auto_observe_outputs: Whether to store model outputs in memory.

    Example with OpenAI::

        import neurosleepnet
        from openai import OpenAI

        client = OpenAI()
        client = neurosleepnet.NSN(client, namespace="my_agent")

        response = client.chat.completions.create(
            model="gpt-4o",
            messages=[{"role": "user", "content": "Who is Alice?"}]
        )

    Example with a custom callable::

        import neurosleepnet

        def my_model(prompt):
            return f"Response to: {prompt}"

        model = neurosleepnet.NSN(my_model, namespace="demo")
        result = model("Who leads the project?")

    Example with LangChain::

        import neurosleepnet
        from langchain_openai import ChatOpenAI

        llm = ChatOpenAI()
        llm = neurosleepnet.NSN(llm, namespace="langchain_agent")
        response = llm.invoke("Who leads the project?")
    """

    def __init__(
        self,
        model: Any,
        namespace: str = "default",
        db_path: str = "neurosleepnet.db",
        recall_limit: int = 5,
        auto_observe_inputs: bool = True,
        auto_observe_outputs: bool = True,
    ):
        # Use object.__setattr__ to avoid triggering our __setattr__ override
        object.__setattr__(self, "_model", model)
        object.__setattr__(self, "_memory", Memory(namespace=namespace, db_path=db_path))
        object.__setattr__(self, "_recall_limit", recall_limit)
        object.__setattr__(self, "_auto_observe_inputs", auto_observe_inputs)
        object.__setattr__(self, "_auto_observe_outputs", auto_observe_outputs)

        # Detect and set up OpenAI-style chat proxy if applicable
        chat_proxy = None
        if hasattr(model, "chat") and hasattr(getattr(model, "chat", None), "completions"):
            chat_proxy = _ChatProxy(model.chat, self._memory, recall_limit)
        object.__setattr__(self, "_chat_proxy", chat_proxy)

        logger.debug(f"NSN wrapped {type(model).__name__} | namespace={namespace!r}")

    # ------------------------------------------------------------------
    # Transparent proxy: forward all attribute access to the real model
    # ------------------------------------------------------------------

    def __getattr__(self, name: str) -> Any:
        # Return the chat proxy for OpenAI-style clients
        if name == "chat":
            proxy = object.__getattribute__(self, "_chat_proxy")
            if proxy is not None:
                return proxy
        # Forward everything else to the real model
        model = object.__getattribute__(self, "_model")
        return getattr(model, name)

    def __setattr__(self, name: str, value: Any) -> None:
        model = object.__getattribute__(self, "_model")
        setattr(model, name, value)

    # ------------------------------------------------------------------
    # Generic callable interception (HuggingFace pipelines, custom models)
    # ------------------------------------------------------------------

    def __call__(self, input_: Any = None, *args, **kwargs) -> Any:
        memory = object.__getattribute__(self, "_memory")
        model = object.__getattribute__(self, "_model")
        recall_limit = object.__getattribute__(self, "_recall_limit")
        auto_in = object.__getattribute__(self, "_auto_observe_inputs")
        auto_out = object.__getattribute__(self, "_auto_observe_outputs")

        # Extract text query from various input formats
        query = ""
        if isinstance(input_, str):
            query = input_
        elif isinstance(input_, dict):
            query = input_.get("input", input_.get("query", input_.get("text", "")))
        elif isinstance(input_, list) and input_:
            last = input_[-1]
            if isinstance(last, dict):
                query = last.get("content", "")

        # Auto-store user input
        if auto_in and query:
            memory.observe(query, source="user_input")

        # Detect LangChain-style .invoke() models and use the right method
        if hasattr(model, "invoke") and callable(getattr(model, "invoke")):
            # Build memory-enriched prompt for string-based LangChain LLMs
            if isinstance(input_, str):
                results = memory.search_hybrid(query, limit=recall_limit)
                if results:
                    context = "\n".join(f"- {r['content']}" for r in results)
                    enriched_input = f"Memory context:\n{context}\n\nUser: {input_}"
                    output = model.invoke(enriched_input, *args, **kwargs)
                else:
                    output = model.invoke(input_, *args, **kwargs)
            else:
                output = model.invoke(input_, *args, **kwargs)
        else:
            # Generic callable: just call the model directly
            output = model(input_, *args, **kwargs) if input_ is not None else model(*args, **kwargs)

        # Auto-store model output
        if auto_out and output:
            out_text = output if isinstance(output, str) else str(output)
            memory.observe(out_text[:500], source="llm")

        return output

    # ------------------------------------------------------------------
    # Direct memory access on the wrapper
    # ------------------------------------------------------------------

    @property
    def memory(self) -> Memory:
        """Direct access to the underlying Memory instance."""
        return object.__getattribute__(self, "_memory")

    def remember(self, text: str, source: str = "agent") -> ObserveResult:
        """Manually store a fact into long-term memory."""
        return self.memory.observe(text, source=source)

    def recall(self, query: str, limit: int = 5) -> list:
        """Manually retrieve relevant memories for a query."""
        return self.memory.search_hybrid(query, limit=limit)

    def sleep(self) -> bool:
        """Trigger the offline NREM/REM/Decay consolidation cycle."""
        return self.memory.trigger_sleep()

    def timeline(self, memory_type: str = None, limit: int = 20) -> list:
        """Return chronological memory entries."""
        return self.memory.timeline(memory_type=memory_type, limit=limit)

    def __repr__(self) -> str:
        model = object.__getattribute__(self, "_model")
        memory = object.__getattribute__(self, "_memory")
        return f"NSN(model={type(model).__name__}, namespace={memory.namespace!r})"


# ---------------------------------------------------------------------------
# Convenience function
# ---------------------------------------------------------------------------

def wrap(
    model: Any,
    namespace: str = "default",
    db_path: str = "neurosleepnet.db",
    **kwargs,
) -> NSN:
    """
    Wrap any model with NeuroSleepNet persistent memory.

    Equivalent to ``NSN(model, namespace=namespace, db_path=db_path)``.

    Example::

        import neurosleepnet
        model = neurosleepnet.wrap(my_llm, namespace="agent_1")
    """
    return NSN(model, namespace=namespace, db_path=db_path, **kwargs)

# Alias for wrap
init = wrap
