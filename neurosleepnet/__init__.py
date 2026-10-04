"""
NeuroSleepNet — Cognitive Memory Operating System for SLMs & AI Agents.

Quick start::

    from neurosleepnet import Memory
    m = Memory()
    m.observe("Alice leads the engineering team.")
    results = m.search_hybrid("Who leads engineering?")

Integrations::

    from neurosleepnet.integrations.tool import MemoryTool          # Any agent
    from neurosleepnet.integrations.openai_adapter import MemoryInjector  # OpenAI / GPT
    from neurosleepnet.integrations.langchain import NeurosleepNetHistory  # LangChain
    from neurosleepnet.integrations.api import create_app            # FastAPI REST
"""

__version__ = "0.3.0"
__all__ = ["Memory", "AsyncMemory", "NSN", "wrap", "init", "Runtime", "flush", "close", "__version__"]

def __getattr__(name: str):
    if name == "Memory":
        from .sdk.memory import Memory
        return Memory
    if name == "AsyncMemory":
        from .sdk.async_memory import AsyncMemory
        return AsyncMemory
    if name == "NSN":
        from .sdk.wrapper import NSN
        return NSN
    if name in ("init", "wrap", "Runtime", "flush", "close"):
        import nsn
        return getattr(nsn, name)
    raise AttributeError(f"module 'neurosleepnet' has no attribute '{name}'")

