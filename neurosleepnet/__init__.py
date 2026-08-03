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

from .sdk.memory import Memory
from .sdk.async_memory import AsyncMemory
from .sdk.wrapper import NSN, wrap

__version__ = "0.3.0"
__all__ = ["Memory", "AsyncMemory", "NSN", "wrap", "__version__"]
