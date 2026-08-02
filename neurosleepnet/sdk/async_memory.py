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
