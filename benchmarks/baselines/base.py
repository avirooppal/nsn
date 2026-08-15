"""
Base class for baseline systems.
"""
from abc import ABC, abstractmethod

class BaseSystem(ABC):
    def __init__(self, name: str):
        self.name = name

    @abstractmethod
    def observe(self, content: str, source: str = "agent", metadata: dict = None):
        pass

    @abstractmethod
    def query(self, question: str, limit: int = 5) -> dict:
        """
        Returns dict with keys:
          - answer: str
          - retrieved_ids: list
          - retrieved_scores: list
          - prompt_tokens: int
          - latency_ms: float
        """
        pass

    def sleep(self):
        pass

    def reset(self):
        pass
