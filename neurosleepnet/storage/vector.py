from abc import ABC, abstractmethod
from typing import List, Dict, Any

class VectorStore(ABC):
    """
    Abstract interface for vector stores to perform semantic search.
    """
    @abstractmethod
    def search(self, query_embedding: List[float], limit: int = 5) -> List[Dict[str, Any]]:
        pass
