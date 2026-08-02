from typing import List
from .base import EmbeddingProvider

_MODEL_CACHE = {}

class LocalEmbeddingProvider(EmbeddingProvider):
    """
    Local embedding provider using sentence-transformers, with lazy loading.
    """
    def __init__(self, model_name: str = "all-MiniLM-L6-v2"):
        self.model_name = model_name

    @property
    def model(self):
        if self.model_name not in _MODEL_CACHE:
            from sentence_transformers import SentenceTransformer
            _MODEL_CACHE[self.model_name] = SentenceTransformer(self.model_name)
        return _MODEL_CACHE[self.model_name]

    def embed(self, text: str) -> List[float]:
        return self.model.encode(text).tolist()

    def embed_batch(self, texts: List[str]) -> List[List[float]]:
        return self.model.encode(texts).tolist()
