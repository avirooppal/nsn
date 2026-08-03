import functools
from typing import List
from .base import EmbeddingProvider

_MODEL_CACHE = {}


class LocalEmbeddingProvider(EmbeddingProvider):
    """
    Local embedding provider using sentence-transformers, with lazy loading
    and an in-process LRU cache to avoid re-embedding identical content.
    """

    def __init__(self, model_name: str = "all-MiniLM-L6-v2"):
        self.model_name = model_name
        # Bound LRU cache per instance — avoids paying model cost twice
        # (DuplicateDetector embeds, then observe() embeds the same string again)
        self._embed_cached = functools.lru_cache(maxsize=512)(self._embed_raw)

    @property
    def model(self):
        if self.model_name not in _MODEL_CACHE:
            from sentence_transformers import SentenceTransformer
            _MODEL_CACHE[self.model_name] = SentenceTransformer(self.model_name)
        return _MODEL_CACHE[self.model_name]

    def _embed_raw(self, text: str) -> tuple:
        """Returns a tuple (hashable) for caching. Callers convert to list."""
        return tuple(self.model.encode(text).tolist())

    def embed(self, text: str) -> List[float]:
        return list(self._embed_cached(text))

    def embed_batch(self, texts: List[str]) -> List[List[float]]:
        # For batch, use the model directly (more efficient than individual cached calls)
        return self.model.encode(texts).tolist()
