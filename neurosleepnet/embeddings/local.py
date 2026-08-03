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
        out = self.model.encode(text, output_value='token_embeddings')
        return tuple(tuple(t) for t in out.tolist())

    def embed(self, text: str) -> List[List[float]]:
        return [list(t) for t in self._embed_cached(text)]

    def embed_batch(self, texts: List[str]) -> List[List[List[float]]]:
        # For batch, use the model directly (more efficient than individual cached calls)
        out = self.model.encode(texts, output_value='token_embeddings')
        if isinstance(out, list):
            return [t.tolist() for t in out]
        return out.tolist()
