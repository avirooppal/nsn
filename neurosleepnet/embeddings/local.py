import functools
import threading
from typing import List
from .base import EmbeddingProvider

_MODEL_CACHE = {}
_EMBED_CACHE_SIZE = 2048  # Phase 1: expanded LRU cache for query embeddings


class LocalEmbeddingProvider(EmbeddingProvider):
    """
    Local embedding provider using sentence-transformers, with lazy loading
    and an in-process LRU cache to avoid re-embedding identical content.

    Phase 1 Optimization:
      - Expanded LRU cache (2048 entries) for token-level embeddings
      - Dedicated query-level mean-pooled embedding cache (LRU 2048)
        for fast FAISS-compatible vector lookups without recomputation
      - Thread-safe model access for parallel ThreadPoolExecutor calls
    """

    def __init__(self, model_name: str = "all-MiniLM-L6-v2"):
        self.model_name = model_name
        self._model_lock = threading.Lock()
        # Bound LRU cache per instance for token-level embeddings
        self._embed_cached = functools.lru_cache(maxsize=_EMBED_CACHE_SIZE)(self._embed_raw)
        # Phase 1: Separate LRU cache for mean-pooled query vectors (FAISS-ready)
        self._query_embed_cached = functools.lru_cache(maxsize=_EMBED_CACHE_SIZE)(self._query_embed_raw)

    @property
    def model(self):
        if self.model_name not in _MODEL_CACHE:
            with self._model_lock:
                # Double-checked locking
                if self.model_name not in _MODEL_CACHE:
                    from sentence_transformers import SentenceTransformer
                    _MODEL_CACHE[self.model_name] = SentenceTransformer(self.model_name)
        return _MODEL_CACHE[self.model_name]

    def _embed_raw(self, text: str) -> tuple:
        """Returns a tuple (hashable) for caching. Callers convert to list."""
        out = self.model.encode(text, output_value='token_embeddings')
        return tuple(tuple(t) for t in out.tolist())

    def _query_embed_raw(self, text: str) -> tuple:
        """
        Phase 1: Returns a single mean-pooled sentence embedding vector (hashable tuple).
        Used for FAISS search — avoids token-level overhead for retrieval.
        """
        import numpy as np
        out = self.model.encode(text, output_value='sentence_embedding')
        if hasattr(out, 'tolist'):
            return tuple(out.tolist())
        return tuple(float(x) for x in out)

    def embed(self, text: str) -> List[List[float]]:
        return [list(t) for t in self._embed_cached(text)]

    def embed_query(self, text: str) -> List[float]:
        """
        Phase 1: Returns a mean-pooled embedding vector for query retrieval.
        Cached via LRU — subsequent identical queries have 0ms embedding cost.
        """
        return list(self._query_embed_cached(text))

    def embed_batch(self, texts: List[str]) -> List[List[List[float]]]:
        # For batch, use the model directly (more efficient than individual cached calls)
        out = self.model.encode(texts, output_value='token_embeddings')
        if isinstance(out, list):
            return [t.tolist() for t in out]
        return out.tolist()
