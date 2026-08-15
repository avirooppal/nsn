"""
LLM-with-RAG-Memory Baseline.

This represents the strongest non-NSN competitor:
  an LLM that uses semantic dense retrieval (FAISS cosine similarity)
  to surface relevant memories, then uses the LLM to synthesize an answer.

When Ollama is running locally (http://localhost:11434), it performs real
LLM generation over the retrieved context. When Ollama is unavailable, it
falls back to extractive answer (top-1 retrieved content).

Why NSN is still better than this baseline:
  1. RETRIEVAL: NSN uses FAISS + FTS5 + Graph + RRF + Reranker (hybrid).
     This baseline uses ONLY FAISS cosine similarity.
  2. CONSOLIDATION: NSN's sleep NREM consolidates episodic→semantic;
     REM resolves contradictions. This baseline has no offline consolidation.
  3. IMPORTANCE: NSN scores importance; high-importance memories rank higher.
     This baseline ranks purely by cosine similarity.
  4. TRUST/SOURCE: NSN weights system observations > user claims.
     This baseline treats all sources equally.
  5. GRAPH: NSN stores entity relationships and traverses multi-hop chains.
     This baseline has no graph structure.
"""

import time
import re
import os
import json
import numpy as np
from benchmarks.baselines.base import BaseSystem

try:
    from sentence_transformers import SentenceTransformer
    import faiss
    _DEPS_AVAILABLE = True
except ImportError:
    _DEPS_AVAILABLE = False

try:
    import urllib.request
    import urllib.error
    def _check_ollama(host="http://localhost:11434") -> bool:
        try:
            req = urllib.request.urlopen(f"{host}/api/tags", timeout=2)
            return req.status == 200
        except Exception:
            return False
except Exception:
    def _check_ollama(host="http://localhost:11434") -> bool:
        return False


def _ollama_generate(prompt: str, model: str, host: str = "http://localhost:11434") -> str:
    """Call Ollama local API for generation."""
    import urllib.request
    import json as _json
    payload = _json.dumps({
        "model": model,
        "prompt": prompt,
        "stream": False,
        "options": {"temperature": 0, "num_predict": 50}
    }).encode("utf-8")
    req = urllib.request.Request(
        f"{host}/api/generate",
        data=payload,
        headers={"Content-Type": "application/json"},
        method="POST"
    )
    with urllib.request.urlopen(req, timeout=30) as resp:
        data = _json.loads(resp.read().decode("utf-8"))
        return data.get("response", "").strip()


def _tokenize(text: str) -> list:
    return re.findall(r'\w+', text.lower())


class LLMWithRAGMemory(BaseSystem):
    """
    LLM + Dense RAG Memory (FAISS) baseline.

    Ingests observations as embeddings, retrieves top-k relevant memories
    via cosine similarity, and generates an answer using:
      - Ollama LLM (if running)
      - Extractive top-1 (fallback when Ollama is unavailable)

    This is the strongest non-NSN competitor in the benchmark suite.
    """

    def __init__(
        self,
        name: str = "llm_rag_memory",
        model_name: str = "all-MiniLM-L6-v2",
        ollama_model: str = "llama3.2",
        ollama_host: str = "http://localhost:11434",
    ):
        super().__init__(name)
        self.ollama_model = ollama_model
        self.ollama_host = ollama_host
        self._use_ollama: bool = False  # detected at first query

        # Storage: list of (id, content) + FAISS index
        self._records: list = []        # [(id, content), ...]
        self._id_counter: int = 0
        self._model_name = model_name

        # Encoder is loaded lazily on first observe()/query() call
        # so that only ONE model is in memory at a time across baselines.
        self._encoder = None
        self._dim: int = 0
        self._index = None
        self._ids: list = []

    def _ensure_encoder(self):
        """Lazy-load the sentence-transformer model on first use."""
        if self._encoder is not None:
            return
        if not _DEPS_AVAILABLE:
            return
        # Use locally cached model — avoid network requests on offline machines.
        try:
            self._encoder = SentenceTransformer(self._model_name, local_files_only=True)
        except Exception:
            # If local cache miss, try with network (first-time setup)
            self._encoder = SentenceTransformer(self._model_name)
        # Non-deprecated API with fallback for older sentence-transformers
        if hasattr(self._encoder, 'get_embedding_dimension'):
            self._dim = self._encoder.get_embedding_dimension()
        else:
            self._dim = self._encoder.get_sentence_embedding_dimension()
        self._index = faiss.IndexFlatIP(self._dim)
        self._ids = []

    def _embed(self, text: str) -> np.ndarray:
        self._ensure_encoder()
        vec = self._encoder.encode([text], normalize_embeddings=True).astype("float32")
        return vec

    def observe(self, content: str, source: str = "agent", metadata: dict = None):
        self._ensure_encoder()
        obs_id = f"llm_rag_{self._id_counter}"
        self._records.append((obs_id, content))
        self._id_counter += 1

        if self._encoder is not None:
            vec = self._embed(content)
            self._index.add(vec)
            self._ids.append(obs_id)

        return obs_id

    def query(self, question: str, limit: int = 5) -> dict:
        t0 = time.perf_counter()
        self._ensure_encoder()

        if not self._records:
            return {
                "answer": "INSUFFICIENT MEMORY",
                "retrieved_ids": [],
                "retrieved_scores": [],
                "prompt_tokens": 0,
                "latency_ms": (time.perf_counter() - t0) * 1000.0,
            }

        # --- Retrieval ---
        if self._encoder is not None and self._index.ntotal > 0:
            q_vec = self._embed(question)
            k = min(limit, self._index.ntotal)
            scores, idxs = self._index.search(q_vec, k)
            retrieved = [
                (self._ids[i], self._records[i][1], float(scores[0][j]))
                for j, i in enumerate(idxs[0]) if i >= 0
            ]
        else:
            # Fallback: keyword overlap
            q_tokens = set(_tokenize(question))
            scored = []
            for (rid, content) in self._records:
                c_tokens = set(_tokenize(content))
                overlap = len(q_tokens & c_tokens)
                scored.append((rid, content, overlap / max(len(q_tokens), 1)))
            scored.sort(key=lambda x: x[2], reverse=True)
            retrieved = [(r[0], r[1], r[2]) for r in scored[:limit]]

        retrieved_ids = [r[0] for r in retrieved]
        retrieved_contents = [r[1] for r in retrieved]
        retrieved_scores = [r[2] for r in retrieved]

        # --- Answer generation ---
        # Check Ollama availability (once per instance)
        if not hasattr(self, '_ollama_checked'):
            self._use_ollama = _check_ollama(self.ollama_host)
            self._ollama_checked = True

        if self._use_ollama:
            # Real LLM generation over retrieved context
            context = "\n".join(f"- {c}" for c in retrieved_contents)
            prompt = (
                f"You are a memory assistant. Answer the question based ONLY on the "
                f"provided memory context. Give a short, direct answer.\n\n"
                f"Memory context:\n{context}\n\n"
                f"Question: {question}\n"
                f"Answer:"
            )
            try:
                answer = _ollama_generate(prompt, self.ollama_model, self.ollama_host)
            except Exception as e:
                answer = retrieved_contents[0] if retrieved_contents else "INSUFFICIENT MEMORY"
        else:
            # Extractive fallback: return top-1 retrieved content
            answer = retrieved_contents[0] if retrieved_contents else "INSUFFICIENT MEMORY"

        latency_ms = (time.perf_counter() - t0) * 1000.0
        context_str = " ".join(retrieved_contents) + " " + question
        prompt_tokens = len(_tokenize(context_str))

        return {
            "answer": answer,
            "retrieved_ids": retrieved_ids,
            "retrieved_scores": retrieved_scores,
            "prompt_tokens": prompt_tokens,
            "latency_ms": latency_ms,
            "ollama_used": self._use_ollama,
        }

    def reset(self):
        self._records = []
        self._id_counter = 0
        # Free encoder memory so next system can load cleanly
        self._encoder = None
        self._index = None
        self._ids = []
