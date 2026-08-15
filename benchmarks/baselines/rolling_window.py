"""
Rolling Window Memory Baseline.

Represents a naive "LLM with memory" — keeps the last `window_size`
observations in a sliding buffer. When answering a query, it scans
the buffer for keyword overlap and returns the most recent matching content.

This is strictly weaker than NSN because:
  1. Old facts beyond the window are permanently lost (no persistent store)
  2. No semantic retrieval — pure keyword match on a tiny buffer
  3. No consolidation, no trust scoring, no graph structure
  4. Recall drops to 0% as soon as gold memory falls outside the window
"""

import time
import re
from collections import deque
from benchmarks.baselines.base import BaseSystem


def _tokenize(text: str) -> set:
    return set(re.findall(r'\b\w{3,}\b', text.lower()))


class RollingWindowSystem(BaseSystem):
    """
    Sliding-window "LLM" — the naive memory baseline.

    Models a production LLM that simply keeps recent context in its
    prompt window. Once the window is full, older memories are evicted.
    No semantic vector search. No graph. No sleep.
    """

    def __init__(self, window_size: int = 10, name: str = "rolling_window"):
        super().__init__(name)
        self.window_size = window_size
        self._buffer: deque = deque(maxlen=window_size)   # (id, content)
        self._id_counter = 0

    def observe(self, content: str, source: str = "agent", metadata: dict = None):
        """Add observation to sliding buffer. Returns a sequential window ID."""
        wid = f"win_{self._id_counter}"
        self._buffer.append((wid, content))
        self._id_counter += 1
        return wid

    def query(self, question: str, limit: int = 5) -> dict:
        t0 = time.perf_counter()

        q_tokens = _tokenize(question)
        # Score each buffer entry by token overlap (recency is implicit via order)
        scored = []
        for (wid, content) in self._buffer:
            c_tokens = _tokenize(content)
            overlap = len(q_tokens & c_tokens)
            if overlap > 0:
                scored.append((overlap, wid, content))

        # Sort: highest overlap first, then most-recent (reverse buffer order)
        scored.sort(key=lambda x: x[0], reverse=True)

        retrieved_ids = [s[1] for s in scored[:limit]]
        retrieved_contents = [s[2] for s in scored[:limit]]

        # Answer = most relevant match; fallback to MOST RECENT buffer item
        if retrieved_contents:
            answer = retrieved_contents[0]
        elif self._buffer:
            _, answer = self._buffer[-1]   # most recent
        else:
            answer = "INSUFFICIENT MEMORY"

        latency_ms = (time.perf_counter() - t0) * 1000.0
        full_context = " ".join(c for _, c in self._buffer) + " " + question
        prompt_tokens = len(re.findall(r'\w+', full_context))

        return {
            "answer": answer,
            "retrieved_ids": retrieved_ids,
            "retrieved_scores": [s[0] / max(len(q_tokens), 1) for s in scored[:limit]],
            "prompt_tokens": prompt_tokens,
            "latency_ms": latency_ms,
        }

    def reset(self):
        self._buffer.clear()
        self._id_counter = 0
