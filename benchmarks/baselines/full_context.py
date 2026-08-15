"""
Standard Conversation Buffer Memory Baseline (passes raw chat history).
"""
import time
import re
from benchmarks.baselines.base import BaseSystem

def tokenize(text: str):
    return re.findall(r'\w+', text.lower())

class FullContextSystem(BaseSystem):
    def __init__(self, name: str = "full_context"):
        super().__init__(name)
        self.history = []

    def observe(self, content: str, source: str = "agent", metadata: dict = None):
        self.history.append({"source": source, "content": content})

    def query(self, question: str, limit: int = 5) -> dict:
        t0 = time.perf_counter()
        
        # Build prompt from entire history
        full_text = "\n".join([f"{h['source'].upper()}: {h['content']}" for h in self.history])
        q_lower = question.lower()
        
        # Deterministic extraction logic for baseline evaluation
        answer = "INSUFFICIENT MEMORY"
        for h in reversed(self.history):
            content = h["content"]
            # Basic keyword match fallback
            words = [w for w in tokenize(q_lower) if len(w) > 3]
            if any(w in content.lower() for w in words):
                answer = content
                break

        latency_ms = (time.perf_counter() - t0) * 1000.0
        prompt_tokens = len(tokenize(full_text + " " + question))
        
        return {
            "answer": answer,
            "retrieved_ids": [f"hist_{i}" for i in range(len(self.history))],
            "retrieved_scores": [1.0] * len(self.history),
            "prompt_tokens": prompt_tokens,
            "latency_ms": latency_ms
        }

    def reset(self):
        self.history = []
