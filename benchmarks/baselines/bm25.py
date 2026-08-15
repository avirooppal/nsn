"""
BM25 / SQLite FTS5 Baseline System using NSN's underlying SQLiteAdapter.
"""
import time
import re
import os
from benchmarks.baselines.base import BaseSystem
from neurosleepnet.storage.sqlite import SQLiteAdapter

def tokenize(text: str):
    return re.findall(r'\w+', text.lower())

class BM25System(BaseSystem):
    def __init__(self, db_path: str = "bench_bm25.db", name: str = "bm25"):
        super().__init__(name)
        self.db_path = db_path
        if os.path.exists(db_path):
            os.remove(db_path)
        self.storage = SQLiteAdapter(db_path=db_path)

    def observe(self, content: str, source: str = "agent", metadata: dict = None):
        import uuid, datetime
        mem_id = str(uuid.uuid4())
        self.storage.store(
            memory_id=mem_id,
            content=content,
            created_at=datetime.datetime.utcnow().isoformat(),
            namespace="bm25_bench"
        )
        return mem_id

    def query(self, question: str, limit: int = 5) -> dict:
        t0 = time.perf_counter()
        results = self.storage.search_keyword(question, limit=limit, namespace="bm25_bench")
        latency_ms = (time.perf_counter() - t0) * 1000.0
        
        retrieved_ids = [r["id"] for r in results]
        retrieved_scores = [1.0] * len(results)
        
        context_str = "\n".join([r["content"] for r in results])
        prompt_tokens = len(tokenize(context_str + " " + question))
        
        answer = results[0]["content"] if results else "INSUFFICIENT MEMORY"
        
        return {
            "answer": answer,
            "retrieved_ids": retrieved_ids,
            "retrieved_scores": retrieved_scores,
            "retrieved_records": results,
            "prompt_tokens": prompt_tokens,
            "latency_ms": latency_ms
        }

    def reset(self):
        if os.path.exists(self.db_path):
            os.remove(self.db_path)
        self.storage = SQLiteAdapter(db_path=self.db_path)
