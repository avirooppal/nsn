"""
NSN Memory Adapter supporting component ablations and complete feature integration.
"""
import time
import re
import os
from benchmarks.baselines.base import BaseSystem
from neurosleepnet.sdk.memory import Memory

def tokenize(text: str):
    return re.findall(r'\w+', text.lower())

class NSNSystem(BaseSystem):
    def __init__(self, db_path: str = "bench_nsn.db", namespace: str = "nsn_bench", name: str = "nsn", ablation_mode: str = None):
        super().__init__(name)
        self.db_path = db_path
        self.namespace = namespace
        self.ablation_mode = ablation_mode
        self._cleanup()
        self.memory = Memory(namespace=namespace, db_path=db_path)

    def _cleanup(self):
        for p in [self.db_path, self.db_path.replace(".db", ".faiss"), self.db_path.replace(".db", ".faiss_ids")]:
            if os.path.exists(p):
                os.remove(p)

    def observe(self, content: str, source: str = "agent", metadata: dict = None):
        res = self.memory.observe(content, source=source, metadata=metadata)
        return res.memory_id if res.stored else None

    def query(self, question: str, limit: int = 5) -> dict:
        t0 = time.perf_counter()
        
        # Determine retrieval mode based on ablation_mode
        if self.ablation_mode == "semantic_only" or self.ablation_mode == "nsn_no_keyword":
            results = self.memory.search(question, limit=limit)
        elif self.ablation_mode == "keyword_only" or self.ablation_mode == "nsn_no_semantic":
            results = self.memory.search_keyword(question, limit=limit)
        elif self.ablation_mode == "graph_only":
            results = self.memory.search_graph(question, limit=limit)
        elif self.ablation_mode == "nsn_no_graph":
            results = self.memory.search_hybrid(question, limit=limit, graph_weight=0.0)
        elif self.ablation_mode == "nsn_no_rrf":
            results = self.memory.search(question, limit=limit)
        else:
            # Full NSN RRF hybrid search
            results = self.memory.search_hybrid(question, limit=limit, adaptive_k=True)
            
        latency_ms = (time.perf_counter() - t0) * 1000.0
        
        retrieved_ids = [r["id"] for r in results]
        retrieved_scores = [r.get("rerank_score", r.get("hybrid_score", r.get("score", 0.0))) for r in results]
        
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

    def sleep(self):
        return self.memory.trigger_sleep()

    def reset(self):
        self._cleanup()
        self.memory = Memory(namespace=self.namespace, db_path=self.db_path)
