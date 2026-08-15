"""
Hybrid RAG Baseline System (FTS5 + FAISS Vector search without NSN Graph or Sleep).
"""
import time
import re
import os
import json
import uuid, datetime
from benchmarks.baselines.base import BaseSystem
from neurosleepnet.storage.sqlite import SQLiteAdapter
from neurosleepnet.storage.local_vector import TieredVectorStore
from neurosleepnet.embeddings.local import LocalEmbeddingProvider

def tokenize(text: str):
    return re.findall(r'\w+', text.lower())

class HybridRAGSystem(BaseSystem):
    def __init__(self, db_path: str = "bench_hybrid.db", name: str = "hybrid_rag"):
        super().__init__(name)
        self.db_path = db_path
        self._cleanup()
        self.storage = SQLiteAdapter(db_path=db_path)
        self.embedder = LocalEmbeddingProvider()
        self.vector_store = TieredVectorStore(
            self.storage,
            index_path=db_path.replace(".db", ".faiss"),
            ids_path=db_path.replace(".db", ".faiss_ids")
        )

    def _cleanup(self):
        for p in [self.db_path, self.db_path.replace(".db", ".faiss"), self.db_path.replace(".db", ".faiss_ids")]:
            if os.path.exists(p):
                os.remove(p)

    def observe(self, content: str, source: str = "agent", metadata: dict = None):
        mem_id = str(uuid.uuid4())
        emb = self.embedder.embed(content)
        self.storage.store(
            memory_id=mem_id,
            content=content,
            created_at=datetime.datetime.utcnow().isoformat(),
            embedding=json.dumps(emb),
            namespace="hybrid_bench"
        )
        self.vector_store.add(mem_id, emb)
        return mem_id

    def query(self, question: str, limit: int = 5) -> dict:
        t0 = time.perf_counter()
        
        # Dense
        q_emb = self.embedder.embed(question)
        dense_res = self.vector_store.search(q_emb, limit=limit*2)
        
        # Keyword
        keyword_res = self.storage.search_keyword(question, limit=limit*2, namespace="hybrid_bench")
        
        # RRF Fusion
        rrf_scores = {}
        all_recs = {}
        for rank, r in enumerate(dense_res):
            id_ = r["id"]
            all_recs[id_] = r
            rrf_scores[id_] = rrf_scores.get(id_, 0.0) + 1.0 / (60 + rank + 1)
            
        for rank, r in enumerate(keyword_res):
            id_ = r["id"]
            if id_ not in all_recs: all_recs[id_] = r
            rrf_scores[id_] = rrf_scores.get(id_, 0.0) + 1.0 / (60 + rank + 1)
            
        sorted_ids = sorted(rrf_scores.keys(), key=lambda x: rrf_scores[x], reverse=True)[:limit]
        final_results = [all_recs[i] for i in sorted_ids]
        
        latency_ms = (time.perf_counter() - t0) * 1000.0
        
        retrieved_ids = [r["id"] for r in final_results]
        retrieved_scores = [rrf_scores[i] for i in sorted_ids]
        
        context_str = "\n".join([r["content"] for r in final_results])
        prompt_tokens = len(tokenize(context_str + " " + question))
        
        answer = final_results[0]["content"] if final_results else "INSUFFICIENT MEMORY"
        
        return {
            "answer": answer,
            "retrieved_ids": retrieved_ids,
            "retrieved_scores": retrieved_scores,
            "retrieved_records": final_results,
            "prompt_tokens": prompt_tokens,
            "latency_ms": latency_ms
        }

    def reset(self):
        self._cleanup()
        self.storage = SQLiteAdapter(db_path=self.db_path)
        self.vector_store = TieredVectorStore(
            self.storage,
            index_path=self.db_path.replace(".db", ".faiss"),
            ids_path=self.db_path.replace(".db", ".faiss_ids")
        )
