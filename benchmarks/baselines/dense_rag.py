"""
Dense Vector RAG Baseline System using NSN's TieredVectorStore + LocalEmbeddingProvider.
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

class DenseRAGSystem(BaseSystem):
    def __init__(self, db_path: str = "bench_dense.db", name: str = "dense_rag"):
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
            namespace="dense_bench"
        )
        self.vector_store.add(mem_id, emb)
        return mem_id

    def query(self, question: str, limit: int = 5) -> dict:
        t0 = time.perf_counter()
        q_emb = self.embedder.embed(question)
        results = self.vector_store.search(q_emb, limit=limit)
        latency_ms = (time.perf_counter() - t0) * 1000.0
        
        retrieved_ids = [r["id"] for r in results]
        retrieved_scores = [r.get("score", 0.0) for r in results]
        
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
        self._cleanup()
        self.storage = SQLiteAdapter(db_path=self.db_path)
        self.vector_store = TieredVectorStore(
            self.storage,
            index_path=self.db_path.replace(".db", ".faiss"),
            ids_path=self.db_path.replace(".db", ".faiss_ids")
        )
