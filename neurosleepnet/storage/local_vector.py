import numpy as np
from typing import List, Dict, Any
from .vector import VectorStore
from .base import StorageAdapter

class LocalVectorStore(VectorStore):
    """
    Local implementation of VectorStore that computes cosine similarity 
    in memory using numpy.
    """
    def __init__(self, storage: StorageAdapter):
        self.storage = storage

    def search(self, query_embedding: List[float], limit: int = 5) -> List[Dict[str, Any]]:
        records = self.storage.list()
        
        records_with_embeddings = [r for r in records if r.get('embedding')]
        if not records_with_embeddings:
            return []

        query_vec = np.array(query_embedding)
        norm_q = np.linalg.norm(query_vec)
        
        results = []
        for r in records_with_embeddings:
            vec = np.array(r['embedding'])
            norm_v = np.linalg.norm(vec)
            
            if norm_q == 0 or norm_v == 0:
                sim = 0.0
            else:
                sim = np.dot(query_vec, vec) / (norm_q * norm_v)
                
            results.append((sim, r))
            
        results.sort(key=lambda x: x[0], reverse=True)
        
        top_k = []
        for sim, r in results[:limit]:
            r_copy = dict(r)
            r_copy['score'] = float(sim)
            top_k.append(r_copy)
            
        return top_k
