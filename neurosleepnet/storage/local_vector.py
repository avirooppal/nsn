import os
import json
import numpy as np
import faiss
from typing import List, Dict, Any
from .vector import VectorStore
from .base import StorageAdapter

class FAISSVectorStore(VectorStore):
    """
    FAISS-based implementation of VectorStore using IndexFlatIP.
    """
    def __init__(self, storage: StorageAdapter, index_path: str = "neurosleepnet.faiss", ids_path: str = "neurosleepnet.faiss_ids"):
        self.storage = storage
        self.index_path = index_path
        self.ids_path = ids_path
        self.dimension = 384
        self._id_map = []
        
        self.index = faiss.IndexFlatIP(self.dimension)
        
        if os.path.exists(self.index_path) and os.path.exists(self.ids_path):
            try:
                self.index = faiss.read_index(self.index_path)
                with open(self.ids_path, 'r', encoding='utf-8') as f:
                    self._id_map = json.load(f)
            except Exception:
                self._build_from_storage()
                self._persist()
        else:
            self._build_from_storage()
            self._persist()

    def _build_from_storage(self):
        records = self.storage.list()
        self.index = faiss.IndexFlatIP(self.dimension)
        self._id_map = []
        
        records_with_embeddings = [r for r in records if r.get('embedding') and len(r['embedding']) == self.dimension]
        if not records_with_embeddings:
            return
            
        embeddings_matrix = np.array([r['embedding'] for r in records_with_embeddings], dtype=np.float32)
        faiss.normalize_L2(embeddings_matrix)
        self.index.add(embeddings_matrix)
        self._id_map = [r['id'] for r in records_with_embeddings]

    def _persist(self):
        faiss.write_index(self.index, self.index_path)
        with open(self.ids_path, 'w', encoding='utf-8') as f:
            json.dump(self._id_map, f)

    def add(self, memory_id: str, embedding: List[float]):
        vec = np.array([embedding], dtype=np.float32)
        if vec.shape[1] != self.dimension:
            return
        faiss.normalize_L2(vec)
        self.index.add(vec)
        self._id_map.append(memory_id)
        self._persist()

    def search(self, query_embedding: List[float], limit: int = 5) -> List[Dict[str, Any]]:
        if self.index.ntotal == 0:
            return []
            
        query_vec = np.array([query_embedding], dtype=np.float32)
        if query_vec.shape[1] != self.dimension:
            return []
            
        faiss.normalize_L2(query_vec)
        
        k = min(limit, self.index.ntotal)
        distances, indices = self.index.search(query_vec, k)
        
        results = []
        for i in range(k):
            idx = int(indices[0][i])
            if idx < 0 or idx >= len(self._id_map):
                continue
            memory_id = self._id_map[idx]
            score = float(distances[0][i])
            
            record = self.storage.get(memory_id)
            if record:
                record['score'] = score
                results.append(record)
                
        return results
