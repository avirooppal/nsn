import os
import json
import numpy as np
import faiss
from typing import List, Dict, Any
from .vector import VectorStore
from .base import StorageAdapter
import logging

logger = logging.getLogger("neurosleepnet.vector")

class TieredVectorStore(VectorStore):
    """
    Tiered FAISS-based Vector Store using ColBERT-style late interaction.
    - Recent Buffer: HNSW (fast ANN)
    - Long-term Store: IVFPQ (compressed)
    """
    def __init__(self, storage: StorageAdapter, index_path: str = "neurosleepnet.faiss", ids_path: str = "neurosleepnet.faiss_ids"):
        self.storage = storage
        self.base_path = index_path.replace('.faiss', '')
        self.hnsw_path = f"{self.base_path}_hnsw.faiss"
        self.ivfpq_path = f"{self.base_path}_ivfpq.faiss"
        self.ids_path = ids_path
        self.dimension = 384
        
        self._next_id = 0
        self._id_map = {} # int -> str (memory_id)
        
        self.buffer_limit = 5000
        
        self.hnsw_index = faiss.IndexIDMap(faiss.IndexHNSWFlat(self.dimension, 32, faiss.METRIC_INNER_PRODUCT))
        self.ivfpq_index = None
        
        self._load()

    def _load(self):
        if os.path.exists(self.ids_path):
            with open(self.ids_path, 'r', encoding='utf-8') as f:
                saved_map = json.load(f)
                self._id_map = {int(k): v for k, v in saved_map.items()}
                if self._id_map:
                    self._next_id = max(self._id_map.keys()) + 1
                    
        if os.path.exists(self.hnsw_path):
            try:
                self.hnsw_index = faiss.read_index(self.hnsw_path)
            except Exception as e:
                logger.error(f"Failed to load HNSW index: {e}")
                
        if os.path.exists(self.ivfpq_path):
            try:
                self.ivfpq_index = faiss.read_index(self.ivfpq_path)
            except Exception as e:
                logger.error(f"Failed to load IVFPQ index: {e}")
                
        if self.hnsw_index.ntotal == 0 and (self.ivfpq_index is None or self.ivfpq_index.ntotal == 0):
            self._build_from_storage()

    def _build_from_storage(self):
        records = self.storage.list()
        self.hnsw_index = faiss.IndexIDMap(faiss.IndexHNSWFlat(self.dimension, 32, faiss.METRIC_INNER_PRODUCT))
        self.ivfpq_index = None
        self._id_map = {}
        self._next_id = 0
        
        for r in records:
            emb = r.get('embedding')
            if emb and isinstance(emb, list) and len(emb) > 0 and isinstance(emb[0], list):
                self.add(r['id'], emb, persist=False)
        self._persist()

    def _persist(self):
        faiss.write_index(self.hnsw_index, self.hnsw_path)
        if self.ivfpq_index is not None:
            faiss.write_index(self.ivfpq_index, self.ivfpq_path)
        with open(self.ids_path, 'w', encoding='utf-8') as f:
            json.dump(self._id_map, f)

    def _build_from_storage_ivfpq(self):
        """Rebuilds the entire store, putting everything into IVFPQ."""
        records = self.storage.list()
        all_vecs = []
        all_ids = []
        
        temp_id_map = {}
        next_id = 0
        
        for r in records:
            emb = r.get('embedding')
            if emb and isinstance(emb, list) and len(emb) > 0 and isinstance(emb[0], list):
                for token_vec in emb:
                    if len(token_vec) == self.dimension:
                        all_vecs.append(token_vec)
                        all_ids.append(next_id)
                        temp_id_map[next_id] = r['id']
                        next_id += 1
                        
        if len(all_vecs) < 1000:
            return # Don't train IVFPQ yet
            
        vec_matrix = np.array(all_vecs, dtype=np.float32)
        faiss.normalize_L2(vec_matrix)
        id_array = np.array(all_ids, dtype=np.int64)
        
        nlist = min(100, max(1, len(all_vecs) // 39))
        quantizer = faiss.IndexFlatIP(self.dimension)
        ivfpq = faiss.IndexIVFPQ(quantizer, self.dimension, nlist, 8, 8)
        
        ivfpq.train(vec_matrix)
        self.ivfpq_index = faiss.IndexIDMap(ivfpq)
        self.ivfpq_index.add_with_ids(vec_matrix, id_array)
        self.hnsw_index = faiss.IndexIDMap(faiss.IndexHNSWFlat(self.dimension, 32, faiss.METRIC_INNER_PRODUCT))
        
        self._id_map = temp_id_map
        self._next_id = next_id
        self._persist()

    def add(self, memory_id: str, embedding: List[List[float]], persist: bool = True):
        if not embedding or not isinstance(embedding[0], list):
            return
            
        vecs = []
        ids = []
        for token_vec in embedding:
            if len(token_vec) == self.dimension:
                vecs.append(token_vec)
                ids.append(self._next_id)
                self._id_map[self._next_id] = memory_id
                self._next_id += 1
                
        if not vecs:
            return
            
        vec_matrix = np.array(vecs, dtype=np.float32)
        faiss.normalize_L2(vec_matrix)
        id_array = np.array(ids, dtype=np.int64)
        
        self.hnsw_index.add_with_ids(vec_matrix, id_array)
        
        if persist:
            if self.hnsw_index.ntotal > self.buffer_limit:
                self._build_from_storage_ivfpq()
            else:
                self._persist()

    def search(self, query_embedding: List[List[float]], limit: int = 5) -> List[Dict[str, Any]]:
        if not query_embedding or not isinstance(query_embedding[0], list):
            return []
            
        query_matrix = np.array(query_embedding, dtype=np.float32)
        if query_matrix.shape[1] != self.dimension:
            return []
            
        faiss.normalize_L2(query_matrix)
        
        k = min(20, self.hnsw_index.ntotal + (self.ivfpq_index.ntotal if self.ivfpq_index else 0))
        if k == 0:
            return []
            
        candidate_memory_ids = set()
        
        if self.hnsw_index.ntotal > 0:
            distances, indices = self.hnsw_index.search(query_matrix, min(k, self.hnsw_index.ntotal))
            for token_indices in indices:
                for idx in token_indices:
                    if idx != -1 and idx in self._id_map:
                        candidate_memory_ids.add(self._id_map[idx])
                        
        if self.ivfpq_index and self.ivfpq_index.ntotal > 0:
            distances, indices = self.ivfpq_index.search(query_matrix, min(k, self.ivfpq_index.ntotal))
            for token_indices in indices:
                for idx in token_indices:
                    if idx != -1 and idx in self._id_map:
                        candidate_memory_ids.add(self._id_map[idx])
                        
        results = []
        for mem_id in candidate_memory_ids:
            record = self.storage.get(mem_id)
            if not record or not record.get('embedding'):
                continue
                
            doc_embedding = record['embedding']
            if not isinstance(doc_embedding, list) or not isinstance(doc_embedding[0], list):
                continue
                
            doc_matrix = np.array(doc_embedding, dtype=np.float32)
            if doc_matrix.shape[1] != self.dimension:
                continue
                
            faiss.normalize_L2(doc_matrix)
            
            similarities = np.dot(query_matrix, doc_matrix.T)
            max_sims = np.max(similarities, axis=1)
            score = float(np.sum(max_sims)) / len(query_matrix)
            
            record['score'] = score
            results.append(record)
            
        results.sort(key=lambda x: x['score'], reverse=True)
        return results[:limit]
