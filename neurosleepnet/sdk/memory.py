from neurosleepnet.config.settings import Settings
from neurosleepnet.storage.sqlite import SQLiteAdapter
from neurosleepnet.memory.schemas import MemoryRecord
from neurosleepnet.embeddings.local import LocalEmbeddingProvider
from neurosleepnet.storage.local_vector import LocalVectorStore
import json

class Memory:
    """
    Core Memory class for NeuroSleepNet.
    """
    def __init__(self):
        self.settings = Settings()
        
        # Storage
        if self.settings.backend == "sqlite":
            self.storage = SQLiteAdapter()
        else:
            raise NotImplementedError(f"Backend {self.settings.backend} not implemented")
            
        # Embedding
        self.embedder = LocalEmbeddingProvider()
        
        # Vector Store
        if getattr(self.settings, 'vector_store', 'local') == "local":
            self.vector_store = LocalVectorStore(self.storage)
        else:
            self.vector_store = LocalVectorStore(self.storage)

    def store(self, content: str, metadata: dict = None, importance: float = 0.0, trust_score: float = 0.5):
        embedding = self.embedder.embed(content)
        
        record = MemoryRecord(
            content=content, 
            metadata=metadata or {}, 
            importance=importance, 
            trust_score=trust_score,
            embedding=embedding
        )
        
        self.storage.store(
            memory_id=record.id,
            content=record.content,
            created_at=record.created_at,
            metadata=json.dumps(record.metadata),
            importance=record.importance,
            trust_score=record.trust_score,
            embedding=json.dumps(record.embedding)
        )
        return record.id

    def get(self, memory_id: str):
        record_dict = self.storage.get(memory_id)
        if record_dict:
            return MemoryRecord.from_dict(record_dict)
        return None

    def list(self):
        records = []
        for record_dict in self.storage.list():
            records.append(MemoryRecord.from_dict(record_dict))
        return records

    def search(self, query: str, limit: int = 5):
        """
        Semantically search memories.
        """
        query_embedding = self.embedder.embed(query)
        results = self.vector_store.search(query_embedding, limit=limit)
        return results

    def search_keyword(self, query: str, limit: int = 5):
        """
        Keyword search memories.
        """
        results = self.storage.search_keyword(query, limit=limit)
        return results
