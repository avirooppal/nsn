from neurosleepnet.config.settings import Settings
from neurosleepnet.storage.sqlite import SQLiteAdapter
from neurosleepnet.memory.schemas import MemoryRecord
import json

class Memory:
    """
    Core Memory class for NeuroSleepNet.
    """
    def __init__(self):
        self.settings = Settings()
        if self.settings.backend == "sqlite":
            self.storage = SQLiteAdapter()
        else:
            raise NotImplementedError(f"Backend {self.settings.backend} not implemented")

    def store(self, content: str, metadata: dict = None, importance: float = 0.0, trust_score: float = 0.5):
        record = MemoryRecord(
            content=content, 
            metadata=metadata or {}, 
            importance=importance, 
            trust_score=trust_score
        )
        
        self.storage.store(
            memory_id=record.id,
            content=record.content,
            created_at=record.created_at,
            metadata=json.dumps(record.metadata),
            importance=record.importance,
            trust_score=record.trust_score
        )
        return record.id
