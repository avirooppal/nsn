from dataclasses import dataclass, asdict, field
import json
from datetime import datetime
import uuid
from typing import Dict, Any, List

@dataclass
class MemoryRecord:
    """
    Standard memory format for NeuroSleepNet.
    """
    content: str
    id: str = None
    created_at: str = None
    metadata: Dict[str, Any] = field(default_factory=dict)
    importance: float = 0.0
    trust_score: float = 0.5
    embedding: List[float] = field(default_factory=list)
    namespace: str = "default"
    memory_type: str = "semantic"
    access_count: int = 0
    last_accessed_at: str = None

    def __post_init__(self):
        if self.id is None:
            self.id = str(uuid.uuid4())
        if self.created_at is None:
            self.created_at = datetime.utcnow().isoformat()

    def to_dict(self):
        return asdict(self)

    def to_json(self):
        return json.dumps(self.to_dict())

    @classmethod
    def from_dict(cls, data: dict):
        from dataclasses import fields
        valid_keys = {f.name for f in fields(cls)}
        filtered = {k: v for k, v in data.items() if k in valid_keys}
        return cls(**filtered)

    @classmethod
    def from_json(cls, json_str):
        return cls.from_dict(json.loads(json_str))
