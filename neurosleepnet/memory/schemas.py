from dataclasses import dataclass, asdict, field
import json
from datetime import datetime
import uuid
from typing import Dict, Any

@dataclass
class MemoryRecord:
    """
    Standard memory format for NeuroSleepNet.
    """
    content: str
    id: str = None
    created_at: str = None
    metadata: Dict[str, Any] = field(default_factory=dict)

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
    def from_dict(cls, data):
        return cls(**data)

    @classmethod
    def from_json(cls, json_str):
        return cls.from_dict(json.loads(json_str))
