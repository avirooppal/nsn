from dataclasses import dataclass, field, asdict
from typing import Dict, Any
import datetime
import uuid

@dataclass
class Observation:
    """
    Raw input from an agent or system before it is processed into a MemoryRecord.
    """
    content: str
    source: str = "unknown"
    id: str = None
    timestamp: str = None
    metadata: Dict[str, Any] = field(default_factory=dict)
    
    def __post_init__(self):
        if self.id is None:
            self.id = str(uuid.uuid4())
        if self.timestamp is None:
            self.timestamp = datetime.datetime.utcnow().isoformat()

    def to_dict(self):
        return asdict(self)
