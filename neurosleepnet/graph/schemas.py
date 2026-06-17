from dataclasses import dataclass, field
from typing import Dict, Any
import datetime
import uuid

@dataclass
class GraphNode:
    """
    Represents an entity or concept in the knowledge graph.
    """
    label: str
    name: str
    id: str = None
    properties: Dict[str, Any] = field(default_factory=dict)
    created_at: str = None

    def __post_init__(self):
        if self.id is None:
            self.id = str(uuid.uuid4())
        if self.created_at is None:
            self.created_at = datetime.datetime.utcnow().isoformat()
