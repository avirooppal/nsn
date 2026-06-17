from dataclasses import dataclass, field
from typing import Dict, Any

@dataclass
class TrustProfile:
    """
    Trust breakdown for an observation or memory.
    """
    source_score: float = 0.5
    recency_score: float = 0.5
    consistency_score: float = 0.5
    final_score: float = 0.5
    metadata: Dict[str, Any] = field(default_factory=dict)
