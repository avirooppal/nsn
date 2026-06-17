from neurosleepnet.perception.schemas import Observation
from neurosleepnet.trust.source import SourceScorer
from neurosleepnet.trust.recency import RecencyScorer
from neurosleepnet.trust.consistency import ConsistencyScorer
from neurosleepnet.trust.schemas import TrustProfile

class TrustEngine:
    """
    Combines individual trust components to calculate a final trust score.
    """
    def __init__(self, memory):
        self.source_scorer = SourceScorer()
        self.recency_scorer = RecencyScorer()
        self.consistency_scorer = ConsistencyScorer(memory)

    def calculate(self, observation: Observation) -> TrustProfile:
        source_score = self.source_scorer.score(observation.source)
        recency_score = self.recency_scorer.score(observation.timestamp)
        consistency_score = self.consistency_scorer.score(observation.content)
        
        # Weighted average for final score
        # Weights: Source (30%), Recency (20%), Consistency (50%)
        final = (source_score * 0.3) + (recency_score * 0.2) + (consistency_score * 0.5)
        
        return TrustProfile(
            source_score=source_score,
            recency_score=recency_score,
            consistency_score=consistency_score,
            final_score=round(final, 4)
        )
