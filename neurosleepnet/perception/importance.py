from neurosleepnet.perception.schemas import Observation

class ImportanceScorer:
    """
    Assigns an importance score to an observation.
    """
    def score(self, observation: Observation) -> float:
        base_score = 0.1
        content_lower = observation.content.lower()
        
        # High impact keywords
        if any(kw in content_lower for kw in ['error', 'critical', 'fail', 'bug', 'urgent']):
            base_score += 0.5
            
        # Action/Goal keywords
        if any(kw in content_lower for kw in ['goal', 'task', 'plan', 'must', 'important']):
            base_score += 0.3
            
        # Length bonus
        if len(observation.content) > 50:
            base_score += 0.1
            
        return min(1.0, base_score)
