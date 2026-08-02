import re
from neurosleepnet.perception.schemas import Observation

class ImportanceScorer:
    """
    Assigns an importance score to an observation based on multiple heuristics.
    """
    def score(self, observation: Observation) -> float:
        base_score = 0.1
        content = observation.content
        content_lower = content.lower()
        
        # High impact keywords
        if any(kw in content_lower for kw in ['error', 'critical', 'fail', 'bug', 'urgent']):
            base_score += 0.5
            
        # Action/Goal keywords
        if any(kw in content_lower for kw in ['goal', 'task', 'plan', 'must', 'important']):
            base_score += 0.3

        # Information density
        words = content_lower.split()
        total_words = len(words)
        if total_words > 0:
            unique_words = len(set(words))
            base_score += (unique_words / total_words) * 0.1

        # Named entity density: count capitalized multi-word sequences
        entity_matches = re.findall(r'\b[A-Z][a-z]+(?:\s+[A-Z][a-z]+)+\b', content)
        entity_count = len(entity_matches)
        base_score += min(entity_count * 0.05, 0.15)
            
        # Length tiers
        length = len(content)
        if length > 200:
            base_score += 0.15
        elif length > 50:
            base_score += 0.08
            
        return round(min(1.0, base_score), 4)
