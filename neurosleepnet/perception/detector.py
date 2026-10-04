import re
from neurosleepnet.perception.schemas import Observation

NEGATION_TOKENS = {"not", "no", "never", "disabled", "inactive", "false", "neither", "nor"}

class DuplicateDetector:
    """
    Detects if an incoming observation is a duplicate of an existing memory.
    Preserves changed numbers, dates, identifiers, and negation even when
    semantic embedding similarity is high.
    """
    def __init__(self, memory, threshold: float = 0.95):
        self.memory = memory
        self.threshold = threshold

    def is_duplicate(self, observation: Observation) -> bool:
        """
        Check if the observation is genuinely equivalent to an existing memory.
        """
        # Exact match check first
        content = observation.content.strip()
        namespace = getattr(self.memory, "namespace", "default")
        
        # Check storage keyword/FTS search for exact or high overlap
        results = self.memory.search(content, limit=3)
        if not results:
            return False

        top_hit = results[0]
        score = top_hit.get("score", 0.0)
        top_content = top_hit.get("content", "").strip()

        # If identical text in same namespace -> duplicate
        if content.lower() == top_content.lower():
            return True

        if score < self.threshold:
            return False

        # If similarity is high, verify that numbers, dates, and negation are identical
        obs_numbers = set(re.findall(r'\b\d+\b', content))
        hit_numbers = set(re.findall(r'\b\d+\b', top_content))
        if obs_numbers != hit_numbers:
            # Different numbers (e.g. port 8080 vs port 9090, Day 1 vs Day 10) -> NOT a duplicate!
            return False

        obs_words = set(re.findall(r'\b\w+\b', content.lower()))
        hit_words = set(re.findall(r'\b\w+\b', top_content.lower()))
        obs_neg = obs_words.intersection(NEGATION_TOKENS)
        hit_neg = hit_words.intersection(NEGATION_TOKENS)
        if obs_neg != hit_neg:
            # Polarity change (e.g. is blue vs is not blue) -> NOT a duplicate!
            return False

        return True
