from neurosleepnet.perception.schemas import Observation

class DuplicateDetector:
    """
    Detects if an incoming observation is a duplicate of an existing memory.
    """
    def __init__(self, memory, threshold: float = 0.95):
        self.memory = memory
        self.threshold = threshold

    def is_duplicate(self, observation: Observation) -> bool:
        """
        Check if the observation already exists in memory.
        """
        results = self.memory.search(observation.content, limit=1)
        if results and results[0].get('score', 0.0) >= self.threshold:
            return True
        return False
