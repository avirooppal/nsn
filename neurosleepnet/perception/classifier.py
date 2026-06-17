from neurosleepnet.perception.schemas import Observation

class MemoryClassifier:
    """
    Classifies an observation into episodic, semantic, or procedural memory.
    """
    def classify(self, observation: Observation) -> str:
        content_lower = observation.content.lower()
        
        # Procedural: How-to knowledge
        if any(kw in content_lower for kw in ['how to', 'step by step', 'instruction', 'guide']):
            return 'procedural'
            
        # Episodic: Personal experiences or events
        if any(kw in content_lower for kw in ['yesterday', 'today', 'i saw', 'we went', 'happened', 'user clicked']):
            return 'episodic'
            
        # Default to Semantic: Factual knowledge
        return 'semantic'
