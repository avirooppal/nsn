class ConsistencyScorer:
    """
    Evaluates trust based on consistency with existing memories.
    Uses semantic similarity and negation heuristics to detect conflicts.
    """
    def __init__(self, memory):
        self.memory = memory
        self.negation_words = {'not', 'never', 'false', 'no', 'cannot', "don't", "doesn't", "isn't", 'dislike', 'hate'}

    def score(self, content: str) -> float:
        results = self.memory.search(content, limit=1)
        if not results:
            return 0.8 # Novel information
            
        top_match = results[0]
        top_score = top_match.get('score', 0.0)
        
        content_words = set(content.lower().split())
        
        if top_score > 0.7:
            existing_words = set(top_match['content'].lower().split())
            
            obs_has_negation = any(w in content_words for w in self.negation_words)
            mem_has_negation = any(w in existing_words for w in self.negation_words)
            
            # If one has negation and the other doesn't in a highly similar context, flag conflict
            if obs_has_negation != mem_has_negation:
                return 0.2 # Conflict detected
            else:
                return 1.0 # Consistent reinforcement
                
        return 0.8 # Somewhat novel
