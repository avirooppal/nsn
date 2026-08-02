ANTONYM_PAIRS = [
    ('success', 'failure'), ('succeed', 'fail'), ('win', 'lose'),
    ('safe', 'dangerous'), ('correct', 'incorrect'), ('true', 'false'),
    ('healthy', 'sick'), ('increase', 'decrease'), ('start', 'stop'),
    ('approve', 'reject'), ('create', 'destroy'), ('love', 'hate'),
    ('hot', 'cold'), ('fast', 'slow'), ('open', 'closed'),
]

class ConsistencyScorer:
    """
    Evaluates trust based on consistency with existing memories.
    Uses semantic similarity, negation heuristics, and antonym detection to catch conflicts.
    """
    def __init__(self, memory):
        self.memory = memory
        self.negation_words = {'not', 'never', 'false', 'no', 'cannot', "don't", "doesn't", "isn't"}

    def score(self, content: str) -> float:
        results = self.memory.search(content, limit=1)
        if not results:
            return 0.8
            
        top_match = results[0]
        top_score = top_match.get('score', 0.0)
        
        if top_score < 0.6:
            return 0.8
            
        content_words = set(content.lower().split())
        existing_words = set(top_match['content'].lower().split())
        
        obs_has_negation = any(w in content_words for w in self.negation_words)
        mem_has_negation = any(w in existing_words for w in self.negation_words)
        
        if obs_has_negation != mem_has_negation:
            return 0.15
            
        # Check antonym pairs
        for w1, w2 in ANTONYM_PAIRS:
            if (w1 in content_words and w2 in existing_words) or (w2 in content_words and w1 in existing_words):
                return 0.2
                
        if top_score > 0.85:
            return 1.0
            
        return 0.8
