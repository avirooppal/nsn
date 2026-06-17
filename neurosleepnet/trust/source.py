class SourceScorer:
    """
    Evaluates the trust score based on the origin of the information.
    """
    def __init__(self):
        self.source_trust_levels = {
            'system': 1.0,
            'user': 0.9,
            'agent': 0.7,
            'web': 0.4,
            'unknown': 0.5
        }

    def score(self, source: str) -> float:
        return self.source_trust_levels.get(source.lower(), 0.5)
