class SourceScorer:
    """
    Evaluates the trust score based on the origin of the information.
    Supports a broad registry of common agent/system source names.
    """

    # Curated source trust registry
    SOURCE_TRUST_LEVELS = {
        # System-level sources (highest trust)
        'system': 1.0,
        'system_log': 1.0,
        'ground_truth': 1.0,
        # Human/user sources
        'user': 0.9,
        'user_input': 0.9,
        'human': 0.9,
        # LLM-generated content
        'llm': 0.8,
        'gpt': 0.8,
        'claude': 0.8,
        'gemini': 0.8,
        # Agent/tool sources
        'agent': 0.75,
        'agent_sensor': 0.75,
        'tool': 0.75,
        'function_call': 0.75,
        # Automated/batch pipelines
        'batch': 0.7,
        'pipeline': 0.7,
        'rag': 0.65,
        # External/unverified sources
        'web': 0.4,
        'web_scrape': 0.4,
        'unknown': 0.5,
    }

    def score(self, source: str) -> float:
        return self.SOURCE_TRUST_LEVELS.get(source.lower(), 0.6)
