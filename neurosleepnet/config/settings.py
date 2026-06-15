class Settings:
    """
    Centralized configuration for NeuroSleepNet.
    """
    def __init__(self):
        # Base defaults
        self.backend = "sqlite"
        self.vector_store = "local"
        self.cache = "memory"
