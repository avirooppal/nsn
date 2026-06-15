import os
import yaml

class Settings:
    """
    Centralized configuration for NeuroSleepNet.
    """
    def __init__(self):
        self._load_defaults()

    def _load_defaults(self):
        current_dir = os.path.dirname(__file__)
        defaults_path = os.path.join(current_dir, "defaults.yaml")
        
        with open(defaults_path, "r") as f:
            config = yaml.safe_load(f)
            
        self.backend = config.get("backend", "sqlite")
        self.vector_store = config.get("vector_store", "local")
        self.cache = config.get("cache", "memory")

