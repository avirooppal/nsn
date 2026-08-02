import os
import yaml

class Settings:
    """
    Centralized configuration for NeuroSleepNet.
    """
    def __init__(self):
        self._load_defaults()
        self._load_env()

    def _load_defaults(self):
        current_dir = os.path.dirname(__file__)
        defaults_path = os.path.join(current_dir, "defaults.yaml")
        
        with open(defaults_path, "r") as f:
            config = yaml.safe_load(f)
            
        self.backend = config.get("backend", "sqlite")
        self.vector_store = config.get("vector_store", "local")
        self.cache = config.get("cache", "memory")
        self.namespace = config.get("namespace", "default")
        self.embedding_model = config.get("embedding_model", "all-MiniLM-L6-v2")
        self.duplicate_threshold = config.get("duplicate_threshold", 0.9)

    def _load_env(self):
        if 'NSN_BACKEND' in os.environ:
            self.backend = os.environ['NSN_BACKEND']
        if 'NSN_NAMESPACE' in os.environ:
            self.namespace = os.environ['NSN_NAMESPACE']
        if 'NSN_EMBEDDING_MODEL' in os.environ:
            self.embedding_model = os.environ['NSN_EMBEDDING_MODEL']
        if 'NSN_DUPLICATE_THRESHOLD' in os.environ:
            self.duplicate_threshold = float(os.environ['NSN_DUPLICATE_THRESHOLD'])
        self.vector_store = os.getenv("NSN_VECTOR_STORE", self.vector_store)
        self.cache = os.getenv("NSN_CACHE", self.cache)
