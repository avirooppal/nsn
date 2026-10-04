"""
Common generator interface for benchmark evaluation.
Provides:
  - BaseGenerator: abstract interface
  - MockGenerator: deterministic / test fixture generator with fail injection
  - OllamaGenerator: local SLM generator with explicit failure handling
"""
import time
from abc import ABC, abstractmethod
from typing import Optional, Dict, Any

class BaseGenerator(ABC):
    @abstractmethod
    def generate(self, prompt: str, system_prompt: Optional[str] = None) -> str:
        """Generate an answer string given prompt and optional system prompt."""
        pass

    def set_seed(self, seed: int):
        """Set random seed for reproducibility if supported."""
        pass


class MockGenerator(BaseGenerator):
    """
    Deterministic mock generator for tests.
    Supports injecting specific answers, simulating crashes on matching prompts,
    and verifying seed propagation.
    """
    def __init__(self, responses: Optional[Dict[str, str]] = None, fail_on_prompts: Optional[list] = None, seed: Optional[int] = None):
        self.responses = responses or {}
        self.fail_on_prompts = list(fail_on_prompts or [])
        self.seed = seed
        self.last_prompt = None
        self.call_count = 0

    def set_seed(self, seed: int):
        self.seed = seed

    def generate(self, prompt: str, system_prompt: Optional[str] = None) -> str:
        self.call_count += 1
        self.last_prompt = prompt

        for bad in self.fail_on_prompts:
            if bad in prompt:
                raise RuntimeError(f"MockGenerator crashed intentionally on prompt containing: {bad}")

        for k, v in self.responses.items():
            if k in prompt:
                return v

        return f"Mock answer (seed={self.seed})"


class OllamaGenerator(BaseGenerator):
    """
    Calls local Ollama instance without falling back silently to extraction.
    Raises RuntimeError on connection or generation failures.
    """
    def __init__(self, model: str = "llama3:latest", host: str = "http://localhost:11434", seed: Optional[int] = None):
        self.model = model
        self.host = host
        self.seed = seed

    def set_seed(self, seed: int):
        self.seed = seed

    def generate(self, prompt: str, system_prompt: Optional[str] = None) -> str:
        import urllib.request
        import json as _json

        options: Dict[str, Any] = {"temperature": 0, "num_predict": 50}
        if self.seed is not None:
            options["seed"] = self.seed

        payload_dict = {
            "model": self.model,
            "prompt": prompt,
            "stream": False,
            "options": options,
        }
        if system_prompt:
            payload_dict["system"] = system_prompt

        payload = _json.dumps(payload_dict).encode("utf-8")
        req = urllib.request.Request(
            f"{self.host}/api/generate",
            data=payload,
            headers={"Content-Type": "application/json"},
            method="POST",
        )
        try:
            with urllib.request.urlopen(req, timeout=30) as resp:
                data = _json.loads(resp.read().decode("utf-8"))
                return data.get("response", "").strip()
        except Exception as e:
            raise RuntimeError(f"Ollama generation failed for model {self.model} at {self.host}: {e}") from e
