"""
Standard research benchmark adapters (LongMemEval, LoCoMo, LoCoMo-Plus).
Preserves benchmark specs, train/dev/test separation, and reports dataset status.
"""
import os
import json

class BenchmarkAdapter:
    def __init__(self, name: str, data_dir: str = "benchmarks/datasets"):
        self.name = name
        self.data_dir = data_dir
        self.dataset_path = os.path.join(data_dir, f"{name}.json")

    def is_available(self) -> bool:
        return os.path.exists(self.dataset_path)

    def load_data(self, split: str = "test") -> list:
        if not self.is_available():
            raise FileNotFoundError(
                f"NOT AVAILABLE — DATASET REQUIRED for {self.name}.\n"
                f"Please obtain the official dataset and place it at {self.dataset_path}.\n"
                f"Official sources:\n"
                f"- LongMemEval: https://github.com/xiaowu-jiang/LongMemEval\n"
                f"- LoCoMo / LoCoMo-Plus: https://github.com/zjunlp/LoCoMo"
            )
        
        with open(self.dataset_path, "r", encoding="utf-8") as f:
            data = json.load(f)
            
        if isinstance(data, dict) and split in data:
            return data[split]
        elif isinstance(data, list):
            return data
        return []

class LongMemEvalAdapter(BenchmarkAdapter):
    def __init__(self, data_dir: str = "benchmarks/datasets/longmemeval"):
        super().__init__("longmemeval", data_dir)

class LoCoMoAdapter(BenchmarkAdapter):
    def __init__(self, data_dir: str = "benchmarks/datasets/locomo"):
        super().__init__("locomo", data_dir)

class LoCoMoPlusAdapter(BenchmarkAdapter):
    def __init__(self, data_dir: str = "benchmarks/datasets/locomo_plus"):
        super().__init__("locomo_plus", data_dir)
