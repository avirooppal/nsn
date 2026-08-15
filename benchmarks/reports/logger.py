"""
Raw experiment logger writing structured JSONL & JSON records.
"""
import os
import json
import time
import datetime

class BenchmarkLogger:
    def __init__(self, experiment_id: str, output_dir: str = "benchmarks/results/raw"):
        self.experiment_id = experiment_id
        self.output_dir = output_dir
        os.makedirs(output_dir, exist_ok=True)
        self.jsonl_path = os.path.join(output_dir, f"{experiment_id}.jsonl")
        self.records = []

    def log_query(
        self,
        benchmark: str,
        query_id: str,
        system: str,
        question: str,
        ground_truth: str,
        answer: str,
        correct: bool,
        retrieved_ids: list = None,
        retrieved_ranks: list = None,
        retrieved_scores: list = None,
        latency_ms: float = 0.0,
        prompt_tokens: int = 0,
        output_tokens: int = 0,
        memory_count: int = 0,
        metadata: dict = None
    ):
        record = {
            "experiment_id": self.experiment_id,
            "timestamp": datetime.datetime.utcnow().isoformat(),
            "benchmark": benchmark,
            "query_id": query_id,
            "system": system,
            "question": question,
            "ground_truth": ground_truth,
            "answer": answer,
            "correct": correct,
            "retrieved_memory_ids": retrieved_ids or [],
            "retrieved_ranks": retrieved_ranks or [],
            "retrieval_scores": retrieved_scores or [],
            "latency_ms": latency_ms,
            "prompt_tokens": prompt_tokens,
            "output_tokens": output_tokens,
            "memory_count": memory_count,
            "metadata": metadata or {}
        }
        self.records.append(record)
        with open(self.jsonl_path, "a", encoding="utf-8") as f:
            f.write(json.dumps(record) + "\n")
