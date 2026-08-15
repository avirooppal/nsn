# NeuroSleepNet Research Evaluation Framework (`benchmarks/`)

This directory contains a complete, reproducible, research-grade evaluation and benchmarking framework for **NeuroSleepNet (NSN)**.

---

## 1. Quick Start

### Run Framework Unit Tests
```bash
python -m unittest benchmarks/tests/test_framework.py
```

### Run Synthetic Benchmark across Baseline Systems
```bash
python -m benchmarks.run --benchmark synthetic --system nsn --memory-size 10
```

### Compare All Baselines (NSN vs BM25 vs Dense RAG vs Hybrid RAG vs Full Context)
```bash
python -m benchmarks.run --benchmark synthetic --system all --memory-size 20
```

### Run Persistence Survival Test
```bash
python -m benchmarks.run --benchmark persistence
```

### Run Offline Sleep Consolidation Ablation Study
```bash
python -m benchmarks.run --benchmark ablation
```

---

## 2. Standard Research Benchmarks (LongMemEval, LoCoMo, LoCoMo-Plus)

To run standard benchmarks:
```bash
python -m benchmarks.run --benchmark longmemeval
python -m benchmarks.run --benchmark locomo
python -m benchmarks.run --benchmark locomo_plus
```

If dataset files are not detected locally under `benchmarks/datasets/`, the benchmark harness cleanly reports:
`NOT AVAILABLE — DATASET REQUIRED` along with the official repository links.

---

## 3. Directory Architecture

- **`config/`**: Evaluation parameters (`default.yaml`), scaling settings (`scaling.yaml`), and ablation matrices (`ablations.yaml`).
- **`datasets/`**: Adapters for LongMemEval, LoCoMo, and LoCoMo-Plus (`adapters.py`) and synthetic test case generators (`synthetic.py`).
- **`baselines/`**: Unified `BaseSystem` adapters for Full Context, BM25, Dense RAG, Hybrid RAG, and NSN.
- **`metrics/`**: NLP metrics (`nlp.py`), retrieval metrics (`retrieval.py`), and statistical tests (`statistics.py`).
- **`runners/`**: Execution engines (`evaluator.py`), persistence survival (`persistence.py`), and sleep ablations (`ablation.py`).
- **`reports/`**: Structured JSONL raw logger (`logger.py`) and Markdown/LaTeX research paper table generator (`generator.py`).
- **`results/`**: Output directory for raw logs, processed JSON summaries, figures, and tables.
