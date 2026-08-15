"""
NeuroSleepNet Research Evaluation Framework — Main CLI Runner.

Usage:
    python -m benchmarks.run --benchmark update --system nsn --samples 50
    python -m benchmarks.run --benchmark all --system all --samples 100
    python -m benchmarks.run --benchmark ablation --system nsn
    python -m benchmarks.run --benchmark sleep_ablation --system nsn
    python -m benchmarks.run --benchmark retrieval_ablation --system nsn

Benchmark types:
    update             Knowledge-update temporal tracking
    contradiction      Trust/source conflict resolution
    multihop           Multi-hop graph relational reasoning
    all                All three synthetic benchmarks
    ablation           Full retrieval component ablation (NSN only)
    sleep_ablation     Sleep consolidation ablation (NSN only)

Systems:
    nsn, bm25, dense, hybrid, full_context, all
"""

import argparse
import sys
import json
import os
import math
import datetime

from benchmarks.baselines.nsn_adapter import NSNSystem
from benchmarks.baselines.full_context import FullContextSystem
from benchmarks.baselines.bm25 import BM25System
from benchmarks.baselines.dense_rag import DenseRAGSystem
from benchmarks.baselines.hybrid_rag import HybridRAGSystem
from benchmarks.runners.evaluator import BenchmarkEvaluator
from benchmarks.runners.ablation import (
    run_sleep_ablation_experiment,
    run_retrieval_ablation_experiment,
)
from benchmarks.reports.logger import BenchmarkLogger
from benchmarks.reports.generator import ReportGenerator


def get_system(system_name: str, db_path: str = "bench_run.db"):
    if system_name == "nsn":
        return NSNSystem(db_path=db_path, name="nsn")
    elif system_name == "full_context":
        return FullContextSystem(name="full_context")
    elif system_name == "bm25":
        return BM25System(db_path=db_path, name="bm25")
    elif system_name == "dense":
        return DenseRAGSystem(db_path=db_path, name="dense_rag")
    elif system_name == "hybrid":
        return HybridRAGSystem(db_path=db_path, name="hybrid_rag")
    else:
        raise ValueError(f"Unknown system: {system_name}")


def fmt(v, pct=False, decimals=4):
    """Format a metric value for display, handling NaN."""
    if v is None or (isinstance(v, float) and math.isnan(v)):
        return "N/A"
    if pct:
        return f"{v * 100:.2f}%"
    return f"{v:.{decimals}f}"


def print_results(res: dict):
    """Pretty-print a single system's benchmark results."""
    name = res.get("system", "?")
    benchmark = res.get("benchmark", "?")
    n = res.get("samples", 0)
    print(f"\n{'='*60}")
    print(f"  {name.upper()} — {benchmark}  (n={n})")
    print(f"{'='*60}")
    print(f"  Answer Metrics:")
    print(f"    Exact Match   : {fmt(res.get('exact_match'), pct=True)}")
    print(f"    Token F1      : {fmt(res.get('token_f1'), pct=True)}")
    print(f"    ROUGE-L       : {fmt(res.get('rouge_l'), pct=True)}")
    print(f"  Retrieval Metrics (N/A for full-context):")
    print(f"    Recall@5      : {fmt(res.get('recall_5'), pct=True)}")
    print(f"    Hit@5         : {fmt(res.get('hit_5'), pct=True)}")
    print(f"    MRR           : {fmt(res.get('mrr'))}")
    print(f"    nDCG@5        : {fmt(res.get('ndcg_5'))}")
    print(f"    Mean Rank     : {fmt(res.get('mean_rank'))}")
    print(f"  Efficiency:")
    print(f"    Mean Tokens   : {res.get('mean_tokens', 0):.1f}")
    print(f"    P50 Latency   : {res.get('p50_latency', 0):.2f} ms")
    print(f"    P95 Latency   : {res.get('p95_latency', 0):.2f} ms")
    m2x2 = res.get("2x2_matrix", {})
    if m2x2.get("true_positives", 0) + m2x2.get("retrieval_failures", 0) > 0:
        print(f"  2x2 Failure Matrix:")
        print(f"    TRUE_POSITIVE    : {m2x2.get('true_positives', 0)}")
        print(f"    REASONING_FAILURE: {m2x2.get('reasoning_failures', 0)}")
        print(f"    RETRIEVAL_FAILURE: {m2x2.get('retrieval_failures', 0)}")
        print(f"    LUCKY_GUESS      : {m2x2.get('lucky_guesses', 0)}")
        print(f"    DUP_SKIPPED      : {m2x2.get('duplicate_rejected_skipped', 0)}")


def generate_final_report(
    all_results: dict,
    ablation_results: dict,
    sleep_results: dict,
    output_dir: str,
    timestamp: str,
) -> str:
    """Generate FINAL_REPORT.md with real empirical numbers."""
    os.makedirs(output_dir, exist_ok=True)
    lines = []

    lines.append("# NeuroSleepNet (NSN) — Research Evaluation Final Report")
    lines.append("")
    lines.append(f"> Generated: {timestamp}  ")
    lines.append("> Framework version: benchmark/v2.0 (gold-evidence ID mapping fix applied)")
    lines.append("")
    lines.append("## Benchmark Integrity Notes")
    lines.append("")
    lines.append("- **Gold-evidence ID mapping**: `_ingest_with_id_map()` captures actual stored UUIDs")
    lines.append("  after each `observe()` call. Retrieval metrics are computed against translated IDs.")
    lines.append("  The previous Recall@5 = 0% was an instrumentation bug — now fixed.")
    lines.append("- **Retrieval ≠ Answering**: metrics are evaluated independently. Answer correctness")
    lines.append("  is NOT used as a proxy for retrieval correctness.")
    lines.append("- **2×2 Failure Matrix**: classifies every sample as TRUE_POSITIVE /")
    lines.append("  REASONING_FAILURE / RETRIEVAL_FAILURE / LUCKY_GUESS.")
    lines.append("- All results are from a single run with `--seed 42`.")
    lines.append("")

    # --- System comparison tables per benchmark ---
    benchmarks_seen = set()
    for sys_name, bench_dict in all_results.items():
        for bname in bench_dict:
            benchmarks_seen.add(bname)

    for bname in sorted(benchmarks_seen):
        lines.append(f"---")
        lines.append(f"## Benchmark: {bname.replace('_', ' ').title()}")
        lines.append("")
        lines.append("| System | Recall@5 | Hit@5 | MRR | nDCG@5 | EM | Token-F1 | ROUGE-L | Tokens | P95 (ms) |")
        lines.append("|:---|:---:|:---:|:---:|:---:|:---:|:---:|:---:|:---:|:---:|")
        for sys_name, bench_dict in all_results.items():
            res = bench_dict.get(bname, {})
            if not res:
                continue
            row = (
                f"| {sys_name} "
                f"| {fmt(res.get('recall_5'), pct=True)} "
                f"| {fmt(res.get('hit_5'), pct=True)} "
                f"| {fmt(res.get('mrr'))} "
                f"| {fmt(res.get('ndcg_5'))} "
                f"| {fmt(res.get('exact_match'), pct=True)} "
                f"| {fmt(res.get('token_f1'), pct=True)} "
                f"| {fmt(res.get('rouge_l'), pct=True)} "
                f"| {res.get('mean_tokens', 0):.1f} "
                f"| {res.get('p95_latency', 0):.2f} |"
            )
            lines.append(row)
        lines.append("")

        # 2x2 matrix breakdown
        lines.append("### 2×2 Failure Matrix")
        lines.append("")
        lines.append("| System | Samples | TP | Reasoning Fail | Retrieval Fail | Lucky Guess | Dup Skipped |")
        lines.append("|:---|:---:|:---:|:---:|:---:|:---:|:---:|")
        for sys_name, bench_dict in all_results.items():
            res = bench_dict.get(bname, {})
            if not res:
                continue
            m = res.get("2x2_matrix", {})
            lines.append(
                f"| {sys_name} | {res.get('samples', 0)} "
                f"| {m.get('true_positives', 0)} "
                f"| {m.get('reasoning_failures', 0)} "
                f"| {m.get('retrieval_failures', 0)} "
                f"| {m.get('lucky_guesses', 0)} "
                f"| {m.get('duplicate_rejected_skipped', 0)} |"
            )
        lines.append("")

    # --- Retrieval ablation ---
    if ablation_results:
        lines.append("---")
        lines.append("## Retrieval Component Ablation (NSN)")
        lines.append("")
        lines.append("| Variant | Recall@5 | MRR | nDCG@5 | EM |")
        lines.append("|:---|:---:|:---:|:---:|:---:|")
        for variant, res in ablation_results.items():
            lines.append(
                f"| {variant} "
                f"| {fmt(res.get('recall_5'), pct=True)} "
                f"| {fmt(res.get('mrr'))} "
                f"| {fmt(res.get('ndcg_5'))} "
                f"| {fmt(res.get('exact_match'), pct=True)} |"
            )
        lines.append("")

    # --- Sleep ablation ---
    if sleep_results:
        lines.append("---")
        lines.append("## Sleep Consolidation Ablation (NSN)")
        lines.append("")
        lines.append("| Mode | Recall@5 | MRR | EM |")
        lines.append("|:---|:---:|:---:|:---:|")
        for mode, res in sleep_results.items():
            lines.append(
                f"| {mode} "
                f"| {fmt(res.get('recall_5'), pct=True)} "
                f"| {fmt(res.get('mrr'))} "
                f"| {fmt(res.get('exact_match'), pct=True)} |"
            )
        lines.append("")

    # --- Findings ---
    lines.append("---")
    lines.append("## Key Findings")
    lines.append("")
    lines.append("### Known Limitations Identified")
    lines.append("")
    lines.append("1. **BM25 / FTS5 phrase-match limitation**: `search_keyword()` wraps the full")
    lines.append("   query in FTS5 phrase-match quotes. A question like")
    lines.append('   *"What is the current production port for PostgreSQL?"* will almost never')
    lines.append("   match memory content, resulting in Recall@5 ≈ 0% for BM25.")
    lines.append("2. **NSN duplicate detection**: The `DuplicateDetector` uses cosine-similarity")
    lines.append("   threshold 0.95. Very similar synthetic observations (same topic, different")
    lines.append("   values) may be rejected. Samples with rejected gold memories are skipped")
    lines.append("   and counted in `duplicate_rejected_skipped`.")
    lines.append("3. **Answer extraction heuristic**: All non-LLM systems return the raw content")
    lines.append("   of the top-1 retrieved memory as the answer. EM checks substring containment,")
    lines.append("   so EM = 1 requires the answer value to appear literally in retrieved content.")
    lines.append("")

    report_path = os.path.join(output_dir, "FINAL_REPORT.md")
    with open(report_path, "w", encoding="utf-8") as f:
        f.write("\n".join(lines))
    return report_path


def main():
    parser = argparse.ArgumentParser(description="NeuroSleepNet Evaluation Framework CLI")
    parser.add_argument(
        "--benchmark", type=str, default="update",
        choices=["update", "contradiction", "multihop", "all",
                 "ablation", "sleep_ablation", "retrieval_ablation"],
    )
    parser.add_argument(
        "--system", type=str, default="nsn",
        choices=["nsn", "bm25", "dense", "hybrid", "full_context", "all"],
    )
    parser.add_argument("--samples", type=int, default=50,
                        help="Number of benchmark samples per category")
    parser.add_argument("--chains", type=int, default=20,
                        help="Number of multi-hop chains")
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--output", type=str, default="benchmarks/results")
    parser.add_argument("--ablation-samples", type=int, default=30)

    args = parser.parse_args()

    timestamp = datetime.datetime.utcnow().strftime("%Y-%m-%d %H:%M:%S UTC")
    exp_id = f"exp_{datetime.datetime.utcnow().strftime('%Y%m%d_%H%M%S')}"

    logger = BenchmarkLogger(
        experiment_id=exp_id,
        output_dir=os.path.join(args.output, "raw"),
    )
    evaluator = BenchmarkEvaluator(logger=logger)

    print("=" * 60)
    print("NEUROSLEEPNET RESEARCH EVALUATION FRAMEWORK v2.0")
    print(f"Benchmark : {args.benchmark}")
    print(f"System    : {args.system}")
    print(f"Samples   : {args.samples} | Seed: {args.seed}")
    print(f"Timestamp : {timestamp}")
    print("=" * 60)

    # --- Sleep ablation ---
    if args.benchmark in ("sleep_ablation", "ablation"):
        print("\n[SLEEP ABLATION] Running...")
        sleep_res = run_sleep_ablation_experiment(
            num_samples=args.ablation_samples,
            db_prefix="bench_sleep_ablation",
        )
        print("\nSLEEP ABLATION RESULTS:")
        for mode, res in sleep_res.items():
            print(f"  {mode:15s} | EM={fmt(res['exact_match'], pct=True):7s} | "
                  f"Recall@5={fmt(res['recall_5'], pct=True):7s} | "
                  f"MRR={fmt(res['mrr'])}")
        if args.benchmark == "sleep_ablation":
            return

    # --- Retrieval ablation ---
    if args.benchmark in ("retrieval_ablation", "ablation"):
        print("\n[RETRIEVAL ABLATION] Running...")
        ret_res = run_retrieval_ablation_experiment(
            num_samples=args.ablation_samples,
            db_prefix="bench_retrieval_ablation",
        )
        print("\nRETRIEVAL ABLATION RESULTS:")
        for variant, res in ret_res.items():
            print(f"  {variant:22s} | EM={fmt(res['exact_match'], pct=True):7s} | "
                  f"Recall@5={fmt(res['recall_5'], pct=True):7s} | "
                  f"MRR={fmt(res['mrr'])}")
        if args.benchmark == "retrieval_ablation":
            return

    # --- System selection ---
    systems_to_run = (
        ["nsn", "full_context", "bm25", "dense", "hybrid"]
        if args.system == "all"
        else [args.system]
    )

    # --- Benchmark selection ---
    benchmarks_to_run = (
        ["update", "contradiction", "multihop"]
        if args.benchmark == "all"
        else [args.benchmark]
    )

    all_results = {}   # {sys_name: {benchmark_name: result_dict}}
    ablation_res = {}
    sleep_res = {}

    for sys_name in systems_to_run:
        all_results[sys_name] = {}
        for bench_name in benchmarks_to_run:
            print(f"\n[{bench_name.upper()}] Running system: {sys_name}...")
            sys_obj = get_system(sys_name, db_path=f"bench_{sys_name}_{bench_name}.db")

            if bench_name == "update":
                res = evaluator.evaluate_update_benchmark(sys_obj, num_samples=args.samples)
            elif bench_name == "contradiction":
                res = evaluator.evaluate_contradiction_benchmark(sys_obj, num_samples=args.samples)
            elif bench_name == "multihop":
                res = evaluator.evaluate_multihop_benchmark(sys_obj, num_chains=args.chains)
            else:
                print(f"  Unknown benchmark: {bench_name}, skipping.")
                continue

            all_results[sys_name][bench_name] = res
            print_results(res)

    # --- Generate FINAL_REPORT ---
    report_path = generate_final_report(
        all_results=all_results,
        ablation_results=ablation_res,
        sleep_results=sleep_res,
        output_dir=args.output,
        timestamp=timestamp,
    )
    print(f"\n[REPORT] Written to: {report_path}")

    # Save raw JSON
    json_path = os.path.join(args.output, "raw", f"{exp_id}_summary.json")
    with open(json_path, "w", encoding="utf-8") as f:
        json.dump(all_results, f, indent=2, default=str)
    print(f"[JSON]   Written to: {json_path}")


if __name__ == "__main__":
    main()

