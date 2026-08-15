"""
Generates summary Markdown files, JSON reports, CSVs, and research paper tables.
Handles NaN values (N/A) from retrieval metrics that are inapplicable for
full-context baselines.
"""
import os
import json
import csv
import math
from benchmarks.metrics.statistics import compute_summary_statistics


def _fmt(v, pct=False, dec=4):
    """Format a metric value for table display, handling NaN."""
    if v is None or (isinstance(v, float) and math.isnan(v)):
        return "N/A"
    if pct:
        return f"{v * 100:.2f}%"
    return f"{v:.{dec}f}"


class ReportGenerator:
    def __init__(self, output_dir: str = "benchmarks/results"):
        self.output_dir = output_dir
        self.tables_dir = os.path.join(output_dir, "tables")
        os.makedirs(self.tables_dir, exist_ok=True)

    def generate_research_table(self, results_by_system: dict) -> str:
        """
        Produces a research paper standard comparison table.
        Handles NaN retrieval metrics (N/A) for full-context baselines.
        """
        header = (
            "| System | Samples | Recall@5 | Hit@5 | MRR | nDCG@5 |"
            " EM | Token-F1 | ROUGE-L | Tokens | P95 (ms) |\n"
        )
        divider = "|:---|:---:|:---:|:---:|:---:|:---:|:---:|:---:|:---:|:---:|:---:|\n"
        rows = ""

        for sys_name, res in results_by_system.items():
            rows += (
                f"| {sys_name} "
                f"| {res.get('samples', 0)} "
                f"| {_fmt(res.get('recall_5'), pct=True)} "
                f"| {_fmt(res.get('hit_5'), pct=True)} "
                f"| {_fmt(res.get('mrr'))} "
                f"| {_fmt(res.get('ndcg_5'))} "
                f"| {_fmt(res.get('exact_match'), pct=True)} "
                f"| {_fmt(res.get('token_f1'), pct=True)} "
                f"| {_fmt(res.get('rouge_l'), pct=True)} "
                f"| {res.get('mean_tokens', 0):.1f} "
                f"| {res.get('p95_latency', 0):.2f} |\n"
            )

        table_md = header + divider + rows
        with open(
            os.path.join(self.tables_dir, "research_table.md"), "w", encoding="utf-8"
        ) as f:
            f.write(table_md)
        return table_md

    def generate_ablation_table(self, ablation_results: dict) -> str:
        """Produces ablation study research table."""
        header = "| NSN Variant | Samples | Recall@5 | MRR | nDCG@5 | EM | Tokens | P95 (ms) |\n"
        divider = "|:---|:---:|:---:|:---:|:---:|:---:|:---:|:---:|\n"
        rows = ""
        for variant, res in ablation_results.items():
            rows += (
                f"| {variant} "
                f"| {res.get('samples', 0)} "
                f"| {_fmt(res.get('recall_5'), pct=True)} "
                f"| {_fmt(res.get('mrr'))} "
                f"| {_fmt(res.get('ndcg_5'))} "
                f"| {_fmt(res.get('exact_match'), pct=True)} "
                f"| {res.get('mean_tokens', 0):.1f} "
                f"| {res.get('p95_latency', 0):.2f} |\n"
            )
        table_md = header + divider + rows
        with open(
            os.path.join(self.tables_dir, "ablation_table.md"), "w", encoding="utf-8"
        ) as f:
            f.write(table_md)
        return table_md

    def generate_comparison_csv(self, results_by_system: dict, filename: str = "comparison.csv") -> str:
        """Export all metrics to CSV for further analysis."""
        csv_path = os.path.join(self.tables_dir, filename)
        fields = [
            "system", "benchmark", "samples",
            "recall_5", "hit_5", "mrr", "ndcg_5", "mean_rank",
            "exact_match", "token_f1", "rouge_l",
            "mean_tokens", "p50_latency", "p95_latency",
            "tp", "reasoning_failures", "retrieval_failures", "lucky_guesses",
        ]
        with open(csv_path, "w", newline="", encoding="utf-8") as f:
            writer = csv.DictWriter(f, fieldnames=fields, extrasaction="ignore")
            writer.writeheader()
            for sys_name, res in results_by_system.items():
                m = res.get("2x2_matrix", {})
                row = {
                    "system": sys_name,
                    "benchmark": res.get("benchmark", ""),
                    "samples": res.get("samples", 0),
                    "recall_5": res.get("recall_5", float("nan")),
                    "hit_5": res.get("hit_5", float("nan")),
                    "mrr": res.get("mrr", float("nan")),
                    "ndcg_5": res.get("ndcg_5", float("nan")),
                    "mean_rank": res.get("mean_rank", float("nan")),
                    "exact_match": res.get("exact_match", 0.0),
                    "token_f1": res.get("token_f1", 0.0),
                    "rouge_l": res.get("rouge_l", 0.0),
                    "mean_tokens": res.get("mean_tokens", 0),
                    "p50_latency": res.get("p50_latency", 0),
                    "p95_latency": res.get("p95_latency", 0),
                    "tp": m.get("true_positives", 0),
                    "reasoning_failures": m.get("reasoning_failures", 0),
                    "retrieval_failures": m.get("retrieval_failures", 0),
                    "lucky_guesses": m.get("lucky_guesses", 0),
                }
                writer.writerow(row)
        return csv_path

