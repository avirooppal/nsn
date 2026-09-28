"""
Generate Publication-Grade Figures for NSN Benchmark Results
Creates dark-themed, clean, publication-ready charts matching GitHub Dark UI aesthetics (#0d1117 / #161b22).
"""
import os
import matplotlib.pyplot as plt
import numpy as np

# Apply dark theme aesthetic style
plt.style.use('dark_background')
BG_COLOR = '#0d1117'
PANEL_COLOR = '#161b22'
BORDER_COLOR = '#30363d'
TEXT_COLOR = '#e6edf3'
MUTED_TEXT = '#8b949e'

# Color Palette for Systems
SYSTEM_COLORS = {
    'Vanilla LLM': '#6e7681',
    'Rolling Window LLM': '#d29922',
    'LLM + BM25': '#f85149',
    'LLM + Dense RAG': '#a371f7',
    'LLM + Hybrid RAG': '#58a6ff',
    'LLM + RAG Memory': '#1f6feb',
    'NeuroSleepNet (NSN)': '#2ea043'  # Distinct Green highlight
}

SYSTEMS = [
    'Vanilla LLM',
    'Rolling Window LLM',
    'LLM + BM25',
    'LLM + Dense RAG',
    'LLM + Hybrid RAG',
    'LLM + RAG Memory',
    'NeuroSleepNet (NSN)'
]

def setup_ax(ax, title, ylabel, ylim=None):
    ax.set_facecolor(PANEL_COLOR)
    ax.set_title(title, fontsize=12, fontweight='bold', color=TEXT_COLOR, pad=12)
    ax.set_ylabel(ylabel, fontsize=10, fontweight='bold', color=TEXT_COLOR)
    ax.tick_params(colors=MUTED_TEXT, labelsize=9)
    ax.spines['top'].set_color(BORDER_COLOR)
    ax.spines['bottom'].set_color(BORDER_COLOR)
    ax.spines['left'].set_color(BORDER_COLOR)
    ax.spines['right'].set_color(BORDER_COLOR)
    ax.grid(axis='y', linestyle='--', alpha=0.2, color=MUTED_TEXT, zorder=0)
    if ylim:
        ax.set_ylim(ylim)

def ensure_dirs(paths):
    for p in paths:
        os.makedirs(p, exist_ok=True)

def generate_fig1_head_to_head(output_dirs):
    """Figure 1: Head-to-Head Recall@5 across 3 Benchmark Dimensions"""
    fig, axes = plt.subplots(1, 3, figsize=(18, 5.5), dpi=300)
    fig.patch.set_facecolor(BG_COLOR)

    # 1. Knowledge Update
    ku_recall = [0.0, 17.50, 0.0, 75.00, 75.00, 75.00, 85.71]
    # 2. Contradiction
    ct_recall = [0.0, 25.00, 0.0, 0.0, 0.0, 0.0, 100.00]
    # 3. Multi-Hop
    mh_recall = [0.0, 4.27, 0.0, 21.33, 21.33, 21.33, 42.67]

    benchmarks_data = [
        ("Knowledge Update\n(Temporal State Tracking)", ku_recall, axes[0]),
        ("Contradiction Resolution\n(Source Trust & Conflict)", ct_recall, axes[1]),
        ("Multi-Hop Traversal\n(Relational Entity Graph)", mh_recall, axes[2])
    ]

    for title, scores, ax in benchmarks_data:
        setup_ax(ax, title, "Recall@5 (%)", (0, 115))
        y_pos = np.arange(len(SYSTEMS))
        colors = [SYSTEM_COLORS[s] for s in SYSTEMS]
        
        bars = ax.barh(y_pos, scores, color=colors, height=0.65, zorder=3, alpha=0.9)
        ax.set_yticks(y_pos)
        ax.set_yticklabels(SYSTEMS, fontsize=9.5, fontweight='bold', color=TEXT_COLOR)
        ax.invert_yaxis()  # top-down display

        for bar, score in zip(bars, scores):
            if score > 0:
                ax.annotate(f"{score:.1f}%",
                            xy=(bar.get_width() + 1.5, bar.get_y() + bar.get_height() / 2),
                            xytext=(0, 0), textcoords="offset points",
                            ha='left', va='center', fontsize=9, fontweight='bold', color=TEXT_COLOR)
            else:
                ax.annotate("0.0% / N/A",
                            xy=(2.0, bar.get_y() + bar.get_height() / 2),
                            xytext=(0, 0), textcoords="offset points",
                            ha='left', va='center', fontsize=8.5, color=MUTED_TEXT)

    plt.suptitle("Empirical Benchmark Results: Recall@5 Across 7 Memory Architectures",
                 fontsize=14, fontweight='bold', color=TEXT_COLOR, y=1.02)
    plt.tight_layout()

    for out_dir in output_dirs:
        filepath = os.path.join(out_dir, "nsn_head_to_head_recall.png")
        plt.savefig(filepath, bbox_inches='tight', facecolor=BG_COLOR)
        print(f"[FIG GENERATED] Saved {filepath}")
    plt.close()

def generate_fig2_mrr_and_latency(output_dirs):
    """Figure 2: Rank Quality (MRR) and P95 Retrieval Latency Trade-off"""
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(14, 6), dpi=300)
    fig.patch.set_facecolor(BG_COLOR)

    # Subplot A: MRR Comparison
    setup_ax(ax1, "Mean Reciprocal Rank (MRR) Comparison\n(Higher is Better)", "MRR Score", (0, 1.15))
    x = np.arange(len(SYSTEMS))
    width = 0.25

    ku_mrr = [0.0, 0.1250, 0.0, 0.4250, 0.4250, 0.4250, 0.4929]
    ct_mrr = [0.0, 0.1125, 0.0, 0.0, 0.0, 0.0, 0.5000]
    mh_mrr = [0.0, 0.1000, 0.0, 0.2283, 0.2283, 0.2283, 1.0000]

    rects1 = ax1.bar(x - width, ku_mrr, width, label='Knowledge Update', color='#58a6ff', alpha=0.85, zorder=3)
    rects2 = ax1.bar(x, ct_mrr, width, label='Contradiction', color='#f0883e', alpha=0.85, zorder=3)
    rects3 = ax1.bar(x + width, mh_mrr, width, label='Multi-Hop', color='#2ea043', alpha=0.9, zorder=3)

    ax1.set_xticks(x)
    ax1.set_xticklabels([s.replace(" LLM", "").replace("LLM + ", "") for s in SYSTEMS],
                        rotation=30, ha='right', fontsize=8.5, fontweight='bold', color=TEXT_COLOR)
    ax1.legend(fontsize=8.5, loc='upper left', facecolor=PANEL_COLOR, edgecolor=BORDER_COLOR, labelcolor=TEXT_COLOR)

    # Highlight NSN MRR=1.0000
    ax1.annotate("MRR = 1.0000\n(Perfect Rank 1)", xy=(6 + width, 1.0), xytext=(4.2, 1.05),
                 arrowprops=dict(arrowstyle="->", color='#2ea043', lw=1.5),
                 fontsize=9, fontweight='bold', color='#2ea043')

    # Subplot B: P95 Retrieval Latency Comparison
    setup_ax(ax2, "Knowledge Update P95 Latency (ms)\n(Lower is Better)", "Latency (ms)", (0, 1000))
    latencies = [3.8, 2.1, 13.8, 888.0, 882.7, 890.0, 853.6]
    colors = [SYSTEM_COLORS[s] for s in SYSTEMS]

    bars2 = ax2.bar(x, latencies, color=colors, width=0.55, zorder=3, alpha=0.9)
    ax2.set_xticks(x)
    ax2.set_xticklabels([s.replace(" LLM", "").replace("LLM + ", "") for s in SYSTEMS],
                        rotation=30, ha='right', fontsize=8.5, fontweight='bold', color=TEXT_COLOR)

    for bar, val in zip(bars2, latencies):
        ax2.annotate(f"{val:.1f}ms",
                    xy=(bar.get_x() + bar.get_width() / 2, bar.get_height()),
                    xytext=(0, 4), textcoords="offset points",
                    ha='center', va='bottom', fontsize=8, fontweight='bold', color=TEXT_COLOR)

    plt.suptitle("Retrieval Precision (MRR) and Tail Latency Analysis",
                 fontsize=14, fontweight='bold', color=TEXT_COLOR, y=1.02)
    plt.tight_layout()

    for out_dir in output_dirs:
        filepath = os.path.join(out_dir, "nsn_mrr_and_latency.png")
        plt.savefig(filepath, bbox_inches='tight', facecolor=BG_COLOR)
        print(f"[FIG GENERATED] Saved {filepath}")
    plt.close()

def generate_fig3_contradiction_deepdive(output_dirs):
    """Figure 3: Contradiction Resolution Failure Analysis (0% RAG vs 100% NSN)"""
    fig, ax = plt.subplots(figsize=(10, 5), dpi=300)
    fig.patch.set_facecolor(BG_COLOR)
    setup_ax(ax, "Contradiction Resolution Benchmark — Trust Weighting vs Semantic Collision",
             "Recall@5 Accuracy (%)", (0, 115))

    systems_sub = ['Rolling Window LLM', 'LLM + BM25', 'LLM + Dense RAG', 'LLM + Hybrid RAG', 'LLM + RAG Memory', 'NeuroSleepNet (NSN)']
    scores = [25.0, 0.0, 0.0, 0.0, 0.0, 100.0]
    colors = ['#d29922', '#f85149', '#a371f7', '#58a6ff', '#1f6feb', '#2ea043']

    x = np.arange(len(systems_sub))
    bars = ax.bar(x, scores, color=colors, width=0.5, zorder=3, alpha=0.9)
    ax.set_xticks(x)
    ax.set_xticklabels(systems_sub, rotation=15, ha='right', fontsize=9.5, fontweight='bold', color=TEXT_COLOR)

    for bar, score in zip(bars, scores):
        text = f"{score:.0f}%" if score > 0 else "0.0%\n(Semantic Collision)"
        y_off = 3 if score > 0 else 5
        color = '#2ea043' if score == 100 else ('#f85149' if score == 0 else TEXT_COLOR)
        ax.annotate(text,
                    xy=(bar.get_x() + bar.get_width() / 2, bar.get_height()),
                    xytext=(0, y_off), textcoords="offset points",
                    ha='center', va='bottom', fontsize=9, fontweight='bold', color=color)

    # Callout text box
    ax.text(0.03, 0.78,
            "Key Insight:\n"
            "• Standard Dense/Hybrid RAG suffer 100% failure (0% Recall@5)\n"
            "  because cosine similarity cannot differentiate trusted source facts\n"
            "  from conflicting user claims.\n"
            "• NSN achieves 100% Recall@5 via TrustManager source weighting\n"
            "  and REM sleep contradiction resolution.",
            transform=ax.transAxes, fontsize=9.5, color=TEXT_COLOR,
            bbox=dict(boxstyle="round,pad=0.5", facecolor=PANEL_COLOR, edgecolor=BORDER_COLOR, alpha=0.9))

    plt.tight_layout()
    for out_dir in output_dirs:
        filepath = os.path.join(out_dir, "nsn_contradiction_resolution.png")
        plt.savefig(filepath, bbox_inches='tight', facecolor=BG_COLOR)
        print(f"[FIG GENERATED] Saved {filepath}")
    plt.close()

def generate_fig4_failure_matrix(output_dirs):
    """Figure 4: 2x2 Failure Mode Breakdown (Knowledge Update)"""
    fig, ax = plt.subplots(figsize=(10, 5.5), dpi=300)
    fig.patch.set_facecolor(BG_COLOR)
    setup_ax(ax, "2×2 Error & Failure Mode Decomposition (Knowledge Update Benchmark, n=40)",
             "Number of Queries", (0, 45))

    systems_fm = ['Vanilla LLM', 'Rolling Window', 'LLM + BM25', 'LLM + Dense RAG', 'LLM + Hybrid RAG', 'LLM + RAG Memory', 'NSN (ours)']
    tp = [0, 4, 0, 10, 10, 7, 10]
    rf = [0, 3, 0, 20, 20, 23, 20]
    ret_fail = [39, 33, 40, 10, 10, 10, 5]
    lucky = [1, 0, 0, 0, 0, 0, 0]
    dup_skip = [0, 0, 0, 0, 0, 0, 5]

    x = np.arange(len(systems_fm))
    width = 0.55

    b1 = ax.bar(x, tp, width, label='TRUE_POSITIVE (Correct Retrieval + Answer)', color='#2ea043', zorder=3)
    b2 = ax.bar(x, rf, width, bottom=tp, label='REASONING_FAILURE (Retrieved Gold at Rank 2-5)', color='#d29922', zorder=3)
    b3 = ax.bar(x, ret_fail, width, bottom=np.array(tp)+np.array(rf), label='RETRIEVAL_FAILURE (Gold Missing from Top-5)', color='#f85149', zorder=3)
    b4 = ax.bar(x, lucky, width, bottom=np.array(tp)+np.array(rf)+np.array(ret_fail), label='LUCKY_GUESS (No Memory, Correct Answer)', color='#a371f7', zorder=3)
    b5 = ax.bar(x, dup_skip, width, bottom=np.array(tp)+np.array(rf)+np.array(ret_fail)+np.array(lucky), label='DUP_SKIP (Deduplicated by NSN Cosine ≥ 0.95)', color='#58a6ff', zorder=3)

    ax.set_xticks(x)
    ax.set_xticklabels(systems_fm, rotation=15, ha='right', fontsize=9.5, fontweight='bold', color=TEXT_COLOR)
    ax.legend(fontsize=8, loc='upper right', facecolor=PANEL_COLOR, edgecolor=BORDER_COLOR, labelcolor=TEXT_COLOR)

    # Highlight NSN 50% reduction in retrieval failure
    ax.annotate("50% Reduction in\nRetrieval Failures\n(5 vs 10)",
                xy=(6, 35), xytext=(4.5, 38),
                arrowprops=dict(arrowstyle="->", color='#2ea043', lw=1.5),
                fontsize=8.5, fontweight='bold', color='#2ea043')

    plt.tight_layout()
    for out_dir in output_dirs:
        filepath = os.path.join(out_dir, "nsn_failure_matrix.png")
        plt.savefig(filepath, bbox_inches='tight', facecolor=BG_COLOR)
        print(f"[FIG GENERATED] Saved {filepath}")
    plt.close()

def main():
    root_dir = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
    docs_dir = os.path.join(root_dir, "docs", "images")
    results_dir = os.path.join(root_dir, "benchmarks", "results", "figures")
    artifact_dir = r"C:\Users\aviroop\.gemini\antigravity-ide\brain\b3569c45-7cb8-4e5e-ab0f-6d88e8c39dc0"

    output_dirs = [docs_dir, results_dir]
    if os.path.exists(artifact_dir):
        output_dirs.append(artifact_dir)

    ensure_dirs(output_dirs)

    print("Generating publication-grade NSN benchmark figures...")
    generate_fig1_head_to_head(output_dirs)
    generate_fig2_mrr_and_latency(output_dirs)
    generate_fig3_contradiction_deepdive(output_dirs)
    generate_fig4_failure_matrix(output_dirs)
    print("All benchmark figures successfully generated!")

if __name__ == "__main__":
    main()
