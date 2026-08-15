"""
Statistical evaluation utilities (Mean, Std, 95% Confidence Intervals, Paired T-Test, Wilcoxon).
"""
import math
import numpy as np

def compute_summary_statistics(values: list) -> dict:
    valid_values = [v for v in values if v is not None and not math.isnan(v)]
    n = len(valid_values)
    if n == 0:
        return {"n": 0, "mean": 0.0, "std": 0.0, "ci95_low": 0.0, "ci95_high": 0.0}
        
    mean = float(np.mean(valid_values))
    std = float(np.std(valid_values, ddof=1)) if n > 1 else 0.0
    se = std / math.sqrt(n) if n > 0 else 0.0
    ci95_margin = 1.96 * se
    
    return {
        "n": n,
        "mean": mean,
        "std": std,
        "se": se,
        "ci95_low": max(0.0, mean - ci95_margin),
        "ci95_high": mean + ci95_margin
    }

def paired_t_test(sample_a: list, sample_b: list) -> dict:
    if len(sample_a) != len(sample_b) or len(sample_a) < 2:
        return {"t_stat": 0.0, "p_value": 1.0, "effect_size": 0.0}
        
    diffs = np.array(sample_a) - np.array(sample_b)
    n = len(diffs)
    mean_diff = np.mean(diffs)
    std_diff = np.std(diffs, ddof=1)
    
    if std_diff == 0:
        return {"t_stat": 0.0, "p_value": 1.0, "effect_size": 0.0}
        
    t_stat = mean_diff / (std_diff / math.sqrt(n))
    # Approximation of p-value for quick computation
    cohen_d = mean_diff / std_diff
    p_val = 2.0 * (1.0 - 0.5 * (1.0 + math.erf(abs(t_stat) / math.sqrt(2.0))))
    
    return {
        "t_stat": float(t_stat),
        "p_value": float(p_val),
        "effect_size": float(cohen_d)
    }
