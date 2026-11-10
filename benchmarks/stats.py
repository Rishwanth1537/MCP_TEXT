"""
Statistical reporting utilities for MCP Hallucination Checker benchmarks.

Provides:
- Stratified bootstrap 95% confidence intervals (>= 10,000 resamples) for rate metrics and AUROC.
- Exact McNemar test (binomial test on discordant pairs) for paired detector comparisons.
- Exploratory sample-size flagging for comparisons with fewer than 100 positive cases.
- Wilson score interval helper for proportions.
"""
from __future__ import annotations

import math
from typing import Any, Dict, List, Optional, Sequence, Tuple, Union
import numpy as np


def wilson_interval(k: int, n: int, confidence: float = 0.95) -> Tuple[float, float]:
    """
    Wilson score interval for a binomial proportion k / n.
    Returns (ci_low, ci_high) in [0, 1].
    """
    if n <= 0:
        return (float("nan"), float("nan"))
    if k < 0 or k > n:
        raise ValueError(f"k ({k}) must be between 0 and n ({n})")
    
    # 95% -> z ~ 1.95996
    z = 1.959963984540054 if abs(confidence - 0.95) < 1e-4 else float(
        np.abs(np.quantile(np.random.standard_normal(100000), 1 - (1 - confidence) / 2))
    )
    p = k / n
    denom = 1.0 + (z * z) / n
    center = (p + (z * z) / (2.0 * n)) / denom
    margin = (z * math.sqrt(p * (1.0 - p) / n + (z * z) / (4.0 * n * n))) / denom
    
    low = max(0.0, center - margin)
    high = min(1.0, center + margin)
    return (round(low, 4), round(high, 4))


def bootstrap_rate_ci(
    values: Sequence[Union[bool, int, float]],
    n_resamples: int = 10000,
    alpha: float = 0.05,
    seed: int = 42,
) -> Tuple[float, Tuple[float, float]]:
    """
    Bootstrap confidence interval for a single binary rate (e.g. false block rate, recall).
    Returns: (point_estimate, (ci_low, ci_high))
    """
    arr = np.asarray(values, dtype=float)
    n = len(arr)
    if n == 0:
        return (float("nan"), (float("nan"), float("nan")))
    
    point_est = float(np.mean(arr))
    if n == 1 or np.all(arr == arr[0]):
        return (round(point_est, 4), (round(point_est, 4), round(point_est, 4)))

    rng = np.random.default_rng(seed)
    # Generate bootstrap sample means
    idx = rng.integers(0, n, size=(n_resamples, n))
    boot_means = np.mean(arr[idx], axis=1)
    
    low = float(np.percentile(boot_means, 100 * (alpha / 2.0)))
    high = float(np.percentile(boot_means, 100 * (1.0 - alpha / 2.0)))
    return (round(point_est, 4), (round(max(0.0, low), 4), round(min(1.0, high), 4)))


def fast_auroc(y_true: np.ndarray, y_score: np.ndarray) -> float:
    """Fast rank-based computation of AUROC using Mann-Whitney U."""
    pos = y_score[y_true == 1]
    neg = y_score[y_true == 0]
    n_pos, n_neg = len(pos), len(neg)
    if n_pos == 0 or n_neg == 0:
        return float("nan")
    
    # Ranks of all scores
    all_scores = np.concatenate([pos, neg])
    ranks = np.argsort(np.argsort(all_scores)) + 1
    rank_pos = ranks[:n_pos]
    u = np.sum(rank_pos) - n_pos * (n_pos + 1) / 2.0
    return float(u / (n_pos * n_neg))


def bootstrap_auroc_ci(
    y_true: Sequence[Union[bool, int]],
    y_score: Sequence[float],
    n_resamples: int = 10000,
    alpha: float = 0.05,
    seed: int = 42,
) -> Tuple[float, Tuple[float, float]]:
    """
    Stratified bootstrap confidence interval for AUROC.
    Resamples positive and negative examples independently with replacement.
    Returns: (point_estimate, (ci_low, ci_high))
    """
    y_t = np.asarray(y_true, dtype=int)
    y_s = np.asarray(y_score, dtype=float)
    if len(y_t) != len(y_s):
        raise ValueError("Length of y_true and y_score must match")
    
    pos_scores = y_s[y_t == 1]
    neg_scores = y_s[y_t == 0]
    n_pos, n_neg = len(pos_scores), len(neg_scores)
    
    if n_pos == 0 or n_neg == 0:
        return (float("nan"), (float("nan"), float("nan")))
    
    point_est = fast_auroc(y_t, y_s)
    
    rng = np.random.default_rng(seed)
    pos_idx = rng.integers(0, n_pos, size=(n_resamples, n_pos))
    neg_idx = rng.integers(0, n_neg, size=(n_resamples, n_neg))
    
    # Vectorized / batched Mann-Whitney calculation across resamples
    # For large n_resamples, calculate pairwise rank statistics
    # U = sum_{i,j} I(pos[i] > neg[j]) + 0.5 * I(pos[i] == neg[j])
    boot_aurocs = np.empty(n_resamples, dtype=float)
    for b in range(n_resamples):
        bp = pos_scores[pos_idx[b]]
        bn = neg_scores[neg_idx[b]]
        # Calculate AUC for this bootstrap resample
        all_s = np.concatenate([bp, bn])
        ranks = np.argsort(np.argsort(all_s)) + 1
        u = np.sum(ranks[:n_pos]) - n_pos * (n_pos + 1) / 2.0
        boot_aurocs[b] = u / (n_pos * n_neg)
        
    low = float(np.percentile(boot_aurocs, 100 * (alpha / 2.0)))
    high = float(np.percentile(boot_aurocs, 100 * (1.0 - alpha / 2.0)))
    return (round(point_est, 4), (round(max(0.0, low), 4), round(min(1.0, high), 4)))


def mcnemar_exact(
    a_correct: Sequence[bool],
    b_correct: Sequence[bool],
) -> Dict[str, Any]:
    """
    Exact two-sided McNemar test for two detectors on identical evaluation cases.
    Computes exact binomial p-value on discordant pairs.
    """
    a = [bool(x) for x in a_correct]
    b = [bool(x) for x in b_correct]
    if len(a) != len(b):
        raise ValueError(f"Lengths must match ({len(a)} vs {len(b)})")
    
    only_a = sum(1 for x, y in zip(a, b) if x and not y)
    only_b = sum(1 for x, y in zip(a, b) if not x and y)
    n_discordant = only_a + only_b
    
    if n_discordant == 0:
        return {
            "only_a_correct": 0,
            "only_b_correct": 0,
            "n_discordant": 0,
            "p": 1.0,
            "significant_at_05": False,
        }
    
    k = min(only_a, only_b)
    # Two-sided exact binomial test with p=0.5: 2 * P(X <= min(only_a, only_b))
    p_val = min(1.0, 2.0 * sum(math.comb(n_discordant, i) for i in range(k + 1)) / (2.0 ** n_discordant))
    
    return {
        "only_a_correct": only_a,
        "only_b_correct": only_b,
        "n_discordant": n_discordant,
        "p": round(p_val, 4),
        "significant_at_05": p_val < 0.05,
    }


def check_exploratory(n_positives: int, label: str = "dataset") -> Dict[str, Any]:
    """
    Flags any evaluation or comparison with fewer than 100 positive cases as 'exploratory'.
    Ensures statistical power caveats are visible.
    """
    is_exploratory = n_positives < 100
    status = "EXPLORATORY (n_pos < 100)" if is_exploratory else "SUFFICIENT (n_pos >= 100)"
    note = (
        f"Warning: {label} has only {n_positives} positive cases (< 100). "
        "Statistical power is limited; confidence intervals will be wide. "
        "Results must be reported as exploratory, not confirmatory."
        if is_exploratory else f"{label} has {n_positives} positive cases."
    )
    return {
        "is_exploratory": is_exploratory,
        "n_positives": n_positives,
        "status": status,
        "note": note,
    }


def format_rate_with_ci(
    k: int,
    n: int,
    method: str = "wilson",
    n_resamples: int = 10000,
) -> str:
    """
    Convenience formatter: returns string like '58.1% [40.8, 73.6] (n=31)'.
    """
    if n == 0:
        return "N/A (n=0)"
    rate = (k / n) * 100.0
    if method == "wilson":
        ci_low, ci_high = wilson_interval(k, n)
    else:
        sample = [1] * k + [0] * (n - k)
        _, (ci_low, ci_high) = bootstrap_rate_ci(sample, n_resamples=n_resamples)
    return f"{rate:.1f}% [{ci_low * 100:.1f}, {ci_high * 100:.1f}] (n={n})"
