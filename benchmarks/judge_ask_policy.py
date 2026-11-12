"""
J5 Panel Aggregation & Three-Tier Ask Policy:
- Combines 2, 3, or 5 judge error-scores (mean, max, or grouped-cross-fitted logistic regression).
- Routes each call into ALLOW (score <= tau_allow), ASK (tau_allow < score < tau_block), or BLOCK (score >= tau_block).
- Reports ask rate on correct calls, false-block rate, error catch rates (BLOCK-only & union), AUROC, ECE, and per-member contributions.
"""
from __future__ import annotations

import sys
from pathlib import Path
from typing import Any, Dict, List, Optional, Sequence, Tuple

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "benchmarks"))

from stats import check_exploratory, fast_auroc, wilson_interval  # noqa: E402


def _fit_logistic_l2(X: np.ndarray, y: np.ndarray, reg: float = 1.0, max_iter: int = 200) -> Tuple[np.ndarray, float]:
    """Pure-NumPy L2-regularized logistic regression via IRLS / Newton-Raphson."""
    n, d = X.shape
    Xb = np.hstack([X, np.ones((n, 1), dtype=float)])
    w = np.zeros(d + 1, dtype=float)
    pen = np.eye(d + 1, dtype=float) * reg
    pen[-1, -1] = 0.0  # do not regularize intercept
    for _ in range(max_iter):
        z = np.clip(Xb @ w, -30.0, 30.0)
        p = 1.0 / (1.0 + np.exp(-z))
        grad = Xb.T @ (p - y) + pen @ w
        W = np.maximum(p * (1.0 - p), 1e-6)
        H = (Xb.T * W) @ Xb + pen
        step = np.linalg.solve(H, grad)
        w -= step
        if np.linalg.norm(step) < 1e-6:
            break
    return w[:-1], float(w[-1])


def _group_kfold_indices(groups: Sequence[str], n_splits: int) -> List[Tuple[np.ndarray, np.ndarray]]:
    g = np.asarray(list(groups))
    uniq = sorted(set(g))
    k = min(n_splits, len(uniq))
    group_to_fold = {grp: idx % k for idx, grp in enumerate(uniq)}
    folds = np.array([group_to_fold[grp] for grp in g])
    splits = []
    for f in range(k):
        va = np.where(folds == f)[0]
        tr = np.where(folds != f)[0]
        splits.append((tr, va))
    return splits


def expected_calibration_error(y_true_err: np.ndarray, probs: np.ndarray, n_bins: int = 10) -> float:
    """Compute ECE across equal-width probability bins."""
    y = np.asarray(y_true_err, dtype=float)
    p = np.clip(np.asarray(probs, dtype=float), 0.0, 1.0)
    bins = np.linspace(0.0, 1.0, n_bins + 1)
    ece = 0.0
    n = len(y)
    if n == 0:
        return 0.0
    for i in range(n_bins):
        lo, hi = bins[i], bins[i + 1]
        mask = (p >= lo) & (p <= hi if i == n_bins - 1 else p < hi)
        if mask.any():
            acc = float(y[mask].mean())
            conf = float(p[mask].mean())
            ece += (float(mask.sum()) / n) * abs(acc - conf)
    return float(ece)


def combine_panel_scores(
    score_matrix: np.ndarray,
    method: str = "mean",
    y_true_err: Optional[np.ndarray] = None,
    groups: Optional[Sequence[str]] = None,
    n_splits: int = 5,
) -> np.ndarray:
    """Combine (N, M) judge error-probabilities into an (N,) panel score."""
    X = np.asarray(score_matrix, dtype=float)
    if X.ndim != 2 or X.shape[0] == 0:
        raise ValueError("score_matrix must be a non-empty 2D array (n_cases, n_judges).")
    if method == "mean":
        return X.mean(axis=1)
    if method == "max":
        return X.max(axis=1)
    if method == "logistic":
        if y_true_err is None or groups is None:
            raise ValueError("logistic combiner requires y_true_err and groups for grouped cross-fitting.")
        y = np.asarray(y_true_err, dtype=int)
        g = np.asarray(list(groups))
        unique_g = np.unique(g)
        k = min(n_splits, len(unique_g))
        if k < 2:
            raise ValueError("Need at least 2 distinct tool/task groups for grouped cross-fitting.")
        oof = np.zeros(len(y), dtype=float)
        for tr, va in _group_kfold_indices(g, k):
            if len(np.unique(y[tr])) < 2:
                oof[va] = X[va].mean(axis=1)
                continue
            w, b = _fit_logistic_l2(X[tr], y[tr])
            z = np.clip(X[va] @ w + b, -30.0, 30.0)
            oof[va] = 1.0 / (1.0 + np.exp(-z))
        return oof
    raise ValueError(f"Unknown panel combination method: {method}")


def apply_ask_policy(
    panel_scores: np.ndarray,
    tau_allow: float,
    tau_block: float,
) -> np.ndarray:
    """Return array of decisions in {'ALLOW', 'ASK', 'BLOCK'}."""
    if tau_allow > tau_block:
        raise ValueError(f"tau_allow ({tau_allow}) must be <= tau_block ({tau_block}).")
    s = np.asarray(panel_scores, dtype=float)
    out = np.full(len(s), "ASK", dtype=object)
    out[s <= tau_allow] = "ALLOW"
    out[s >= tau_block] = "BLOCK"
    return out


def evaluate_ask_policy(
    score_matrix: np.ndarray,
    y_true_err: np.ndarray,
    tau_allow: float,
    tau_block: float,
    method: str = "mean",
    groups: Optional[Sequence[str]] = None,
    judge_names: Optional[List[str]] = None,
) -> Dict[str, Any]:
    y_err = np.asarray(y_true_err, dtype=bool)
    y_ok = ~y_err
    panel_scores = combine_panel_scores(score_matrix, method=method, y_true_err=y_err, groups=groups)
    decisions = apply_ask_policy(panel_scores, tau_allow=tau_allow, tau_block=tau_block)

    is_block = decisions == "BLOCK"
    is_ask = decisions == "ASK"
    is_any = is_block | is_ask

    n_ok = int(y_ok.sum())
    n_err = int(y_err.sum())

    fb_block = int(is_block[y_ok].sum())
    ask_ok = int(is_ask[y_ok].sum())
    catch_block = int(is_block[y_err].sum())
    catch_union = int(is_any[y_err].sum())

    # Per-member contribution: fraction of panel BLOCK/ASK calls where member individually flagged (>= 0.5)
    names = judge_names or [f"judge_{i}" for i in range(score_matrix.shape[1])]
    contributions = {}
    for j, name in enumerate(names):
        member_flag = score_matrix[:, j] >= 0.5
        contributions[name] = {
            "share_of_panel_block": float(member_flag[is_block].mean()) if is_block.any() else 0.0,
            "share_of_panel_ask": float(member_flag[is_ask].mean()) if is_ask.any() else 0.0,
        }

    return {
        "method": method,
        "tau_allow": float(tau_allow),
        "tau_block": float(tau_block),
        "n_total": int(len(y_err)),
        "n_correct": n_ok,
        "n_wrong": n_err,
        "exploratory": check_exploratory(n_err, label="ask_policy"),
        "false_block_rate": fb_block / n_ok if n_ok else 0.0,
        "false_block_ci_wilson": wilson_interval(fb_block, n_ok),
        "ask_rate_correct": ask_ok / n_ok if n_ok else 0.0,
        "ask_rate_ci_wilson": wilson_interval(ask_ok, n_ok),
        "catch_rate_block": catch_block / n_err if n_err else 0.0,
        "catch_rate_block_ci_wilson": wilson_interval(catch_block, n_err),
        "catch_rate_union": catch_union / n_err if n_err else 0.0,
        "catch_rate_union_ci_wilson": wilson_interval(catch_union, n_err),
        "auroc": float(fast_auroc(y_err, panel_scores)) if (n_ok > 0 and n_err > 0) else None,
        "ece": expected_calibration_error(y_err, panel_scores),
        "member_contributions": contributions,
    }
