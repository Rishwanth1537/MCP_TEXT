"""
Unit tests for benchmarks/stats.py
"""
import math
import numpy as np
import pytest

from benchmarks.stats import (
    wilson_interval,
    bootstrap_rate_ci,
    fast_auroc,
    bootstrap_auroc_ci,
    mcnemar_exact,
    check_exploratory,
    format_rate_with_ci,
)


def test_wilson_interval():
    low, high = wilson_interval(50, 100)
    assert 0.39 <= low <= 0.42
    assert 0.58 <= high <= 0.61
    assert low < 0.5 < high

    # Edge cases
    assert wilson_interval(0, 10) == (0.0, 0.2775)
    assert wilson_interval(10, 10) == (0.7225, 1.0)
    assert math.isnan(wilson_interval(0, 0)[0])


def test_bootstrap_rate_ci():
    values = [1] * 60 + [0] * 40
    point, (low, high) = bootstrap_rate_ci(values, n_resamples=1000, seed=123)
    assert point == 0.6
    assert 0.48 <= low <= 0.53
    assert 0.67 <= high <= 0.72


def test_fast_auroc_and_bootstrap():
    y_true = np.array([0, 0, 0, 1, 1, 1])
    y_score = np.array([0.1, 0.2, 0.3, 0.7, 0.8, 0.9])
    auc = fast_auroc(y_true, y_score)
    assert auc == 1.0

    # Intermediate AUROC
    y_true2 = [0, 0, 1, 1]
    y_score2 = [0.1, 0.8, 0.2, 0.9]
    assert fast_auroc(np.array(y_true2), np.array(y_score2)) == 0.75

    point, (low, high) = bootstrap_auroc_ci(y_true2, y_score2, n_resamples=500, seed=42)
    assert point == 0.75
    assert 0.0 <= low <= high <= 1.0


def test_mcnemar_exact():
    # Symmetric discordant pairs: p = 1.0
    a = [True, True, False, False]
    b = [True, False, True, False]
    res = mcnemar_exact(a, b)
    assert res["only_a_correct"] == 1
    assert res["only_b_correct"] == 1
    assert res["p"] == 1.0
    assert not res["significant_at_05"]

    # Strongly asymmetric: 12 vs 1
    a_long = [True] * 12 + [False]
    b_long = [False] * 12 + [True]
    res2 = mcnemar_exact(a_long, b_long)
    assert res2["only_a_correct"] == 12
    assert res2["only_b_correct"] == 1
    assert res2["p"] < 0.01
    assert res2["significant_at_05"]


def test_check_exploratory():
    res_small = check_exploratory(31, label="AgentHallu")
    assert res_small["is_exploratory"]
    assert "EXPLORATORY" in res_small["status"]
    assert "31" in res_small["note"]

    res_large = check_exploratory(150, label="Full benchmark")
    assert not res_large["is_exploratory"]
    assert "SUFFICIENT" in res_large["status"]


def test_format_rate_with_ci():
    s = format_rate_with_ci(19, 31, method="wilson")
    assert "61.3%" in s
    assert "n=31" in s
