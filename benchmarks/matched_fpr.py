"""
Matched operating point comparison: Context Checker vs LLM Judges.
(Phase A3 of mcp-hallucination-checker improvement plan)

Evaluates on three datasets:
  1. AgentHallu (human-attributed argument errors, 31 pos / 93 neg)
  2. DeRef fcrb (natural model-made errors, seeded dev sample)
  3. ArgGround-Bench (injected absent-value errors, seeded sample)

For each dataset and judge:
  - Reports judge binary operating point (FBR, Detection) with 95% CIs.
  - Reports checker catch rate at fixed 5% and 10% FBR.
  - Reports checker catch rate at matched judge FBR.
  - Outputs full ROC/PR curve points for the continuous checker.
  - Computes exact McNemar test on paired correctness at the matched point.
  - Flags datasets with n_pos < 100 as exploratory.

Usage:
  python benchmarks/matched_fpr.py [--out benchmarks/matched_fpr_results.json]
"""
from __future__ import annotations

import argparse
import ast
import json
import math
import sys
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
sys.path.insert(0, str(ROOT / "benchmarks"))

from server.pipeline.arg_ground import call_score  # noqa: E402
from benchmarks.stats import (  # noqa: E402
    wilson_interval,
    bootstrap_rate_ci,
    fast_auroc,
    bootstrap_auroc_ci,
    mcnemar_exact,
    check_exploratory,
)

B = ROOT / "benchmarks"


def lit(x: Any) -> Any:
    if isinstance(x, str):
        try:
            return json.loads(x)
        except Exception:
            try:
                return ast.literal_eval(x)
            except Exception:
                return x
    return x


def blocked_from_row(r: Dict[str, Any]) -> bool:
    """Judge verdict -> blocked. Unparseable counts as a miss."""
    return (not r["supported"]) if r.get("parsed", True) else bool(r.get("label", False))


def compute_roc_pr(
    y_true_error: np.ndarray,
    scores: np.ndarray,
    thresholds: Optional[np.ndarray] = None,
) -> Dict[str, Any]:
    """
    scores in [0, 1] where lower score = more likely an error (call is blocked if score < tau).
    y_true_error: True if the call is an error/hallucination, False if clean/consistent.
    """
    if thresholds is None:
        thresholds = np.unique(np.concatenate(([0.0], scores, [1.0001])))
        thresholds.sort()

    n_pos = int(np.sum(y_true_error))
    n_neg = int(np.sum(~y_true_error))

    roc_points = []
    pr_points = []

    for tau in thresholds:
        # blocked if score < tau
        blocked = scores < tau
        tp = int(np.sum(blocked & y_true_error))
        fp = int(np.sum(blocked & ~y_true_error))

        tpr = tp / n_pos if n_pos > 0 else 0.0
        fpr = fp / n_neg if n_neg > 0 else 0.0
        precision = tp / (tp + fp) if (tp + fp) > 0 else 1.0

        roc_points.append({"tau": round(float(tau), 4), "fpr": round(fpr, 4), "tpr": round(tpr, 4)})
        pr_points.append({"tau": round(float(tau), 4), "recall": round(tpr, 4), "precision": round(precision, 4)})

    auroc = fast_auroc(y_true_error.astype(int), 1.0 - scores)
    return {"roc": roc_points, "pr": pr_points, "auroc": round(auroc, 4)}


def get_checker_at_fpr(
    y_true_error: np.ndarray,
    scores: np.ndarray,
    target_fbr: float,
) -> Dict[str, Any]:
    """
    Finds tau such that checker false-block rate (FBR) on clean calls is <= target_fbr (closest).
    Returns catch rate (detection on errors) and exact FBR.
    """
    clean_scores = scores[~y_true_error]
    error_scores = scores[y_true_error]
    n_clean = len(clean_scores)
    n_err = len(error_scores)

    sorted_clean = np.sort(clean_scores)
    # k clean calls blocked if tau > clean_scores[k-1]
    # target k <= target_fbr * n_clean
    k_max = int(math.floor(target_fbr * n_clean))
    if k_max <= 0:
        tau = float(sorted_clean[0]) if n_clean > 0 else 0.0
        # If even lowest score blocks > 0, set tau to 0.0
        actual_k = int(np.sum(clean_scores < tau))
        if actual_k > k_max:
            tau = 0.0
    elif k_max >= n_clean:
        tau = 1.0001
    else:
        # tau strictly above the k_max-th smallest clean score
        tau = float(sorted_clean[k_max - 1]) + 1e-6

    blocked_clean = clean_scores < tau
    blocked_error = error_scores < tau

    fbr = float(np.mean(blocked_clean)) if n_clean > 0 else 0.0
    det = float(np.mean(blocked_error)) if n_err > 0 else 0.0

    _, fbr_ci = bootstrap_rate_ci(blocked_clean, n_resamples=5000)
    _, det_ci = bootstrap_rate_ci(blocked_error, n_resamples=5000)

    return {
        "tau": round(tau, 4),
        "target_fbr": round(target_fbr, 4),
        "fbr": round(fbr, 4),
        "fbr_ci": list(fbr_ci),
        "det": round(det, 4),
        "det_ci": list(det_ci),
        "blocked_all": scores < tau,
    }


DATASETS = {
    "agenthallu": {
        "name": "AgentHallu (human-attributed)",
        "cases_file": "data/agenthallu/cases_eval.json",
        "judges": [
            ("Claude Opus (agent)", "llm_judge_args_ah_agent_opus_dev_v0.json"),
            ("Claude Sonnet (agent)", "llm_judge_args_ah_agent_sonnet_dev_v0.json"),
            ("Claude Haiku (agent)", "llm_judge_args_ah_agent_haiku_dev_v0.json"),
            ("Nemotron 550B (API t=0)", "llm_judge_args_ah_nvidia_nemotron_3_ultra_550b_a55b_dev_v0.json"),
            ("Nemotron 4B (API t=0)", "llm_judge_args_ah_nvidia_nemotron_3_5_bolt_e4b_dev_v0.json"),
        ],
    },
    "fcrb": {
        "name": "DeRef fcrb (natural errors)",
        "cases_file": "data/deref/cases_fcrb.json",
        "judges": [
            ("Claude Opus (agent)", "llm_judge_args_fcrb_agent_opus_dev_v0.json"),
            ("Claude Sonnet (agent)", "llm_judge_args_fcrb_agent_sonnet_dev_v0.json"),
            ("Claude Haiku (agent)", "llm_judge_args_fcrb_agent_haiku_dev_v0.json"),
            ("Nemotron 550B (API t=0)", "llm_judge_args_fcrb_nvidia_nvidia_nemotron_3_ultra_550b_a55b_dev_v0.json"),
            ("Nemotron 4B (API t=0)", "llm_judge_args_fcrb_nvidia_nvidia_nemotron_3_5_bolt_e4b_dev_v0.json"),
            ("gpt-oss-120b (Groq t=0)", "llm_judge_args_fcrb_groq_openai_gpt_oss_120b_dev_v0.json"),
            ("gpt-oss-20b (Groq t=0)", "llm_judge_args_fcrb_groq_openai_gpt_oss_20b_dev_v0.json"),
        ],
    },
    "injected": {
        "name": "ArgGround-Bench (injected errors)",
        "cases_file": "argground_bench.json",
        "judges": [
            ("Claude Opus (agent)", "llm_judge_args_injected_agent_opus_dev_v0.json"),
            ("Claude Sonnet (agent)", "llm_judge_args_injected_agent_sonnet_dev_v0.json"),
            ("Claude Haiku (agent)", "llm_judge_args_injected_agent_haiku_dev_v0.json"),
            ("gpt-oss-120b (Groq t=0)", "llm_judge_args_injected_gpt_oss_120b_dev_v0.json"),
            ("gpt-oss-20b (Groq t=0)", "llm_judge_args_injected_gpt_oss_20b_dev_v0.json"),
        ],
    },
}


def evaluate_dataset(tag: str, cfg: Dict[str, Any]) -> Dict[str, Any]:
    c_path = B / cfg["cases_file"]
    if not c_path.exists():
        print(f"Skipping {tag}: cases file {c_path} not found")
        return {}

    raw_cases = json.loads(c_path.read_text(encoding="utf-8"))
    case_list = raw_cases.get("cases", raw_cases)
    cases_by_id = {c["id"]: c for c in case_list}

    # Find common IDs present across all available judge files for this benchmark
    available_judges = []
    for j_name, j_file in cfg["judges"]:
        jp = B / j_file
        if jp.exists():
            available_judges.append((j_name, j_file, jp))

    if not available_judges:
        print(f"No judge files found for {tag}")
        return {}

    # Read judge row IDs
    first_judge = json.loads(available_judges[0][2].read_text(encoding="utf-8"))["rows"]
    eval_ids = [r["id"] for r in first_judge if r["id"] in cases_by_id]

    print(f"\n=======================================================")
    print(f"Evaluating {cfg['name']} (n={len(eval_ids)} calls)")
    print(f"=======================================================")

    # Prepare ground truth and checker scores
    y_error_list = []
    scores_list = []
    for cid in eval_ids:
        c = cases_by_id[cid]
        # label: True = clean/consistent, False = hallucinated/error
        lbl = c.get("label")
        if isinstance(lbl, str):
            lbl = lbl.lower() in ("true", "1")
        is_error = not bool(lbl)
        y_error_list.append(is_error)

        args = lit(c.get("call", {})).get("arguments", {}) if "call" in c else lit(c.get("arguments", {}))
        premise = c.get("premise", "")
        sc = call_score(args or {}, premise)
        scores_list.append(sc)

    y_error = np.array(y_error_list, dtype=bool)
    scores = np.array(scores_list, dtype=float)

    n_pos = int(np.sum(y_error))
    n_neg = int(np.sum(~y_error))
    exp_info = check_exploratory(n_pos, label=cfg["name"])
    print(f"Dataset breakdown: {n_pos} errors, {n_neg} clean calls.")
    if exp_info["is_exploratory"]:
        print(f"[{exp_info['status']}] {exp_info['note']}")

    # Continuous checker ROC / PR curves
    curves = compute_roc_pr(y_error, scores)
    print(f"Checker continuous AUROC: {curves['auroc']}")

    # Fixed false-block operating points for the checker
    chk_at_5 = get_checker_at_fpr(y_error, scores, 0.05)
    chk_at_10 = get_checker_at_fpr(y_error, scores, 0.10)
    print(f"Checker @ 5% FBR:  Catch={chk_at_5['det']:.1%} [{chk_at_5['det_ci'][0]:.1%},{chk_at_5['det_ci'][1]:.1%}] (actual FBR={chk_at_5['fbr']:.1%})")
    print(f"Checker @ 10% FBR: Catch={chk_at_10['det']:.1%} [{chk_at_10['det_ci'][0]:.1%},{chk_at_10['det_ci'][1]:.1%}] (actual FBR={chk_at_10['fbr']:.1%})")

    # Matched comparison with each judge
    judge_comparisons = []
    print("\n--- Judge Operating Points vs Matched Checker ---")
    print("%-24s | %-16s | %-16s | %-16s | %-16s" % (
        "Judge", "Judge FBR (95% CI)", "Judge Catch (95% CI)", "Checker @ Match", "McNemar (J vs Chk)"
    ))

    for j_name, j_file, jp in available_judges:
        j_data = {r["id"]: r for r in json.loads(jp.read_text(encoding="utf-8"))["rows"]}
        if not all(cid in j_data for cid in eval_ids):
            continue

        j_blocked = np.array([blocked_from_row(j_data[cid]) for cid in eval_ids], dtype=bool)
        j_fbr = float(np.mean(j_blocked[~y_error]))
        j_det = float(np.mean(j_blocked[y_error]))

        _, j_fbr_ci = bootstrap_rate_ci(j_blocked[~y_error], n_resamples=5000)
        _, j_det_ci = bootstrap_rate_ci(j_blocked[y_error], n_resamples=5000)

        # Checker evaluated at matching Judge FBR
        chk_matched = get_checker_at_fpr(y_error, scores, j_fbr)

        # Paired McNemar: correctness of judge vs correctness of matched checker
        # Correct on clean: blocked is False; Correct on error: blocked is True
        j_correct = (j_blocked == y_error)
        chk_correct = (chk_matched["blocked_all"] == y_error)
        mc = mcnemar_exact(j_correct, chk_correct)

        row = {
            "judge_name": j_name,
            "verdict_type": "binary (single operating point)",
            "judge_fbr": round(j_fbr, 4),
            "judge_fbr_ci": list(j_fbr_ci),
            "judge_det": round(j_det, 4),
            "judge_det_ci": list(j_det_ci),
            "checker_matched_fbr": chk_matched["fbr"],
            "checker_matched_det": chk_matched["det"],
            "checker_matched_det_ci": chk_matched["det_ci"],
            "mcnemar": mc,
        }
        judge_comparisons.append(row)

        print("%-24s | %5.1f%% [%.1f,%.1f] | %5.1f%% [%.1f,%.1f] | %5.1f%% [%.1f,%.1f] | %d vs %d (p=%.4f)" % (
            j_name,
            100 * j_fbr, 100 * j_fbr_ci[0], 100 * j_fbr_ci[1],
            100 * j_det, 100 * j_det_ci[0], 100 * j_det_ci[1],
            100 * chk_matched["det"], 100 * chk_matched["det_ci"][0], 100 * chk_matched["det_ci"][1],
            mc["only_a_correct"], mc["only_b_correct"], mc["p"],
        ))

    return {
        "dataset_name": cfg["name"],
        "n_calls": len(eval_ids),
        "n_positives": n_pos,
        "n_negatives": n_neg,
        "exploratory": exp_info,
        "checker_auroc": curves["auroc"],
        "checker_at_5pct_fbr": chk_at_5,
        "checker_at_10pct_fbr": chk_at_10,
        "judge_comparisons": judge_comparisons,
        "roc_curve": curves["roc"],
        "pr_curve": curves["pr"],
    }


def main():
    parser = argparse.ArgumentParser(description="Matched FPR evaluation")
    parser.add_argument("--out", default=str(B / "matched_fpr_results.json"))
    args = parser.parse_args()

    results = {}
    for tag, cfg in DATASETS.items():
        res = evaluate_dataset(tag, cfg)
        if res:
            results[tag] = res

    Path(args.out).write_text(json.dumps(results, indent=2, default=str), encoding="utf-8")
    print(f"\nWrote matched FPR results to {args.out}")


if __name__ == "__main__":
    main()
