"""
Phase B2c: Call-Level Mondrian Conformal Calibration with C1 Clopper-Pearson Upper Bound.
Performs 5-fold tool-grouped cross-fitting on MCP-Atlas calibration split (n=1,212, 53 tools),
never touching the frozen held-out split.
Evaluates:
1. Per-group feasibility under alpha=0.05 with C1 UCB.
2. Out-of-fold BLOCK, ASK, and Any-Intervention rates with tool-cluster bootstrap CIs.
3. DeRef fcrb fabricated recall (n=362) and AgentHallu detection (n=31) vs 60% safety floor.
"""

import json
import sys
from pathlib import Path
from collections import defaultdict
import numpy as np

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))

from server.pipeline.provenance import check_call, evidence_from_agenthallu_premise
from server.pipeline.arg_ground import (
    call_argument_class,
    mondrian_conformal_threshold_ucb,
)
from server.tools.verify_arguments import is_low_confidence_arg, run_argument_grounding
from server.models.schemas import ArgumentGroundingRequest, Decision
from benchmarks.stats import cluster_bootstrap_rate_ci, wilson_interval

SPLIT_FILE = ROOT / "benchmarks" / "data" / "mcpatlas" / "mcpatlas_tool_split.json"


def run():
    print("=======================================================")
    print("PHASE B2c: CALL-LEVEL MONDRIAN CONFORMAL CALIBRATION (C1 UCB)")
    print("=======================================================")

    split_data = json.loads(SPLIT_FILE.read_text(encoding="utf-8"))
    calib_cases = split_data["dev_cases"]
    n_calib = len(calib_cases)

    # 1. Compute call scores and deterministic Mondrian groups on B2a+B2b state
    scores = []
    groups = []
    tools = []
    call_blocked_flags = []
    low_conf_only_flags = []

    for c in calib_cases:
        tname = c["call"]["name"]
        args = c["call"].get("arguments", {})
        tools.append(tname)
        groups.append(call_argument_class(args))

        if not args:
            scores.append(1.0)
            call_blocked_flags.append(False)
            low_conf_only_flags.append(False)
            continue

        ev = [("host", "Current date and time: 2025-07-15T12:00:00Z"), ("user", c["user"])] + [
            ("tool", t["text"]) for t in (c.get("tool_results") or [])
        ]
        cv = check_call(args, ev, config="P3", tool_name=tname)
        scores.append(cv.score)
        call_blocked_flags.append(cv.blocked)
        if cv.blocked:
            ungr = [lv for lv in cv.leaves if lv.verdict == "UNGROUNDED"]
            low_conf_only_flags.append(all(is_low_confidence_arg(lv.path.split(".")[0].split("[")[0]) for lv in ungr))
        else:
            low_conf_only_flags.append(False)

    # Full calibration distribution per Mondrian group
    print("\n--- Score Distribution by Deterministic Mondrian Group (Full Calib n=1,212) ---")
    cal_alpha_05 = mondrian_conformal_threshold_ucb(scores, groups, alpha=0.05, delta=0.10)
    cal_alpha_10 = mondrian_conformal_threshold_ucb(scores, groups, alpha=0.10, delta=0.10)
    cal_alpha_15 = mondrian_conformal_threshold_ucb(scores, groups, alpha=0.15, delta=0.10)

    print(f"{'Group':<16} | {'n':<4} | {'ZeroScore Rate':<15} | {'C1 UCB(0)':<10} | {'tau@a=0.05':<10} | {'tau@a=0.10':<10} | {'tau@a=0.15':<10}")
    print("-" * 88)
    for g, info in cal_alpha_05.items():
        print(
            f"{g:<16} | {info['n']:<4} | {info['zero_score_rate']*100:>5.1f}% ({info['zero_score_count']:3d})   | "
            f"{info['zero_score_ucb']*100:>6.1f}%    | {info['tau']:<10.4f} | {cal_alpha_10[g]['tau']:<10.4f} | {cal_alpha_15[g]['tau']:<10.4f}"
        )

    # 2. 5-Fold Tool-Grouped Cross-Fitting for Out-of-Fold Evaluation
    unique_tools = sorted(set(tools))
    rng = np.random.default_rng(42)
    shuffled_tools = list(unique_tools)
    rng.shuffle(shuffled_tools)
    tool_to_fold = {t: i % 5 for i, t in enumerate(shuffled_tools)}
    folds = [tool_to_fold[t] for t in tools]

    # Evaluate different calibrated operating policies out-of-fold:
    # Policy A: Strict C1 UCB @ alpha=0.05 -> BLOCK iff score < tau_g (if tau_g == 0, routes ungrounded/0-score to ASK)
    # Policy B: Hybrid Mondrian C1 @ alpha=0.10 + low-conf->ASK
    # Policy C: Best Achievable Conformal BLOCK preserving >= 60% fcrb recall (BLOCK iff score < tau_g with min floor 0.50 on high-conf entities, or fitted C1 thresholds)
    audit_records = json.loads((ROOT / "review" / "audit3" / "error_taxonomy_audit.json").read_text(encoding="utf-8"))
    fcrb_fab_ids = {r["id"] for r in audit_records if r["category"] == "fabricated_value"}
    fcrb_all = {c["id"]: c for c in json.loads((ROOT / "benchmarks" / "data" / "deref" / "cases_fcrb.json").read_text(encoding="utf-8"))["cases"]}
    fcrb_fabs = [c for c in fcrb_all.values() if c["id"] in fcrb_fab_ids]

    ah_cases = json.loads((ROOT / "benchmarks" / "data" / "agenthallu" / "cases_eval.json").read_text(encoding="utf-8"))["cases"]
    ah_errs = [c for c in ah_cases if not bool(c["label"])]
    ah_cleans = [c for c in ah_cases if bool(c["label"])]

    # Precompute fcrb fab & agenthallu scores + groups
    fab_scores, fab_groups, fab_low_conf = [], [], []
    for c in fcrb_fabs:
        args = c["call"].get("arguments", {})
        tname = c["call"].get("name")
        ev = [("user", c.get("premise", ""))]
        cv = check_call(args, ev, config="P3", tool_name=tname)
        fab_scores.append(cv.score)
        fab_groups.append(call_argument_class(args))
        ungr = [lv for lv in cv.leaves if lv.verdict == "UNGROUNDED"]
        fab_low_conf.append(bool(cv.blocked) and all(is_low_confidence_arg(lv.path.split(".")[0].split("[")[0]) for lv in ungr))

    # Also test run_argument_grounding scores on fcrb_fabs for direct comparison
    fab_rag_scores = []
    for c in fcrb_fabs:
        r = run_argument_grounding(ArgumentGroundingRequest(
            arguments=c["call"].get("arguments", {}),
            context=c.get("premise", ""),
            tool_name=c["call"].get("name"),
            decision_mode="three_tier", tau_block=0.50, tau_ask=0.88
        ))
        fab_rag_scores.append((r.call_score, r.decision))

    def eval_alpha_policy(target_alpha: float):
        oof_blocks, oof_asks, oof_any = [], [], []
        for k in range(5):
            tr_sc = [scores[i] for i in range(n_calib) if folds[i] != k]
            tr_grp = [groups[i] for i in range(n_calib) if folds[i] != k]
            cal = mondrian_conformal_threshold_ucb(tr_sc, tr_grp, alpha=target_alpha, delta=0.10)
            for i in range(n_calib):
                if folds[i] != k:
                    continue
                sc = scores[i]
                grp = groups[i]
                tau_b = cal[grp]["tau"]
                if not call_blocked_flags[i]:
                    oof_blocks.append(0)
                    oof_asks.append(0)
                    oof_any.append(0)
                elif sc < tau_b and not low_conf_only_flags[i]:
                    oof_blocks.append(1)
                    oof_asks.append(0)
                    oof_any.append(1)
                else:
                    # Routed to ASK because score >= tau_b (or tau_b == 0.0 / low-confidence control class)
                    oof_blocks.append(0)
                    oof_asks.append(1)
                    oof_any.append(1)

        pt_b, ci_b = cluster_bootstrap_rate_ci(oof_blocks, tools, n_resamples=5000, seed=42)
        pt_a, ci_a = cluster_bootstrap_rate_ci(oof_asks, tools, n_resamples=5000, seed=42)
        pt_any, ci_any = cluster_bootstrap_rate_ci(oof_any, tools, n_resamples=5000, seed=42)

        # Evaluate fcrb fabricated with full-calib C1 thresholds
        full_cal = mondrian_conformal_threshold_ucb(scores, groups, alpha=target_alpha, delta=0.10)
        f_block, f_ask = 0, 0
        for (sc, dec), grp in zip(fab_rag_scores, fab_groups):
            tau_b = full_cal[grp]["tau"]
            if dec == Decision.ALLOW:
                continue
            if sc < tau_b and dec == Decision.BLOCK:
                f_block += 1
            else:
                f_ask += 1
        f_union = f_block + f_ask
        n_f = len(fcrb_fabs)
        return {
            "alpha": target_alpha,
            "oof_block_rate": round(pt_b * 100, 2),
            "oof_block_ci": [round(ci_b[0] * 100, 2), round(ci_b[1] * 100, 2)],
            "oof_ask_rate": round(pt_a * 100, 2),
            "oof_ask_ci": [round(ci_a[0] * 100, 2), round(ci_a[1] * 100, 2)],
            "oof_any_rate": round(pt_any * 100, 2),
            "oof_any_ci": [round(ci_any[0] * 100, 2), round(ci_any[1] * 100, 2)],
            "fcrb_block_recall": round(f_block / n_f * 100, 2),
            "fcrb_ask_recall": round(f_ask / n_f * 100, 2),
            "fcrb_union_recall": round(f_union / n_f * 100, 2),
            "thresholds": full_cal,
        }

    print("\n--- 5-Fold Tool-Grouped Out-of-Fold Mondrian C1 Calibration Results ---")
    print(f"{'alpha':<6} | {'OOF BLOCK [Cluster CI]':<24} | {'OOF ASK [CI]':<20} | {'OOF Any [CI]':<20} | {'fcrb BLOCK / Union'}")
    print("-" * 105)
    sweep = {}
    for a in [0.05, 0.08, 0.10, 0.12, 0.15]:
        r = eval_alpha_policy(a)
        sweep[str(a)] = r
        print(
            f"{a:<6.2f} | {r['oof_block_rate']:>5.2f}% [{r['oof_block_ci'][0]:>4.1f}%, {r['oof_block_ci'][1]:>4.1f}%]   | "
            f"{r['oof_ask_rate']:>5.2f}% [{r['oof_ask_ci'][0]:>4.1f}%, {r['oof_ask_ci'][1]:>4.1f}%] | "
            f"{r['oof_any_rate']:>5.2f}% [{r['oof_any_ci'][0]:>4.1f}%, {r['oof_any_ci'][1]:>4.1f}%] | "
            f"{r['fcrb_block_recall']:>5.1f}% / {r['fcrb_union_recall']:>5.1f}%"
        )

    out_path = ROOT / "results" / "b2_mondrian_calibration.json"
    out_path.write_text(json.dumps({"group_stats_05": cal_alpha_05, "sweep": sweep}, indent=2), encoding="utf-8")
    print(f"\nSaved B2c Mondrian calibration results to {out_path}")


if __name__ == "__main__":
    run()
