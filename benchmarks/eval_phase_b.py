"""
Evaluation harness for Phase B (Cutting False Blocks).

Evaluates on calibration/dev data strictly:
  - Clean MCP-Atlas traffic (1,212 calibration calls, 53 tools)
  - AgentHallu (31 errors, 93 clean calls)
  - Fabricated value errors (DeRef fcrb fabricated subset, n=49)

Reports:
  - BLOCK-tier false-block rate (target: <= 5%)
  - ASK rate on clean calls
  - Fabricated value recall (floor: >= 60%)
  - AgentHallu detection recall (floor: >= 60%)
  All with Wilson and bootstrap 95% CIs.
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path
from typing import Any, Dict, List, Tuple
import numpy as np

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
sys.path.insert(0, str(ROOT / "benchmarks"))

from benchmarks.stats import (
    wilson_interval,
    bootstrap_rate_ci,
    format_rate_with_ci,
)
from server.models.schemas import ArgumentGroundingRequest, Decision
from server.tools.verify_arguments import run_argument_grounding, is_low_confidence_arg
from server.pipeline.provenance import check_call, evidence_from_agenthallu_premise


def load_calibration_mcpatlas() -> List[Dict[str, Any]]:
    split_file = ROOT / "benchmarks" / "data" / "mcpatlas" / "mcpatlas_tool_split.json"
    data = json.loads(split_file.read_text(encoding="utf-8"))
    return data["dev_cases"]


def load_agenthallu() -> Tuple[List[Dict[str, Any]], List[Dict[str, Any]]]:
    ah_file = ROOT / "benchmarks" / "data" / "agenthallu" / "cases_eval.json"
    cases = json.loads(ah_file.read_text(encoding="utf-8"))["cases"]
    errors = [c for c in cases if not bool(c["label"])]
    cleans = [c for c in cases if bool(c["label"])]
    return errors, cleans


def load_fcrb_fabricated() -> List[Dict[str, Any]]:
    audit_file = ROOT / "review" / "audit3" / "error_taxonomy_audit.json"
    records = json.loads(audit_file.read_text(encoding="utf-8"))
    fcrb_fab_ids = {r["id"] for r in records if r["category"] == "fabricated_value"}
    
    cases_file = ROOT / "benchmarks" / "data" / "deref" / "cases_fcrb.json"
    all_fcrb = json.loads(cases_file.read_text(encoding="utf-8"))["cases"]
    return [c for c in all_fcrb if c["id"] in fcrb_fab_ids]


def eval_mcpatlas_clean(
    cases: List[Dict[str, Any]],
    decision_mode: str = "three_tier",
    tau_block: float = 0.50,
    tau_ask: float = 0.88,
    tau_binary: float = 0.88,
) -> Dict[str, Any]:
    n = len(cases)
    blocks = 0
    asks = 0
    allows = 0

    for c in cases:
        args = c["call"].get("arguments", {})
        if not args:
            allows += 1
            continue

        ev = [("user", c["user"])] + [("tool", t["text"]) for t in (c.get("tool_results") or [])]
        cv = check_call(args, ev, config="P3")
        if not cv.blocked:
            allows += 1
        else:
            ungrounded = [lv for lv in cv.leaves if lv.verdict == "UNGROUNDED"]
            if decision_mode == "binary":
                blocks += 1
            else:
                if all(is_low_confidence_arg(lv.path.split(".")[0].split("[")[0]) for lv in ungrounded):
                    asks += 1
                else:
                    blocks += 1

    return {
        "n": n,
        "blocks": blocks,
        "asks": asks,
        "allows": allows,
        "block_rate": blocks / n if n else 0.0,
        "block_ci": wilson_interval(blocks, n),
        "ask_rate": asks / n if n else 0.0,
        "ask_ci": wilson_interval(asks, n),
        "allow_rate": allows / n if n else 0.0,
    }


def eval_fabricated_recall(
    cases: List[Dict[str, Any]],
    decision_mode: str = "three_tier",
    tau_block: float = 0.50,
    tau_ask: float = 0.88,
    tau_binary: float = 0.88,
) -> Dict[str, Any]:
    n = len(cases)
    caught_any = 0
    caught_hard_block = 0

    for c in cases:
        call = c["call"]
        args = call.get("arguments", {})
        ctx = c.get("premise", "")
        req = ArgumentGroundingRequest(
            arguments=args,
            context=ctx,
            tau=tau_binary,
            tau_block=tau_block,
            tau_ask=tau_ask,
            decision_mode=decision_mode,
        )
        res = run_argument_grounding(req)
        if res.decision in (Decision.BLOCK, Decision.ASK):
            caught_any += 1
        if res.decision == Decision.BLOCK:
            caught_hard_block += 1

    return {
        "n": n,
        "caught_any": caught_any,
        "caught_hard_block": caught_hard_block,
        "recall_any": caught_any / n if n else 0.0,
        "recall_any_ci": wilson_interval(caught_any, n),
        "recall_hard_block": caught_hard_block / n if n else 0.0,
        "recall_hard_ci": wilson_interval(caught_hard_block, n),
    }


def eval_agenthallu_detection(
    errors: List[Dict[str, Any]],
    cleans: List[Dict[str, Any]],
    decision_mode: str = "three_tier",
    tau_block: float = 0.50,
    tau_ask: float = 0.88,
    tau_binary: float = 0.88,
) -> Dict[str, Any]:
    n_err = len(errors)
    n_clean = len(cleans)
    
    err_caught = 0
    clean_blocked = 0
    clean_asked = 0

    for c in errors:
        ev = evidence_from_agenthallu_premise(c["premise"])
        args = c["call"].get("arguments", {})
        cv = check_call(args, ev, config="P3")
        if cv.blocked:
            err_caught += 1

    for c in cleans:
        ev = evidence_from_agenthallu_premise(c["premise"])
        args = c["call"].get("arguments", {})
        cv = check_call(args, ev, config="P3")
        if cv.blocked:
            ungrounded = [lv for lv in cv.leaves if lv.verdict == "UNGROUNDED"]
            if decision_mode == "binary":
                clean_blocked += 1
            else:
                if all(is_low_confidence_arg(lv.path.split(".")[0].split("[")[0]) for lv in ungrounded):
                    clean_asked += 1
                else:
                    clean_blocked += 1

    return {
        "n_err": n_err,
        "err_caught": err_caught,
        "detection": err_caught / n_err if n_err else 0.0,
        "detection_ci": wilson_interval(err_caught, n_err),
        "n_clean": n_clean,
        "clean_blocked": clean_blocked,
        "fbr_block": clean_blocked / n_clean if n_clean else 0.0,
        "fbr_block_ci": wilson_interval(clean_blocked, n_clean),
        "clean_asked": clean_asked,
        "fbr_ask": clean_asked / n_clean if n_clean else 0.0,
        "fbr_ask_ci": wilson_interval(clean_asked, n_clean),
    }


def main():
    print("=" * 70)
    print("Phase B Evaluation: Three-Outcome Decision (ALLOW | ASK | BLOCK)")
    print("=" * 70)

    mcpatlas_calib = load_calibration_mcpatlas()
    ah_errors, ah_cleans = load_agenthallu()
    fcrb_fab = load_fcrb_fabricated()

    print(f"Loaded Calibration Clean MCP-Atlas: {len(mcpatlas_calib)} cases (53 tools)")
    print(f"Loaded AgentHallu: {len(ah_errors)} errors, {len(ah_cleans)} clean calls")
    print(f"Loaded DeRef fcrb Fabricated Values: {len(fcrb_fab)} cases")

    # 1. Baseline: Binary P3 at tau=0.88
    print("\n--- 1. Baseline (Binary Mode: ALLOW vs BLOCK at tau=0.88) ---")
    base_mcp = eval_mcpatlas_clean(mcpatlas_calib, decision_mode="binary", tau_binary=0.88)
    base_fab = eval_fabricated_recall(fcrb_fab, decision_mode="binary", tau_binary=0.88)
    base_ah = eval_agenthallu_detection(ah_errors, ah_cleans, decision_mode="binary", tau_binary=0.88)

    print(f"MCP-Atlas Calibration Clean False Blocks: {format_rate_with_ci(base_mcp['blocks'], base_mcp['n'])}")
    print(f"Fabricated Value Recall (fcrb):          {format_rate_with_ci(base_fab['caught_any'], base_fab['n'])}")
    print(f"AgentHallu Detection:                    {format_rate_with_ci(base_ah['err_caught'], base_ah['n_err'])} [EXPLORATORY]")
    print(f"AgentHallu Clean False Blocks:           {format_rate_with_ci(base_ah['clean_blocked'], base_ah['n_clean'])}")

    # 2. Task B1: Three-tier mode (tau_block=0.50, tau_ask=0.88)
    print("\n--- 2. Task B1 (Three-Tier: ALLOW | ASK | BLOCK; tau_block=0.50, tau_ask=0.88) ---")
    b1_mcp = eval_mcpatlas_clean(mcpatlas_calib, decision_mode="three_tier", tau_block=0.50, tau_ask=0.88)
    b1_fab = eval_fabricated_recall(fcrb_fab, decision_mode="three_tier", tau_block=0.50, tau_ask=0.88)
    b1_ah = eval_agenthallu_detection(ah_errors, ah_cleans, decision_mode="three_tier", tau_block=0.50, tau_ask=0.88)

    print(f"MCP-Atlas Hard False Blocks (BLOCK tier): {format_rate_with_ci(b1_mcp['blocks'], b1_mcp['n'])}")
    print(f"MCP-Atlas ASK Rate (Clean Traffic):       {format_rate_with_ci(b1_mcp['asks'], b1_mcp['n'])}")
    print(f"MCP-Atlas ALLOW Rate (Clean Traffic):     {format_rate_with_ci(b1_mcp['allows'], b1_mcp['n'])}")
    print(f"Fabricated Value Recall (BLOCK tier):    {format_rate_with_ci(b1_fab['caught_hard_block'], b1_fab['n'])}")
    print(f"Fabricated Value Recall (BLOCK + ASK):   {format_rate_with_ci(b1_fab['caught_any'], b1_fab['n'])}")
    print(f"AgentHallu Detection (BLOCK + ASK):      {format_rate_with_ci(b1_ah['err_caught'], b1_ah['n_err'])} [EXPLORATORY]")
    print(f"AgentHallu Clean Hard Blocks:            {format_rate_with_ci(b1_ah['clean_blocked'], b1_ah['n_clean'])}")
    print(f"AgentHallu Clean ASK Rate:               {format_rate_with_ci(b1_ah['clean_asked'], b1_ah['n_clean'])}")

    # Summary JSON
    summary = {
        "baseline_binary": {
            "mcpatlas_fbr": base_mcp,
            "fabricated_recall": base_fab,
            "agenthallu": base_ah,
        },
        "b1_three_tier": {
            "mcpatlas": b1_mcp,
            "fabricated_recall": b1_fab,
            "agenthallu": b1_ah,
        },
    }
    out_file = ROOT / "benchmarks" / "phase_b_calibration_results.json"
    out_file.write_text(json.dumps(summary, indent=2), encoding="utf-8")
    print(f"\nWrote calibration results to {out_file.relative_to(ROOT)}")


if __name__ == "__main__":
    main()
