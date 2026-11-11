"""
Evaluation harness for Phase B2 step-by-step ablations:
- B2a: Schema validation (enum, flag, control, buffer) -> ALLOW inside bounds/enums
- B2b: Host current date/time evidence & datetime improvements
- B2c: Call-level Mondrian conformal calibration with C1 upper bound
"""

import json
import sys
from pathlib import Path
from collections import defaultdict
import numpy as np

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))

from server.pipeline.provenance import check_call, evidence_from_agenthallu_premise
from server.tools.verify_arguments import is_low_confidence_arg, run_argument_grounding
from server.models.schemas import ArgumentGroundingRequest, Decision
from benchmarks.stats import cluster_bootstrap_rate_ci, wilson_interval

SPLIT_FILE = ROOT / "benchmarks" / "data" / "mcpatlas" / "mcpatlas_tool_split.json"


def evaluate_b2_step(step_name: str, supply_host_datetime: bool = False):
    print(f"\n=======================================================")
    print(f"EVALUATING PHASE B2 STEP: {step_name} (host_dt={supply_host_datetime})")
    print(f"=======================================================")

    split_data = json.loads(SPLIT_FILE.read_text(encoding="utf-8"))
    calib_cases = split_data["dev_cases"]
    n_calib = len(calib_cases)

    dev_blocks = []
    dev_asks = []
    dev_any = []
    dev_tool_ids = []
    per_call_records = []

    for c in calib_cases:
        tname = c["call"]["name"]
        args = c["call"].get("arguments", {})
        dev_tool_ids.append(tname)

        if not args:
            dev_blocks.append(0)
            dev_asks.append(0)
            dev_any.append(0)
            per_call_records.append({"id": c["id"], "tool": tname, "decision": "ALLOW", "ungrounded": []})
            continue

        ev = [("user", c["user"])] + [("tool", t["text"]) for t in (c.get("tool_results") or [])]
        if supply_host_datetime:
            # MCP-Atlas trajectories were recorded in July/August 2025; supply host anchor clock cleanly
            # Also extract any task date if metadata exists, otherwise standard host anchor
            ev.insert(0, ("host", "Current date and time: 2025-07-15T12:00:00Z"))

        cv = check_call(args, ev, config="P3", tool_name=tname)
        if not cv.blocked:
            dev_blocks.append(0)
            dev_asks.append(0)
            dev_any.append(0)
            per_call_records.append({"id": c["id"], "tool": tname, "decision": "ALLOW", "ungrounded": []})
        else:
            ungrounded = [lv for lv in cv.leaves if lv.verdict == "UNGROUNDED"]
            if all(is_low_confidence_arg(lv.path.split(".")[0].split("[")[0]) for lv in ungrounded):
                dev_blocks.append(0)
                dev_asks.append(1)
                dev_any.append(1)
                per_call_records.append({
                    "id": c["id"], "tool": tname, "decision": "ASK",
                    "ungrounded": [(lv.path, lv.value, lv.reason) for lv in ungrounded]
                })
            else:
                dev_blocks.append(1)
                dev_asks.append(0)
                dev_any.append(1)
                per_call_records.append({
                    "id": c["id"], "tool": tname, "decision": "BLOCK",
                    "ungrounded": [(lv.path, lv.value, lv.reason) for lv in ungrounded]
                })

    pt_block, ci_tool_block = cluster_bootstrap_rate_ci(dev_blocks, dev_tool_ids, n_resamples=5000, seed=42)
    pt_ask, ci_tool_ask = cluster_bootstrap_rate_ci(dev_asks, dev_tool_ids, n_resamples=5000, seed=42)
    pt_any, ci_tool_any = cluster_bootstrap_rate_ci(dev_any, dev_tool_ids, n_resamples=5000, seed=42)

    print(f"\n--- MCP-Atlas Calibration (n={n_calib}, 53 tools) ---")
    print(f"BLOCK Rate:          {pt_block*100:5.2f}% (Cluster CI: [{ci_tool_block[0]*100:4.1f}%, {ci_tool_block[1]*100:4.1f}%], count: {sum(dev_blocks)}/{n_calib})")
    print(f"ASK Rate:            {pt_ask*100:5.2f}% (Cluster CI: [{ci_tool_ask[0]*100:4.1f}%, {ci_tool_ask[1]*100:4.1f}%], count: {sum(dev_asks)}/{n_calib})")
    print(f"Any Intervention:    {pt_any*100:5.2f}% (Cluster CI: [{ci_tool_any[0]*100:4.1f}%, {ci_tool_any[1]*100:4.1f}%], count: {sum(dev_any)}/{n_calib})")

    # Evaluate fcrb fabricated (n=362)
    audit_records = json.loads((ROOT / "review" / "audit3" / "error_taxonomy_audit.json").read_text(encoding="utf-8"))
    fcrb_fab_ids = {r["id"] for r in audit_records if r["category"] == "fabricated_value"}
    fcrb_all = {c["id"]: c for c in json.loads((ROOT / "benchmarks" / "data" / "deref" / "cases_fcrb.json").read_text(encoding="utf-8"))["cases"]}
    fcrb_fabs = [c for c in fcrb_all.values() if c["id"] in fcrb_fab_ids]

    fab_block, fab_ask = 0, 0
    for c in fcrb_fabs:
        tname = c["call"].get("name")
        res = run_argument_grounding(ArgumentGroundingRequest(
            arguments=c["call"].get("arguments", {}),
            context=c.get("premise", ""),
            tool_name=tname,
            decision_mode="three_tier", tau_block=0.50, tau_ask=0.88
        ))
        if res.decision == Decision.BLOCK: fab_block += 1
        elif res.decision == Decision.ASK: fab_ask += 1
    fab_union = fab_block + fab_ask
    n_fab = len(fcrb_fabs)

    # Evaluate AgentHallu (31 err / 93 clean)
    ah_cases = json.loads((ROOT / "benchmarks" / "data" / "agenthallu" / "cases_eval.json").read_text(encoding="utf-8"))["cases"]
    ah_errs = [c for c in ah_cases if not bool(c["label"])]
    ah_cleans = [c for c in ah_cases if bool(c["label"])]

    ah_err_caught = 0
    for c in ah_errs:
        ev = evidence_from_agenthallu_premise(c["premise"])
        cv = check_call(c["call"].get("arguments", {}), ev, config="P3", tool_name=c["call"].get("name"))
        if cv.blocked: ah_err_caught += 1

    ah_clean_block, ah_clean_ask = 0, 0
    for c in ah_cleans:
        ev = evidence_from_agenthallu_premise(c["premise"])
        cv = check_call(c["call"].get("arguments", {}), ev, config="P3", tool_name=c["call"].get("name"))
        if cv.blocked:
            ungr = [lv for lv in cv.leaves if lv.verdict == "UNGROUNDED"]
            if all(is_low_confidence_arg(lv.path.split(".")[0].split("[")[0]) for lv in ungr):
                ah_clean_ask += 1
            else:
                ah_clean_block += 1

    print(f"\n--- Fabricated & Independent Evaluation ---")
    print(f"fcrb Fabricated (n={n_fab}): BLOCK={fab_block}/{n_fab} ({fab_block/n_fab*100:.1f}%), ASK={fab_ask}/{n_fab} ({fab_ask/n_fab*100:.1f}%), Union={fab_union}/{n_fab} ({fab_union/n_fab*100:.1f}%)")
    print(f"AgentHallu Errors (n={len(ah_errs)}): Detection={ah_err_caught}/{len(ah_errs)} ({ah_err_caught/len(ah_errs)*100:.1f}%)")
    print(f"AgentHallu Clean (n={len(ah_cleans)}): BLOCK={ah_clean_block}/{len(ah_cleans)} ({ah_clean_block/len(ah_cleans)*100:.1f}%), ASK={ah_clean_ask}/{len(ah_cleans)} ({ah_clean_ask/len(ah_cleans)*100:.1f}%)")

    summary = {
        "step": step_name,
        "mcpatlas_calib": {
            "n": n_calib,
            "block_count": int(sum(dev_blocks)),
            "block_rate": round(pt_block * 100, 2),
            "block_ci": [round(ci_tool_block[0] * 100, 2), round(ci_tool_block[1] * 100, 2)],
            "ask_count": int(sum(dev_asks)),
            "ask_rate": round(pt_ask * 100, 2),
            "ask_ci": [round(ci_tool_ask[0] * 100, 2), round(ci_tool_ask[1] * 100, 2)],
            "any_count": int(sum(dev_any)),
            "any_rate": round(pt_any * 100, 2),
            "any_ci": [round(ci_tool_any[0] * 100, 2), round(ci_tool_any[1] * 100, 2)],
        },
        "fcrb_fab": {
            "n": n_fab,
            "block_recall": round(fab_block / n_fab * 100, 2),
            "ask_recall": round(fab_ask / n_fab * 100, 2),
            "union_recall": round(fab_union / n_fab * 100, 2),
        },
        "agenthallu": {
            "err_n": len(ah_errs),
            "err_detection": round(ah_err_caught / len(ah_errs) * 100, 2),
            "clean_n": len(ah_cleans),
            "clean_block": ah_clean_block,
            "clean_ask": ah_clean_ask,
            "clean_block_rate": round(ah_clean_block / len(ah_cleans) * 100, 2),
        },
        "per_call_flags": [r for r in per_call_records if r["decision"] != "ALLOW"],
    }
    out_file = ROOT / "results" / f"ablation_{step_name.lower()}.json"
    out_file.write_text(json.dumps(summary, indent=2), encoding="utf-8")
    print(f"Saved results to {out_file}")


if __name__ == "__main__":
    step = sys.argv[1] if len(sys.argv) > 1 else "B2a"
    host_dt = len(sys.argv) > 2 and sys.argv[2].lower() == "host_dt"
    evaluate_b2_step(step, supply_host_datetime=host_dt)
