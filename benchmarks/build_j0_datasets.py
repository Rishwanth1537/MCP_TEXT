"""
Build J0 datasets:
1. Tool-grouped dev & frozen held-out split for DeRef fcrb (benchmarks/data/deref/fcrb_tool_split.json).
2. >=150-case Tool-Selection dataset from real tool lists (BFCL, MCP-Atlas dev, AgentHallu):
   benchmarks/data/tool_selection_eval.json.
3. Blinded 50-item human audit file from fcrb/BFCL disagreements and AgentHallu cases:
   review/audit_judge_50_blind.csv (+ private key review/audit_judge_50_key_DO_NOT_OPEN.json).
"""
from __future__ import annotations

import csv
import hashlib
import json
import random
import sys
from collections import defaultdict
from pathlib import Path
from typing import Any, Dict, List

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

SEED = 42


def sha256_bytes(data: bytes) -> str:
    return hashlib.sha256(data).hexdigest()


def build_fcrb_tool_split() -> Dict[str, Any]:
    import numpy as np  # noqa: PLC0415

    src = ROOT / "benchmarks" / "data" / "deref" / "cases_fcrb.json"
    schema_src = ROOT / "benchmarks" / "data" / "deref" / "cases_fcrb_schema.json"
    cases = json.loads(src.read_text(encoding="utf-8"))["cases"]
    schema_map = {c["id"]: c["premise"] for c in json.loads(schema_src.read_text(encoding="utf-8"))["cases"]}

    # Pin all tools appearing in dev200 (split == "dev", seed=0, limit=200) into dev
    # so dev200 has ZERO id, query_id, or tool_name overlap with frozen_heldout.
    dev_pool = [c for c in cases if c.get("split") == "dev"]
    idx200 = np.random.default_rng(0).permutation(len(dev_pool))[:200]
    dev200 = [dev_pool[i] for i in sorted(idx200)]
    pinned_dev_tools = {(c.get("call") or (c.get("calls") or [{}])[0]).get("name") or "unknown" for c in dev200}

    by_tool: Dict[str, List[Dict[str, Any]]] = defaultdict(list)
    for c in cases:
        call = c.get("call") or (c.get("calls") or [{}])[0]
        tool_name = call.get("name") or "unknown"
        item = dict(c)
        item["tool_name"] = tool_name
        item["premise_with_schema"] = schema_map.get(c["id"], c["premise"])
        by_tool[tool_name].append(item)

    dev_tools = sorted(t for t in by_tool if t in pinned_dev_tools)
    heldout_tools = sorted(t for t in by_tool if t not in pinned_dev_tools)

    dev_cases = [c for t in dev_tools for c in by_tool[t]]
    heldout_cases = [c for t in heldout_tools for c in by_tool[t]]

    # Hard assertion: zero overlap with dev200
    assert not ({c["id"] for c in dev200} & {c["id"] for c in heldout_cases})
    assert not ({c["query_id"] for c in dev200} & {c["query_id"] for c in heldout_cases})
    assert not (pinned_dev_tools & set(heldout_tools))

    heldout_bytes = json.dumps(heldout_cases, sort_keys=True).encode("utf-8")
    heldout_hash = sha256_bytes(heldout_bytes)

    out = {
        "seed": SEED,
        "source": str(src.relative_to(ROOT)),
        "note": "All 89 tools in dev200 are pinned to dev; frozen_heldout has 0 id/query_id/tool overlap with dev200.",
        "n_total_cases": len(cases),
        "n_total_tools": len(by_tool),
        "dev": {"n_tools": len(dev_tools), "n_cases": len(dev_cases), "tools": dev_tools},
        "frozen_heldout": {
            "n_tools": len(heldout_tools),
            "n_cases": len(heldout_cases),
            "tools": heldout_tools,
            "sha256": heldout_hash,
        },
        "dev_cases": dev_cases,
        "heldout_cases": heldout_cases,
    }
    out_path = ROOT / "benchmarks" / "data" / "deref" / "fcrb_tool_split.json"
    out_path.write_text(json.dumps(out, indent=1), encoding="utf-8")
    return {
        "path": str(out_path.relative_to(ROOT)),
        "dev_tools": len(dev_tools),
        "dev_cases": len(dev_cases),
        "heldout_tools": len(heldout_tools),
        "heldout_cases": len(heldout_cases),
        "heldout_sha256": heldout_hash,
    }


def build_tool_selection_set() -> Dict[str, Any]:
    rng = random.Random(SEED)
    out_cases: List[Dict[str, Any]] = []

    # 1. BFCL multi-schema cases (from consensus_cases.json)
    bfcl_path = ROOT / "benchmarks" / "data" / "bfcl" / "consensus_cases.json"
    bfcl = json.loads(bfcl_path.read_text(encoding="utf-8"))
    by_tid: Dict[str, Dict[str, Any]] = {}
    for r in bfcl["records"]:
        if len(r.get("function_schemas", [])) > 1 and r.get("calls"):
            tid = r["test_id"]
            if tid not in by_tid:
                by_tid[tid] = r

    tids = sorted(by_tid.keys())
    rng.shuffle(tids)
    bfcl_right = tids[:45]
    bfcl_wrong = tids[45:90]

    for tid in bfcl_right:
        r = by_tid[tid]
        call = r["calls"][0]
        tool_name = call.get("function")
        schemas = r["function_schemas"]
        out_cases.append({
            "id": f"tsel:bfcl:right:{tid}",
            "source_dataset": "bfcl_consensus",
            "split": "dev",
            "kind": "right_tool",
            "label": True,
            "user_request": r.get("request", ""),
            "context": r.get("request", ""),
            "chosen_tool": tool_name,
            "gold_tool": tool_name,
            "arguments": call.get("arguments", {}),
            "available_tools": schemas,
            "verification_method": "bfcl_gold_function_match_within_real_function_schemas",
        })

    for tid in bfcl_wrong:
        r = by_tid[tid]
        call = r["calls"][0]
        gold_name = call.get("function")
        schemas = r["function_schemas"]
        alt_names = [s["name"] for s in schemas if s["name"] != gold_name]
        wrong_name = rng.choice(alt_names)
        out_cases.append({
            "id": f"tsel:bfcl:wrong:{tid}",
            "source_dataset": "bfcl_consensus",
            "split": "dev",
            "kind": "wrong_tool",
            "label": False,
            "user_request": r.get("request", ""),
            "context": r.get("request", ""),
            "chosen_tool": wrong_name,
            "gold_tool": gold_name,
            "arguments": call.get("arguments", {}),
            "available_tools": schemas,
            "verification_method": "bfcl_gold_tool_mismatch_within_real_function_schemas",
        })

    # 2. MCP-Atlas dev split (using real ENABLED_TOOLS from MCP-Atlas.parquet; never touch heldout tools)
    import pandas as pd  # noqa: PLC0415

    mcp_split = json.loads((ROOT / "benchmarks" / "data" / "mcpatlas" / "mcpatlas_tool_split.json").read_text(encoding="utf-8"))
    dev_mcp_cases = mcp_split["dev_cases"]
    df = pd.read_parquet(ROOT / "benchmarks" / "data" / "mcpatlas" / "raw" / "MCP-Atlas.parquet")
    task_tools = {}
    for _, row in df.iterrows():
        try:
            task_tools[str(row["TASK"])] = json.loads(row["ENABLED_TOOLS"])
        except Exception:
            task_tools[str(row["TASK"])] = []

    mcp_candidates = []
    seen_tasks = set()
    for c in dev_mcp_cases:
        task_id = str(c["task"])
        tname = c["call"]["name"]
        enabled = task_tools.get(task_id, [])
        if len(enabled) >= 2 and tname in enabled and task_id not in seen_tasks:
            seen_tasks.add(task_id)
            mcp_candidates.append((c, enabled))

    rng.shuffle(mcp_candidates)
    mcp_right = mcp_candidates[:40]
    mcp_wrong = mcp_candidates[40:80]

    for c, enabled in mcp_right:
        tname = c["call"]["name"]
        out_cases.append({
            "id": f"tsel:mcpatlas:right:{c['id']}",
            "source_dataset": "mcpatlas_dev",
            "split": "dev",
            "kind": "right_tool",
            "label": True,
            "user_request": c["user"],
            "context": c["user"],
            "chosen_tool": tname,
            "gold_tool": tname,
            "arguments": c["call"].get("arguments", {}),
            "available_tools": [{"name": t} for t in enabled],
            "verification_method": "mcpatlas_reference_trajectory_tool_within_enabled_tools",
        })

    for c, enabled in mcp_wrong:
        gold_name = c["call"]["name"]
        same_prefix = [t for t in enabled if t != gold_name and t.split("_")[0] == gold_name.split("_")[0]]
        pool = same_prefix if same_prefix else [t for t in enabled if t != gold_name]
        wrong_name = rng.choice(pool)
        out_cases.append({
            "id": f"tsel:mcpatlas:wrong:{c['id']}",
            "source_dataset": "mcpatlas_dev",
            "split": "dev",
            "kind": "wrong_tool",
            "label": False,
            "user_request": c["user"],
            "context": c["user"],
            "chosen_tool": wrong_name,
            "gold_tool": gold_name,
            "arguments": c["call"].get("arguments", {}),
            "available_tools": [{"name": t} for t in enabled],
            "verification_method": "mcpatlas_reference_trajectory_mismatch_within_enabled_tools",
        })

    # 3. AgentHallu clean trajectories (right-tool cases)
    ah_cases = json.loads((ROOT / "benchmarks" / "data" / "agenthallu" / "cases.json").read_text(encoding="utf-8"))["cases"]
    ah_clean = [c for c in ah_cases if c.get("label") is True]
    rng.shuffle(ah_clean)
    all_ah_tools = sorted({c["call"]["name"] for c in ah_cases})
    for c in ah_clean[:10]:
        tname = c["call"]["name"]
        distractors = [t for t in all_ah_tools if t != tname]
        tool_list = sorted([tname] + rng.sample(distractors, min(4, len(distractors))))
        out_cases.append({
            "id": f"tsel:agenthallu:right:{c['id']}",
            "source_dataset": "agenthallu",
            "split": "eval",
            "kind": "right_tool",
            "label": True,
            "user_request": c["premise"][:1500],
            "context": c["premise"],
            "chosen_tool": tname,
            "gold_tool": tname,
            "arguments": c["call"].get("arguments", {}),
            "available_tools": [{"name": t} for t in tool_list],
            "verification_method": "agenthallu_clean_gold_trajectory_tool",
        })

    # 4. Natural BFCL wrong-tool selections from consensus_cases.json['malformed']
    seen_nat = set()
    n_natural_wrong = 0
    for m in bfcl.get("malformed", []):
        if m.get("reason") != "function_name_or_count_mismatch":
            continue
        pf = m.get("pred_functions") or []
        gf = m.get("gold_functions") or []
        tid = m.get("test_id")
        if len(pf) == 1 and len(gf) == 1 and pf[0] != gf[0] and tid in by_tid and tid not in seen_nat:
            r = by_tid[tid]
            snames = {s["name"] for s in r.get("function_schemas", [])}
            if pf[0] in snames and gf[0] in snames:
                seen_nat.add(tid)
                n_natural_wrong += 1
                out_cases.append({
                    "id": f"tsel:bfcl_natural:wrong:{tid}:{m.get('model')}",
                    "source_dataset": "bfcl_consensus_natural",
                    "split": "dev",
                    "kind": "wrong_tool_natural",
                    "is_natural": True,
                    "label": False,
                    "user_request": r.get("request", ""),
                    "context": r.get("request", ""),
                    "chosen_tool": pf[0],
                    "gold_tool": gf[0],
                    "arguments": {},
                    "available_tools": r["function_schemas"],
                    "verification_method": f"natural_model_wrong_tool_selection_by_{m.get('model')}_vs_bfcl_gold",
                })

    out_path = ROOT / "benchmarks" / "data" / "tool_selection_eval.json"
    out_obj = {
        "seed": SEED,
        "status": "EXPLORATORY",
        "n_total": len(out_cases),
        "n_right_tool": sum(1 for x in out_cases if x["label"]),
        "n_wrong_tool": sum(1 for x in out_cases if not x["label"]),
        "n_wrong_tool_natural": n_natural_wrong,
        "n_wrong_tool_synthetic": sum(1 for x in out_cases if not x["label"]) - n_natural_wrong,
        "cases": out_cases,
    }
    out_path.write_text(json.dumps(out_obj, indent=1), encoding="utf-8")
    return {
        "path": str(out_path.relative_to(ROOT)),
        "n_total": out_obj["n_total"],
        "n_right": out_obj["n_right_tool"],
        "n_wrong": out_obj["n_wrong_tool"],
        "n_wrong_natural": n_natural_wrong,
    }


def build_disjoint_few_shot_pool() -> Dict[str, Any]:
    """Build 24-item few-shot pool from ArgGround-Bench (Glaive v2), 100% disjoint from fcrb/AgentHallu/MCP-Atlas/BFCL."""
    src = ROOT / "benchmarks" / "argground_bench.json"
    cases = json.loads(src.read_text(encoding="utf-8"))["cases"]
    rng = random.Random(SEED)
    pos = [c for c in cases if c.get("label") is True]
    neg = [c for c in cases if c.get("label") is False]
    rng.shuffle(pos)
    rng.shuffle(neg)
    pool = []
    for c in pos[:12] + neg[:12]:
        pool.append({
            "id": f"fewshot:{c['id']}",
            "source_dataset": "argground_bench_glaive_v2",
            "premise": c["premise"],
            "call": c["call"],
            "supported": bool(c["label"]),
            "unsupported_arguments": [k for k in (c["call"].get("arguments") or {}) if not c["label"]][:1],
        })
    rng.shuffle(pool)
    out_path = ROOT / "benchmarks" / "data" / "few_shot_pool.json"
    out_path.write_text(json.dumps({"seed": SEED, "n": len(pool), "examples": pool}, indent=1), encoding="utf-8")
    return {"path": str(out_path.relative_to(ROOT)), "n": len(pool)}


def build_blinded_50_audit() -> Dict[str, Any]:
    rng = random.Random(SEED)
    from server.pipeline.arg_ground import call_is_grounded  # noqa: PLC0415

    # 30 fcrb/BFCL disagreements (Sonnet vs checker on fcrb + BFCL raw vs cleaned)
    fcrb_cases = {c["id"]: c for c in json.loads((ROOT / "benchmarks" / "data" / "deref" / "cases_fcrb_schema.json").read_text(encoding="utf-8"))["cases"]}
    sonnet_rows = json.loads((ROOT / "benchmarks" / "llm_judge_args_fcrb_agent_sonnet_dev_v0.json").read_text(encoding="utf-8"))["rows"]

    disagree_fcrb = []
    for r in sonnet_rows:
        cid = r["id"]
        c = fcrb_cases.get(cid)
        if not c:
            continue
        call = c.get("call") or (c.get("calls") or [{}])[0]
        chk_ok = call_is_grounded(call.get("arguments", {}), c["premise"])
        sonnet_ok = bool(r.get("supported")) if r.get("parsed") else None
        if sonnet_ok is not None and chk_ok != sonnet_ok:
            disagree_fcrb.append((cid, c, call, "fcrb_checker_vs_sonnet_disagreement"))

    rng.shuffle(disagree_fcrb)
    selected_fcrb = disagree_fcrb[:20]

    # 10 BFCL raw vs cleaned disagreements
    bfcl_cleaned = json.loads((ROOT / "benchmarks" / "data" / "bfcl" / "cases_bfcl_cleaned.json").read_text(encoding="utf-8"))["cases"]
    bfcl_dis = [c for c in bfcl_cleaned if c.get("raw_label") != c.get("label") or c.get("is_format_only")]
    if not bfcl_dis:
        bfcl_dis = bfcl_cleaned
    rng.shuffle(bfcl_dis)
    selected_bfcl = [(c["id"], c, c.get("call") or {}, "bfcl_raw_vs_cleaned_disagreement") for c in bfcl_dis[:10]]

    # 20 AgentHallu cases
    ah_eval = json.loads((ROOT / "benchmarks" / "data" / "agenthallu" / "cases_eval.json").read_text(encoding="utf-8"))["cases"]
    rng.shuffle(ah_eval)
    selected_ah = [(c["id"], c, c.get("call") or {}, "agenthallu_eval") for c in ah_eval[:20]]

    combined = selected_fcrb + selected_bfcl + selected_ah
    rng.shuffle(combined)

    blind_csv = ROOT / "review" / "audit_judge_50_blind.csv"
    key_json = ROOT / "review" / "audit_judge_50_key_DO_NOT_OPEN.json"

    key_rows = []
    with blind_csv.open("w", newline="", encoding="utf-8") as f:
        w = csv.writer(f)
        w.writerow([
            "audit_id",
            "context",
            "tool_name",
            "arguments_json",
            "human_verdict",
            "human_error_type",
            "human_notes",
        ])
        for idx, (orig_id, c, call, bucket) in enumerate(combined, 1):
            aid = f"J_AUD_{idx:03d}"
            w.writerow([
                aid,
                c.get("premise", "")[:4000],
                call.get("name", ""),
                json.dumps(call.get("arguments", {}), ensure_ascii=False),
                "",
                "",
                "",
            ])
            key_rows.append({
                "audit_id": aid,
                "orig_id": orig_id,
                "bucket": bucket,
                "ai_annotated_label": c.get("label"),
            })

    key_json.write_text(json.dumps(key_rows, indent=1), encoding="utf-8")
    return {"blind_csv": str(blind_csv.relative_to(ROOT)), "n_items": len(key_rows)}


def main():
    fcrb_info = build_fcrb_tool_split()
    tsel_info = build_tool_selection_set()
    audit_info = build_blinded_50_audit()
    fs_info = build_disjoint_few_shot_pool()
    print(json.dumps({
        "fcrb_tool_split": fcrb_info,
        "tool_selection": tsel_info,
        "audit_50": audit_info,
        "few_shot_pool": fs_info,
    }, indent=2))


if __name__ == "__main__":
    main()
