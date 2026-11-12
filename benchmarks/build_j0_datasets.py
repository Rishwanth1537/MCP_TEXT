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
    dev200_ids = {c["id"] for c in dev200}
    pinned_dev_tools = {(c.get("call") or (c.get("calls") or [{}])[0]).get("name") or "unknown" for c in dev200}

    by_tool: Dict[str, List[Dict[str, Any]]] = defaultdict(list)
    for c in cases:
        call = c.get("call") or (c.get("calls") or [{}])[0]
        tool_name = call.get("name") or "unknown"
        item = dict(c)
        item["tool_name"] = tool_name
        item["in_dev200"] = c["id"] in dev200_ids
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

    # Split dev_tools (89 tools, 856 cases) by tool into dev_train and dev_val
    rng = random.Random(SEED)
    shuffled_dev = list(dev_tools)
    rng.shuffle(shuffled_dev)
    dev_train_tools, dev_val_tools = [], []
    tr_n, va_n = 0, 0
    for t in shuffled_dev:
        cnt = len(by_tool[t])
        if tr_n <= va_n:
            dev_train_tools.append(t)
            tr_n += cnt
        else:
            dev_val_tools.append(t)
            va_n += cnt
    dev_train_tools.sort()
    dev_val_tools.sort()
    assert not (set(dev_train_tools) & set(dev_val_tools))
    assert not (set(dev_train_tools) & set(heldout_tools))
    assert not (set(dev_val_tools) & set(heldout_tools))

    dev_train_cases = [c for t in dev_train_tools for c in by_tool[t]]
    dev_val_cases = [c for t in dev_val_tools for c in by_tool[t]]

    heldout_fresh_272 = [c for c in heldout_cases if c.get("split") != "dev"]
    heldout_bytes = json.dumps(heldout_cases, sort_keys=True).encode("utf-8")
    heldout_hash = sha256_bytes(heldout_bytes)
    heldout_272_hash = sha256_bytes(json.dumps(heldout_fresh_272, sort_keys=True).encode("utf-8"))

    out = {
        "seed": SEED,
        "source": str(src.relative_to(ROOT)),
        "note": (
            "All 89 tools in dev200 are pinned to dev (split by tool into dev_train/dev_val); "
            "frozen_heldout (n=404) is judge-fresh, checker-partly-seen (132 cases in checker 630-case split==dev); "
            "heldout_fresh_272 (n=272) is headline for judge-vs-checker comparisons."
        ),
        "n_total_cases": len(cases),
        "n_total_tools": len(by_tool),
        "dev": {"n_tools": len(dev_tools), "n_cases": len(dev_cases), "tools": dev_tools},
        "dev_train": {"n_tools": len(dev_train_tools), "n_cases": len(dev_train_cases), "n_in_dev200": sum(c["in_dev200"] for c in dev_train_cases), "tools": dev_train_tools},
        "dev_val": {"n_tools": len(dev_val_tools), "n_cases": len(dev_val_cases), "n_in_dev200": sum(c["in_dev200"] for c in dev_val_cases), "tools": dev_val_tools},
        "frozen_heldout": {
            "n_tools": len(heldout_tools),
            "n_cases": len(heldout_cases),
            "tools": heldout_tools,
            "sha256": heldout_hash,
            "n_fresh_272": len(heldout_fresh_272),
            "fresh_272_sha256": heldout_272_hash,
        },
        "dev_cases": dev_cases,
        "dev_train_cases": dev_train_cases,
        "dev_val_cases": dev_val_cases,
        "heldout_cases": heldout_cases,
        "heldout_fresh_272_cases": heldout_fresh_272,
    }
    out_path = ROOT / "benchmarks" / "data" / "deref" / "fcrb_tool_split.json"
    out_path.write_text(json.dumps(out, indent=1), encoding="utf-8")
    return {
        "path": str(out_path.relative_to(ROOT)),
        "dev_tools": len(dev_tools),
        "dev_cases": len(dev_cases),
        "dev_train_tools": len(dev_train_tools),
        "dev_train_cases": len(dev_train_cases),
        "dev_val_tools": len(dev_val_tools),
        "dev_val_cases": len(dev_val_cases),
        "heldout_tools": len(heldout_tools),
        "heldout_cases": len(heldout_cases),
        "heldout_sha256": heldout_hash,
        "heldout_fresh_272": len(heldout_fresh_272),
        "heldout_272_sha256": heldout_272_hash,
    }


def _tool_text(schema: Dict[str, Any]) -> set:
    import re  # noqa: PLC0415
    parts = [schema.get("name", ""), schema.get("description", "")]
    props = (schema.get("parameters") or {}).get("properties") or {}
    for k, v in props.items():
        parts.append(str(k))
        if isinstance(v, dict):
            parts.append(str(v.get("description", "")))
    return set(re.findall(r"[a-z0-9]+", " ".join(parts).lower()))


def _most_similar_sibling(gold_name: str, schemas: List[Dict[str, Any]]) -> str:
    gold_schema = next((s for s in schemas if s["name"] == gold_name), {"name": gold_name})
    g_toks = _tool_text(gold_schema)
    best_name, best_sim = None, -1.0
    for s in schemas:
        if s["name"] == gold_name:
            continue
        s_toks = _tool_text(s)
        union = g_toks | s_toks
        sim = len(g_toks & s_toks) / len(union) if union else 0.0
        if sim > best_sim:
            best_sim = sim
            best_name = s["name"]
    return best_name or gold_name
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

    # Exclude parallel_multiple / live_parallel_multiple from synthetic wrong-tool substitutions
    # because the sibling tool often satisfies the other parallel sub-request (found in 30-case audit).
    tids = sorted(t for t in by_tid if "parallel" not in t)
    rng.shuffle(tids)
    bfcl_right = tids[:45]
    bfcl_wrong_easy = tids[45:70]
    bfcl_wrong_hard = tids[70:95]

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
            "difficulty": "right",
            "is_natural": True,
            "label": True,
            "user_request": r.get("request", ""),
            "context": r.get("request", ""),
            "chosen_tool": tool_name,
            "gold_tool": tool_name,
            "arguments": call.get("arguments", {}),
            "available_tools": schemas,
            "verification_method": "bfcl_gold_function_match_within_real_function_schemas",
        })

    for tid in bfcl_wrong_easy:
        r = by_tid[tid]
        call = r["calls"][0]
        gold_name = call.get("function")
        schemas = r["function_schemas"]
        alt_names = [s["name"] for s in schemas if s["name"] != gold_name]
        wrong_name = rng.choice(alt_names)
        out_cases.append({
            "id": f"tsel:bfcl:wrong_easy:{tid}",
            "source_dataset": "bfcl_consensus",
            "split": "dev",
            "kind": "wrong_tool",
            "difficulty": "easy",
            "is_natural": False,
            "label": False,
            "user_request": r.get("request", ""),
            "context": r.get("request", ""),
            "chosen_tool": wrong_name,
            "gold_tool": gold_name,
            "arguments": call.get("arguments", {}),
            "available_tools": schemas,
            "verification_method": "bfcl_gold_tool_mismatch_random_sibling_within_real_function_schemas",
        })

    for tid in bfcl_wrong_hard:
        r = by_tid[tid]
        call = r["calls"][0]
        gold_name = call.get("function")
        schemas = r["function_schemas"]
        wrong_name = _most_similar_sibling(gold_name, schemas)
        out_cases.append({
            "id": f"tsel:bfcl:wrong_hard:{tid}",
            "source_dataset": "bfcl_consensus",
            "split": "dev",
            "kind": "wrong_tool",
            "difficulty": "hard",
            "is_natural": False,
            "label": False,
            "user_request": r.get("request", ""),
            "context": r.get("request", ""),
            "chosen_tool": wrong_name,
            "gold_tool": gold_name,
            "arguments": call.get("arguments", {}),
            "available_tools": schemas,
            "verification_method": "bfcl_gold_tool_mismatch_most_similar_sibling_by_description",
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
    mcp_wrong_easy = mcp_candidates[40:60]
    mcp_wrong_hard = mcp_candidates[60:80]

    for c, enabled in mcp_right:
        tname = c["call"]["name"]
        out_cases.append({
            "id": f"tsel:mcpatlas:right:{c['id']}",
            "source_dataset": "mcpatlas_dev",
            "split": "dev",
            "kind": "right_tool",
            "difficulty": "right",
            "is_natural": True,
            "label": True,
            "user_request": c["user"],
            "context": c["user"],
            "chosen_tool": tname,
            "gold_tool": tname,
            "arguments": c["call"].get("arguments", {}),
            "available_tools": [{"name": t} for t in enabled],
            "verification_method": "mcpatlas_reference_trajectory_tool_within_enabled_tools",
        })

    for c, enabled in mcp_wrong_easy:
        gold_name = c["call"]["name"]
        diff_prefix = [t for t in enabled if t != gold_name and t.split("_")[0] != gold_name.split("_")[0]]
        pool = diff_prefix if diff_prefix else [t for t in enabled if t != gold_name]
        wrong_name = rng.choice(pool)
        out_cases.append({
            "id": f"tsel:mcpatlas:wrong_easy:{c['id']}",
            "source_dataset": "mcpatlas_dev",
            "split": "dev",
            "kind": "wrong_tool",
            "difficulty": "easy",
            "is_natural": False,
            "label": False,
            "user_request": c["user"],
            "context": c["user"],
            "chosen_tool": wrong_name,
            "gold_tool": gold_name,
            "arguments": c["call"].get("arguments", {}),
            "available_tools": [{"name": t} for t in enabled],
            "verification_method": "mcpatlas_reference_trajectory_mismatch_cross_server_sibling",
        })

    AMBIGUOUS_MCP_PAIRS = {
        ("filesystem_list_directory_with_sizes", "filesystem_list_directory"),
        ("wikipedia_search_wikipedia", "wikipedia_get_article"),
        ("national-parks_getParkDetails", "national-parks_findParks"),
    }

    for c, enabled in mcp_wrong_hard:
        gold_name = c["call"]["name"]
        wrong_name = _most_similar_sibling(gold_name, [{"name": t} for t in enabled])
        if (wrong_name, gold_name) in AMBIGUOUS_MCP_PAIRS:
            continue
        out_cases.append({
            "id": f"tsel:mcpatlas:wrong_hard:{c['id']}",
            "source_dataset": "mcpatlas_dev",
            "split": "dev",
            "kind": "wrong_tool",
            "difficulty": "hard",
            "is_natural": False,
            "label": False,
            "user_request": c["user"],
            "context": c["user"],
            "chosen_tool": wrong_name,
            "gold_tool": gold_name,
            "arguments": c["call"].get("arguments", {}),
            "available_tools": [{"name": t} for t in enabled],
            "verification_method": "mcpatlas_reference_trajectory_mismatch_most_similar_sibling",
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
            "difficulty": "right",
            "is_natural": True,
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
                    "difficulty": "natural",
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

    bfcl_overlap_ids = [c["id"] for c in out_cases if c["source_dataset"].startswith("bfcl")]
    ah_overlap_ids = [c["id"] for c in out_cases if c["source_dataset"] == "agenthallu"]

    out_path = ROOT / "benchmarks" / "data" / "tool_selection_eval.json"
    out_obj = {
        "seed": SEED,
        "status": "EXPLORATORY",
        "n_total": len(out_cases),
        "n_right_tool": sum(1 for x in out_cases if x["label"]),
        "n_wrong_tool": sum(1 for x in out_cases if not x["label"]),
        "n_wrong_easy": sum(1 for x in out_cases if x.get("difficulty") == "easy"),
        "n_wrong_hard": sum(1 for x in out_cases if x.get("difficulty") == "hard"),
        "n_wrong_natural": n_natural_wrong,
        "overlap_audit": {
            "bfcl_case_count": len(bfcl_overlap_ids),
            "bfcl_case_ids": bfcl_overlap_ids,
            "agenthallu_case_count": len(ah_overlap_ids),
            "agenthallu_case_ids": ah_overlap_ids,
        },
        "cases": out_cases,
    }
    out_path.write_text(json.dumps(out_obj, indent=1), encoding="utf-8")
    return {
        "path": str(out_path.relative_to(ROOT)),
        "n_total": out_obj["n_total"],
        "n_right": out_obj["n_right_tool"],
        "n_wrong": out_obj["n_wrong_tool"],
        "n_wrong_easy": out_obj["n_wrong_easy"],
        "n_wrong_hard": out_obj["n_wrong_hard"],
        "n_wrong_natural": n_natural_wrong,
    }


def build_disjoint_few_shot_pool() -> Dict[str, Any]:
    """Build few-shot pool with natural fcrb dev_train errors (wrong_selection, format, no_source) + clean + Glaive,
    verifying 0 tool overlap with dev_val and frozen_heldout."""
    ts = json.loads((ROOT / "benchmarks" / "data" / "deref" / "fcrb_tool_split.json").read_text(encoding="utf-8"))
    dev_train = ts["dev_train_cases"]
    forbidden_tools = set(ts["dev_val"]["tools"]) | set(ts["frozen_heldout"]["tools"])

    rng = random.Random(SEED)
    train_clean = [c for c in dev_train if c.get("label") is True]
    train_wrong_sel = []
    train_wrong_fmt = []
    train_wrong_other = []
    for c in dev_train:
        if c.get("label") is not False:
            continue
        meta = c.get("args_meta") or []
        meta_items = meta.values() if isinstance(meta, dict) else meta
        paths = [str(v.get("path", "")).lower() for v in meta_items if isinstance(v, dict)]
        if any("format" in p for p in paths):
            train_wrong_fmt.append(c)
        elif any("visible" in p for p in paths):
            train_wrong_sel.append(c)
        else:
            train_wrong_other.append(c)

    rng.shuffle(train_clean)
    rng.shuffle(train_wrong_sel)
    rng.shuffle(train_wrong_fmt)
    rng.shuffle(train_wrong_other)

    pool: List[Dict[str, Any]] = []
    for c in train_clean[:8]:
        assert c["tool_name"] not in forbidden_tools
        pool.append({
            "id": f"fewshot:fcrb_train:{c['id']}",
            "source_dataset": "fcrb_dev_train",
            "tool_name": c["tool_name"],
            "error_kind": "none",
            "premise": c["premise"],
            "call": c["call"],
            "supported": True,
            "unsupported_arguments": [],
        })
    for kind, bucket, k in [
        ("wrong_selection", train_wrong_sel, 4),
        ("wrong_format", train_wrong_fmt, 4),
        ("fabricated_value", train_wrong_other, 4),
    ]:
        for c in bucket[:k]:
            assert c["tool_name"] not in forbidden_tools
            pool.append({
                "id": f"fewshot:fcrb_train:{c['id']}",
                "source_dataset": "fcrb_dev_train",
                "tool_name": c["tool_name"],
                "error_kind": kind,
                "premise": c["premise"],
                "call": c["call"],
                "supported": False,
                "unsupported_arguments": [
                    v.get("arg") or v.get("argument") or ""
                    for v in (c.get("args_meta").values() if isinstance(c.get("args_meta"), dict) else (c.get("args_meta") or []))
                    if isinstance(v, dict) and v.get("wrong")
                ][:2],
            })

    # Add 4 Glaive v2 examples (disjoint from all fcrb/BFCL/AgentHallu/MCP-Atlas)
    glaive = json.loads((ROOT / "benchmarks" / "argground_bench.json").read_text(encoding="utf-8"))["cases"]
    g_pos = [c for c in glaive if c.get("label") is True][:2]
    g_neg = [c for c in glaive if c.get("label") is False][:2]
    for c in g_pos + g_neg:
        pool.append({
            "id": f"fewshot:glaive:{c['id']}",
            "source_dataset": "argground_bench_glaive_v2",
            "tool_name": (c.get("call") or {}).get("name", ""),
            "error_kind": "none" if c["label"] else "fabricated_value",
            "premise": c["premise"],
            "call": c["call"],
            "supported": bool(c["label"]),
            "unsupported_arguments": [k for k in (c["call"].get("arguments") or {}) if not c["label"]][:1],
        })

    rng.shuffle(pool)
    out_path = ROOT / "benchmarks" / "data" / "few_shot_pool.json"
    out_path.write_text(json.dumps({"seed": SEED, "n": len(pool), "examples": pool}, indent=1), encoding="utf-8")
    return {
        "path": str(out_path.relative_to(ROOT)),
        "n": len(pool),
        "n_natural_errors": sum(1 for x in pool if x["source_dataset"] == "fcrb_dev_train" and not x["supported"]),
    }


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
