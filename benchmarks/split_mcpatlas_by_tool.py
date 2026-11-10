"""
Tool-grouped split of MCP-Atlas into calibration/dev and frozen held-out (Phase B protocol).

Every tool (call['name']) is assigned strictly to either calibration/dev or frozen_heldout.
No tool appears in both splits.
Frozen and saved to benchmarks/data/mcpatlas/mcpatlas_tool_split.json.
"""

import hashlib
import json
import random
from collections import defaultdict
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
DATA_FILE = ROOT / "benchmarks" / "data" / "mcpatlas" / "cases_traj.json"
OUT_FILE = ROOT / "benchmarks" / "data" / "mcpatlas" / "mcpatlas_tool_split.json"


def make_tool_split(seed=42, target_dev_ratio=0.5):
    cases = json.loads(DATA_FILE.read_text(encoding="utf-8"))["cases"]
    
    # Group cases by tool name
    by_tool = defaultdict(list)
    for c in cases:
        tool_name = c["call"].get("name", "unknown")
        by_tool[tool_name].append(c)
        
    tools = sorted(by_tool.keys())
    rng = random.Random(seed)
    # Shuffle tools deterministically
    shuffled_tools = list(tools)
    rng.shuffle(shuffled_tools)
    
    dev_tools = set()
    heldout_tools = set()
    
    total_cases = len(cases)
    target_dev_cases = total_cases * target_dev_ratio
    
    dev_cases_count = 0
    # Greedy partition to get close to 50/50 case split while grouping strictly by tool
    # Sort tools by descending call count for smooth bin packing
    sorted_tools = sorted(tools, key=lambda t: len(by_tool[t]), reverse=True)
    rng.shuffle(sorted_tools)  # shuffle with seed to avoid deterministic tool-name bias
    
    # Alternating or greedy knapsack for balanced split
    curr_dev_count = 0
    curr_heldout_count = 0
    
    # Sort by size to balance
    tools_by_size = sorted(tools, key=lambda t: (len(by_tool[t]), t), reverse=True)
    for t in tools_by_size:
        t_len = len(by_tool[t])
        if curr_dev_count <= curr_heldout_count:
            dev_tools.add(t)
            curr_dev_count += t_len
        else:
            heldout_tools.add(t)
            curr_heldout_count += t_len
            
    print(f"Total tools: {len(tools)}, Total calls: {total_cases}")
    print(f"Calibration/Dev tools: {len(dev_tools)}, calls: {curr_dev_count} ({100*curr_dev_count/total_cases:.1f}%)")
    print(f"Frozen Heldout tools:  {len(heldout_tools)}, calls: {curr_heldout_count} ({100*curr_heldout_count/total_cases:.1f}%)")
    
    # Verify strict disjointness
    assert len(dev_tools & heldout_tools) == 0, "Tool overlap detected!"
    
    dev_cases = []
    heldout_cases = []
    for c in cases:
        t = c["call"].get("name", "unknown")
        c_copy = dict(c)
        if t in dev_tools:
            c_copy["tool_split"] = "calibration_dev"
            dev_cases.append(c_copy)
        else:
            c_copy["tool_split"] = "frozen_heldout"
            heldout_cases.append(c_copy)
            
    res = {
        "seed": seed,
        "n_total_cases": total_cases,
        "n_total_tools": len(tools),
        "calibration_dev": {
            "n_tools": len(dev_tools),
            "n_cases": len(dev_cases),
            "tools": sorted(dev_tools),
        },
        "frozen_heldout": {
            "n_tools": len(heldout_tools),
            "n_cases": len(heldout_cases),
            "tools": sorted(heldout_tools),
        },
        "dev_cases": dev_cases,
        "heldout_cases": heldout_cases,
    }
    
    serialized = json.dumps(res, indent=2, ensure_ascii=False)
    sha256 = hashlib.sha256(serialized.encode("utf-8")).hexdigest()
    res["sha256"] = sha256
    
    OUT_FILE.write_text(json.dumps(res, indent=2, ensure_ascii=False), encoding="utf-8")
    print(f"\nWrote frozen tool split -> {OUT_FILE.relative_to(ROOT)}")
    print(f"SHA256: {sha256}")
    return res


if __name__ == "__main__":
    make_tool_split()
