"""
Exploratory control for E1 (not part of the Deviation 21 hypotheses; written after the held-out run):
does the larger evidence of P3 (user text + all earlier tool outputs) hide wrong values?

For each MCP-Atlas call (design half) with at least one string or number leaf, one leaf is replaced by a same-type leaf value taken
from a call of a DIFFERENT task (seeded). A swapped value that happens to occur in this call's evidence is skipped (it would not be
wrong in any checkable sense). Reported: share of swapped calls P0 (user text only) and P3 block, overall and by value type.

  python -I benchmarks/provenance_swap_control.py      (writes benchmarks/provenance_swap_control_design.json)
"""
import json
import random
import sys
from collections import defaultdict
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
from server.pipeline.arg_ground import _norm_text, check_arguments  # noqa: E402
from server.pipeline.provenance import CONTROL_ARGS, PROGRAM_ARGS, check_call  # noqa: E402

B = ROOT / "benchmarks"


def leaves(obj, path=""):
    if isinstance(obj, dict):
        for k, v in obj.items():
            yield from leaves(v, f"{path}.{k}" if path else k)
    elif isinstance(obj, list):
        for i, v in enumerate(obj):
            yield from leaves(v, f"{path}[{i}]")
    else:
        yield path, obj


def set_leaf(obj, path, value):
    import re
    parts = re.findall(r"[^.\[\]]+|\[\d+\]", path)
    cur = obj
    for p in parts[:-1]:
        cur = cur[int(p[1:-1])] if p.startswith("[") else cur[p]
    last = parts[-1]
    if last.startswith("["):
        cur[int(last[1:-1])] = value
    else:
        cur[last] = value


def kind(v):
    if isinstance(v, bool) or v is None:
        return None
    if isinstance(v, (int, float)):
        return "number"
    if isinstance(v, str) and v.strip() and len(v) <= 200:
        return "string"
    return None


def main():
    cases = [c for c in json.loads((B / "data/mcpatlas/cases_traj.json").read_text(encoding="utf-8"))["cases"] if c["half"] == "design"]
    rng = random.Random(2109)
    pool = defaultdict(list)
    for c in cases:
        for p, v in leaves(c["call"]["arguments"] or {}):
            key = p.split(".")[-1].split("[")[0].lower()
            if kind(v) and key not in PROGRAM_ARGS and key not in CONTROL_ARGS:
                pool[kind(v)].append((c["task"], v))
    rows = []
    for c in cases:
        cand = [(p, v) for p, v in leaves(c["call"]["arguments"] or {})
                if kind(v) and p.split(".")[-1].split("[")[0].lower() not in PROGRAM_ARGS | CONTROL_ARGS]
        if not cand:
            continue
        p, v = rng.choice(cand)
        evid_text = _norm_text(c["user"] + "\n" + "\n".join(t["text"] for t in c["tool_results"])[-2_000_000:])
        for _ in range(50):
            task, new = rng.choice(pool[kind(v)])
            if task != c["task"] and new != v and _norm_text(str(new)) not in evid_text:
                break
        else:
            continue
        args = json.loads(json.dumps(c["call"]["arguments"]))
        set_leaf(args, p, new)
        ev = [("user", c["user"])] + [("tool", t["text"]) for t in c["tool_results"]]
        b0 = any(x.verdict == "UNGROUNDED" for x in check_arguments(args, c["user"]))
        b3 = check_call(args, ev, "P3").blocked
        rows.append({"id": c["id"], "kind": kind(v), "first_call": c["idx"] == 0, "path": p, "new": str(new)[:60], "P0": b0, "P3": b3})
    out = {}
    for name, sel in (("all", rows), ("string", [r for r in rows if r["kind"] == "string"]),
                      ("number", [r for r in rows if r["kind"] == "number"]),
                      ("first_call", [r for r in rows if r["first_call"]]), ("later_call", [r for r in rows if not r["first_call"]])):
        n = len(sel)
        out[name] = {"n": n, "P0_blocked": round(sum(r["P0"] for r in sel) / n, 4) if n else None,
                     "P3_blocked": round(sum(r["P3"] for r in sel) / n, 4) if n else None}
        print(name, out[name])
    out["missed_by_P3_examples"] = [r for r in rows if not r["P3"]][:30]
    (B / "provenance_swap_control_design.json").write_text(json.dumps(out, indent=1), encoding="utf-8")


if __name__ == "__main__":
    main()
