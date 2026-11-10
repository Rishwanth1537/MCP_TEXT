"""
E1 provenance-aware grounding (Deviation 21 in review/PREREGISTRATION.md).

  python -I benchmarks/eval_provenance.py --split design     MCP-Atlas design half only (may be run repeatedly while implementing)
  python -I benchmarks/eval_provenance.py --split holdout    MCP-Atlas held-out half + AgentHallu + AgentDojo, ONCE with the frozen code
                                                             (appends an entry to benchmarks/test_run_ledger.jsonl)

Writes benchmarks/provenance_<split>.json.
"""
import argparse
import ast
import datetime
import hashlib
import json
import math
import random
import sys
import time
from collections import Counter, defaultdict
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
sys.path.insert(0, str(ROOT / "benchmarks"))
from server.pipeline.arg_ground import check_arguments  # noqa: E402
from server.pipeline.provenance import (  # noqa: E402
    check_call, evidence_from_agentdojo_premise, evidence_from_agenthallu_premise,
)

B = ROOT / "benchmarks"


def wilson(k, n, z=1.96):
    if n == 0:
        return [None, None]
    p = k / n
    c = (p + z * z / (2 * n)) / (1 + z * z / n)
    h = z * math.sqrt(p * (1 - p) / n + z * z / (4 * n * n)) / (1 + z * z / n)
    return [round(c - h, 4), round(c + h, 4)]


def rate(k, n):
    return {"k": k, "n": n, "rate": round(k / n, 4) if n else None, "wilson95": wilson(k, n)}


def mcnemar_exact(a_correct, b_correct):
    only_a = sum(a and not b for a, b in zip(a_correct, b_correct))
    only_b = sum(b and not a for a, b in zip(a_correct, b_correct))
    n = only_a + only_b
    if n == 0:
        return {"only_a_correct": 0, "only_b_correct": 0, "p": 1.0}
    k = min(only_a, only_b)
    p = min(1.0, 2 * sum(math.comb(n, i) for i in range(k + 1)) / 2 ** n)
    return {"only_a_correct": only_a, "only_b_correct": only_b, "p": round(p, 4)}


def literal(x):
    if not isinstance(x, str):
        return x
    try:
        return json.loads(x)
    except ValueError:
        return ast.literal_eval(x)


# ------------------------------------------------------------------ MCP-Atlas
def mcpatlas(half):
    cases = [c for c in json.loads((B / "data/mcpatlas/cases_traj.json").read_text(encoding="utf-8"))["cases"] if c["half"] == half]
    out = {"n_calls": len(cases), "n_tasks": len({c["task"] for c in cases})}
    variants = {
        "P0_user_only": None,
        "P1_tool_outputs": ("P1", False),
        "P2_nested_dates": ("P2", False),
        "P3_E1": ("P3", False),
        "P3_E1_plus_agent_text": ("P3", True),
    }
    examples = []
    for name, spec in variants.items():
        t0 = time.time()
        rows = []
        for c in cases:
            args = c["call"]["arguments"] or {}
            if spec is None:
                vs = check_arguments(args, c["user"])
                leaves = [(v.argument, v.verdict, "user" if v.verdict == "GROUNDED" else "") for v in vs]
            else:
                ev = [("user", c["user"])] + [("tool", t["text"]) for t in c["tool_results"]] + [("agent", a) for a in c["agent"]]
                cv = check_call(args, ev, spec[0], include_agent=spec[1])
                leaves = [(lv.path, lv.verdict, lv.source) for lv in cv.leaves]
                if name == "P3_E1" and cv.blocked:
                    examples.append({"id": c["id"], "tool": c["call"]["name"],
                                     "ungrounded": [[lv.path, str(lv.value)[:80]] for lv in cv.leaves if lv.verdict == "UNGROUNDED"]})
            rows.append({"idx": c["idx"], "n": len(args), "blocked": any(v == "UNGROUNDED" for _, v, _ in leaves), "leaves": leaves})
        wa = [r for r in rows if r["n"] > 0]
        res = {
            "all_calls": rate(sum(r["blocked"] for r in rows), len(rows)),
            "calls_with_arguments": rate(sum(r["blocked"] for r in wa), len(wa)),
            "first_calls": rate(sum(r["blocked"] for r in wa if r["idx"] == 0), sum(r["idx"] == 0 for r in wa)),
            "later_calls": rate(sum(r["blocked"] for r in wa if r["idx"] > 0), sum(r["idx"] > 0 for r in wa)),
            "leaf_verdicts": dict(Counter(v for r in rows for _, v, _ in r["leaves"])),
            "grounded_by_source": dict(Counter(s for r in rows for _, v, s in r["leaves"] if v == "GROUNDED")),
            "seconds": round(time.time() - t0, 1),
        }
        by = defaultdict(Counter)
        for r in rows:
            for p, v, _ in r["leaves"]:
                by[p.split(".")[0].split("[")[0]][v] += 1
        res["top_ungrounded_args"] = sorted(([a, c["UNGROUNDED"], sum(c.values())] for a, c in by.items() if c["UNGROUNDED"]),
                                            key=lambda x: -x[1])[:12]
        out[name] = res
        print(f"  MCP-Atlas {half} {name:24s} blocked {res['calls_with_arguments']['rate']:.3f} {res['calls_with_arguments']['wilson95']}"
              f"  first {res['first_calls']['rate']:.3f}  later {res['later_calls']['rate']:.3f}  ({res['seconds']}s)")
    random.Random(21).shuffle(examples)
    out["P3_blocked_examples_seeded_sample"] = examples[:25]
    return out


# ------------------------------------------------------------------ AgentHallu
def agenthallu():
    ev_cases = json.loads((B / "data/agenthallu/cases_eval.json").read_text(encoding="utf-8"))["cases"]
    all_cases = json.loads((B / "data/agenthallu/cases.json").read_text(encoding="utf-8"))["cases"]
    out = {}

    def blocked(c, name):
        args = literal(c["call"])["arguments"] or {}
        if name == "P0_existing_premise":
            return any(v.verdict == "UNGROUNDED" for v in check_arguments(args, c["premise"]))
        cfg, agent = {"P1": ("P1", False), "P2": ("P2", False), "P3_E1": ("P3", False), "P3_E1_plus_agent_text": ("P3", True)}[name]
        return check_call(args, evidence_from_agenthallu_premise(c["premise"]), cfg, include_agent=agent).blocked

    names = ["P0_existing_premise", "P1", "P2", "P3_E1", "P3_E1_plus_agent_text"]
    ok = [literal(c["label"]) in (True, "True") for c in ev_cases]
    base = None
    for name in names:
        b = [blocked(c, name) for c in ev_cases]
        correct = [(not x) == o for x, o in zip(b, ok)]
        fb = sum(x for x, o in zip(b, ok) if o)
        dt = sum(x for x, o in zip(b, ok) if not o)
        clean_all = [c for c in all_cases if literal(c["label"]) in (True, "True")]
        fb_all = sum(blocked(c, name) for c in clean_all)
        res = {"detection": rate(dt, sum(not o for o in ok)), "false_blocks_eval93": rate(fb, sum(ok)),
               "false_blocks_all_clean": rate(fb_all, len(clean_all))}
        if base is None:
            base = correct
        else:
            res["mcnemar_vs_P0"] = mcnemar_exact(correct, base)
        out[name] = res
        print(f"  AgentHallu {name:24s} detection {res['detection']['rate']:.3f} {res['detection']['wilson95']}  "
              f"false blocks {res['false_blocks_eval93']['rate']:.3f} (all clean {res['false_blocks_all_clean']['rate']:.3f})")
    return out


# ------------------------------------------------------------------ AgentDojo (secondary, heuristic labels)
def agentdojo(max_neg=4000):
    cases = json.loads((B / "data/agentdojo/cases.json").read_text(encoding="utf-8"))["cases"]
    rng = random.Random(5)                      # same sampling as analyze_agentdojo.py
    out = {}
    tf = lambda x: x in (True, "True")          # noqa: E731
    for model in sorted({c["model"] for c in cases}):
        pos = [c for c in cases if c["model"] == model and tf(c["attack_goal_call"]) and tf(c["injected"]) and tf(c["run_security"])
               and c["attack_type"] not in (None, "None")]
        neg = [c for c in cases if c["model"] == model and c["attack_type"] in (None, "None")]
        rng.shuffle(neg)
        neg = neg[:max_neg]
        res = {"n_pos": len(pos), "n_neg": len(neg)}
        for name in ("P0_context", "P3_E1"):
            def blk(c):
                args = literal(c["call"]).get("arguments", {}) or {}
                if name == "P0_context":
                    return any(v.verdict == "UNGROUNDED" for v in check_arguments(args, c["premise"]))
                return check_call(args, evidence_from_agentdojo_premise(c["premise"]), "P3").blocked
            pb, nb = sum(map(blk, pos)), sum(map(blk, neg))
            res[name] = {"detection": rate(pb, len(pos)), "false_blocks": rate(nb, len(neg))}
            if pos and neg:
                print(f"  AgentDojo {model[:24]:24s} {name:10s} detection {pb / len(pos):.3f}  false blocks {nb / len(neg):.3f}")
        out[model] = res
    return out


def sha(p):
    return hashlib.sha256(Path(p).read_bytes()).hexdigest()[:16]


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--split", choices=["design", "holdout"], required=True)
    a = ap.parse_args()
    code_hash = {p: sha(ROOT / p) for p in ("server/pipeline/provenance.py", "server/pipeline/arg_ground.py", "benchmarks/eval_provenance.py")}
    res = {"split": a.split, "when": datetime.datetime.now().isoformat(timespec="seconds"), "code_sha256_16": code_hash,
           "mcpatlas": mcpatlas(a.split)}
    if a.split == "holdout":
        res["agenthallu"] = agenthallu()
        res["agentdojo"] = agentdojo()
        with open(B / "test_run_ledger.jsonl", "a", encoding="utf-8") as f:
            f.write(json.dumps({"when": res["when"], "script": "eval_provenance.py --split holdout (Deviation 21, E1)",
                                "cases": "mcpatlas cases_traj.json holdout half; agenthallu cases_eval.json c0b3ef1d76baf28a + cases.json; agentdojo cases.json",
                                "code_sha256_16": code_hash, "note": "single run with frozen E1 code"}) + "\n")
    (B / f"provenance_{a.split}.json").write_text(json.dumps(res, indent=1, default=str), encoding="utf-8")
    print("wrote", B / f"provenance_{a.split}.json")


if __name__ == "__main__":
    main()
