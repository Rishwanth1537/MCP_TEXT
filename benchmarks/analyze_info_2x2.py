"""
Analysis of the clean information ablation (Deviation 22). Reads benchmarks/info2x2_<bench>_<condition>_<model>.json.

Per model and benchmark, on the cases that carry a tool definition: false-block rate, detection, Youden J per condition, and exact
McNemar tests on per-case correctness for the registered contrasts:
    information      C_def     vs A_plain      (definition alone)
    instruction      B_rule    vs A_plain      (rule sentence alone)
    rule given def   D_defrule vs C_def
    placebo          E_placebo vs A_plain      (extra text of the same length and form, no relevant information)
    information net  D_defrule vs E_placebo    (definition + rule against placebo + rule)
fcrb only: detection per condition split into "format-convention" errors (a bare city where the definition asks for 'City, State';
heuristic on argument names and the definition text, no labels) and all other errors.

  python -I benchmarks/analyze_info_2x2.py            (writes benchmarks/info2x2_summary.json)
"""
import ast
import json
import math
import re
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
B = ROOT / "benchmarks"
CONTRASTS = [("information", "C_def", "A_plain"), ("instruction", "B_rule", "A_plain"), ("rule_given_def", "D_defrule", "C_def"),
             ("placebo", "E_placebo", "A_plain"), ("information_net", "D_defrule", "E_placebo")]
DEF_FILE = {"fcrb": "cases_fcrb_schema.json", "cfb": "cases_cfb_x_def.json"}
PLAIN_FILE = {"fcrb": "cases_fcrb.json", "cfb": "cases_cfb_x_plain.json"}


def wilson(k, n, z=1.96):
    if n == 0:
        return [None, None]
    p = k / n
    c = (p + z * z / (2 * n)) / (1 + z * z / n)
    h = z * math.sqrt(p * (1 - p) / n + z * z / (4 * n * n)) / (1 + z * z / n)
    return [round(c - h, 3), round(c + h, 3)]


def mcnemar(a, b):
    x = sum(p and not q for p, q in zip(a, b))
    y = sum(q and not p for p, q in zip(a, b))
    n = x + y
    p = 1.0 if n == 0 else min(1.0, 2 * sum(math.comb(n, i) for i in range(min(x, y) + 1)) / 2 ** n)
    return {"only_first_correct": x, "only_second_correct": y, "p": round(p, 4)}


def blocked(r):
    return (not r["supported"]) if r["parsed"] else bool(r["label"])      # unparseable = miss, as in llm_judge_args.summarise


def lit(x):
    return x if not isinstance(x, str) else ast.literal_eval(x)


def format_convention_ids(bench):
    """fcrb: ids whose call has a location-like string argument without a comma while the definition asks for 'City, State'."""
    if bench != "fcrb":
        return set()
    out = set()
    for c in json.loads((B / "data/deref" / DEF_FILE[bench]).read_text(encoding="utf-8"))["cases"]:
        if "Tool definition" not in c["premise"]:
            continue
        d = c["premise"].split("Tool definition", 1)[1]
        if not re.search(r"City, State|city and state|e\.g\.? [A-Z][a-z]+, [A-Z]{2}|[A-Z][a-z]+ [A-Z][a-z]+, [A-Z]{2}\b", d):
            continue
        for call in lit(c.get("calls") or [c["call"]]):
            for k, v in (lit(call).get("arguments") or {}).items():
                if isinstance(v, str) and re.search(r"loc|city|place|area|destination|origin|address", k, re.I) and "," not in v:
                    out.add(c["id"])
    return out


def main():
    summary = {}
    files = sorted(B.glob("info2x2_*_*_*.json"))
    files = [f for f in files if f.name != "info2x2_summary.json"]
    if not files:
        sys.exit("no info2x2_*.json result files yet: run  python benchmarks/run_info_2x2.py  first")
    runs = {}
    for f in files:
        m = re.match(r"info2x2_(fcrb|cfb)_([A-E]_[a-z]+)_(\w+)\.json", f.name)
        if m:
            runs.setdefault((m.group(1), m.group(3)), {})[m.group(2)] = {r["id"]: r for r in json.loads(f.read_text(encoding="utf-8"))["rows"]}
    for (bench, tag), conds in sorted(runs.items()):
        plain = {c["id"]: c["premise"] for c in json.loads((B / "data/deref" / PLAIN_FILE[bench]).read_text(encoding="utf-8"))["cases"]}
        withdef = {c["id"] for c in json.loads((B / "data/deref" / DEF_FILE[bench]).read_text(encoding="utf-8"))["cases"]
                   if len(c["premise"]) > len(plain[c["id"]])}
        ids = sorted(set.intersection(*[set(r) for r in conds.values()]) & withdef)
        fmt = format_convention_ids(bench)
        any_rows = next(iter(conds.values()))
        wrong = [i for i in ids if not any_rows[i]["label"]]
        right = [i for i in ids if any_rows[i]["label"]]
        res = {"n": len(ids), "n_wrong": len(wrong), "n_correct": len(right), "conditions": {}, "contrasts": {},
               "n_format_convention_errors": len([i for i in wrong if i in fmt])}
        print(f"\n{bench} / {tag}: {len(ids)} cases with a definition ({len(wrong)} wrong, {len(right)} correct)")
        print("  %-11s %-22s %-22s %6s %6s | detection: format-convention / other" % ("condition", "false blocks [95%]", "detection [95%]", "J", "unpars"))
        for cond in sorted(conds):
            rows = conds[cond]
            fb = sum(blocked(rows[i]) for i in right)
            dt = sum(blocked(rows[i]) for i in wrong)
            f_ids = [i for i in wrong if i in fmt]
            o_ids = [i for i in wrong if i not in fmt]
            row = {"false_blocks": fb / max(1, len(right)), "false_blocks_ci": wilson(fb, len(right)), "detection": dt / max(1, len(wrong)),
                   "detection_ci": wilson(dt, len(wrong)), "unparseable": sum(not rows[i]["parsed"] for i in ids),
                   "detection_format_convention": [sum(blocked(rows[i]) for i in f_ids), len(f_ids)],
                   "detection_other": [sum(blocked(rows[i]) for i in o_ids), len(o_ids)]}
            row["J"] = row["detection"] - row["false_blocks"]
            res["conditions"][cond] = row
            print("  %-11s %5.1f%% %-15s %5.1f%% %-15s %+6.1f %6d | %d/%d   %d/%d" % (
                cond, 100 * row["false_blocks"], row["false_blocks_ci"], 100 * row["detection"], row["detection_ci"], 100 * row["J"],
                row["unparseable"], *row["detection_format_convention"], *row["detection_other"]))
        for name, x, y in CONTRASTS:
            if x in conds and y in conds:
                cx = [(not blocked(conds[x][i])) == conds[x][i]["label"] for i in ids]
                cy = [(not blocked(conds[y][i])) == conds[y][i]["label"] for i in ids]
                mc = mcnemar(cx, cy)
                dj = res["conditions"][x]["J"] - res["conditions"][y]["J"]
                res["contrasts"][name] = {"first": x, "second": y, "delta_J": round(dj, 4), **mc}
                print("  contrast %-16s %s vs %s: delta J %+5.1f points, McNemar %d vs %d, p = %.4f" % (
                    name, x, y, 100 * dj, mc["only_first_correct"], mc["only_second_correct"], mc["p"]))
        summary[f"{bench}/{tag}"] = res
    (B / "info2x2_summary.json").write_text(json.dumps(summary, indent=1), encoding="utf-8")
    print("\nwrote", B / "info2x2_summary.json")


if __name__ == "__main__":
    main()
