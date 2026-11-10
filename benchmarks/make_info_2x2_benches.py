"""
Clean information ablation (Deviation 22): build the case files for a 2 x 2 design plus a placebo, so that the effect of the
tool DEFINITION can be separated from the effect of the added RULE SENTENCE and from the effect of extra text.

fcrb (same order and ids as cases_fcrb.json, so a seeded sample picks the same cases):
    A  plain                              cases_fcrb.json                      (exists)
    B  rule sentence only                 cases_fcrb_x_rule.json               (new)
    C  definition only                    cases_fcrb_schema.json               (exists)
    D  definition + rule sentence         cases_fcrb_schema_note.json          (exists)
    E  placebo: definition of ANOTHER function sharing no parameter name with the call, length-matched, + rule      cases_fcrb_x_placebo.json   (new)
cfb (context budget applied to the CONVERSATION first, the definition appended afterwards, so every condition keeps the same
3,000 characters of conversation; run these WITHOUT --max-context-chars):
    A  cases_cfb_x_plain.json     C  cases_cfb_x_def.json     D  cases_cfb_x_defrule.json

Only cases that carry a definition in C/D get any addition in B and E. Nothing here reads a label.

  python -I benchmarks/make_info_2x2_benches.py
"""
import json
import random
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
D = ROOT / "benchmarks" / "data" / "deref"
DEF_HEAD = "\n\nTool definition:\n"
RULE_HEAD = ("\n\nTool definition (a value that violates it, meaning wrong type, not one of the allowed enum values, or not in the format the "
             "description asks for, is NOT supported):\n")
RULE_ONLY = ("\n\nNote: a value that violates the tool's definition, meaning wrong type, not one of the allowed enum values, or not in the "
             "format the tool's description asks for, is NOT supported.")
CFB_BUDGET = 3000


def load(name):
    return json.loads((D / name).read_text(encoding="utf-8"))


def save(name, obj, note):
    (D / name).write_text(json.dumps(obj), encoding="utf-8")
    print(f"{name}: {note}")


def trim(text, n):
    return text if len(text) <= n else "[earlier context truncated]\n" + text[-n:]


def param_names(def_text):
    out = set()
    for line in def_text.splitlines():
        try:
            fn = json.loads(line)
        except ValueError:
            continue
        out |= set(((fn.get("parameters") or {}).get("properties") or {}).keys())
    return out


def fcrb():
    a, c = load("cases_fcrb.json"), load("cases_fcrb_schema.json")
    defs = {}
    for x, y in zip(a["cases"], c["cases"]):
        assert x["id"] == y["id"] and y["premise"].startswith(x["premise"])
        tail = y["premise"][len(x["premise"]):]
        if tail:
            assert tail.startswith(DEF_HEAD)
            defs[x["id"]] = tail[len(DEF_HEAD):]
    # B: rule sentence only
    b = load("cases_fcrb.json")
    for x in b["cases"]:
        if x["id"] in defs:
            x["premise"] += RULE_ONLY
    save("cases_fcrb_x_rule.json", b, f"{len(defs)} cases get the rule sentence, no definition")
    # E: placebo = definition text of a different function with the closest length (seeded tie-break), same header as D
    rng = random.Random(22)
    pool = sorted(set(defs.values()), key=len)
    e = load("cases_fcrb.json")
    diffs = []
    for x in e["cases"]:
        own = defs.get(x["id"])
        if own is None:
            continue
        calls = [cl for cl in (x.get("calls") or [x["call"]]) if isinstance(cl, dict)]
        names = {cl.get("name") for cl in calls}
        argn = {k for cl in calls for k in (cl.get("arguments") or {})}
        # another function that shares NO parameter name with this call (so it cannot carry the format of any argument); names only
        cand = [d for d in pool if d != own and not any(n and f'"name": "{n}"' in d for n in names)
                and not (param_names(d) & argn)]
        best = min(abs(len(d) - len(own)) for d in cand)
        pick = rng.choice([d for d in cand if abs(len(d) - len(own)) == best])
        diffs.append(abs(len(pick) - len(own)))
        x["premise"] += RULE_HEAD + pick
    diffs.sort()
    save("cases_fcrb_x_placebo.json", e, f"{len(diffs)} cases get another function's definition; length difference median "
         f"{diffs[len(diffs) // 2]} chars, max {diffs[-1]}")


def cfb():
    a, d = load("cases_cfb.json"), load("cases_cfb_schema_note.json")
    plain, defonly, defrule = load("cases_cfb.json"), load("cases_cfb.json"), load("cases_cfb.json")
    n = 0
    for i, (x, y) in enumerate(zip(a["cases"], d["cases"])):
        assert x["id"] == y["id"] and y["premise"].startswith(x["premise"])
        tail = y["premise"][len(x["premise"]):]
        conv = trim(x["premise"], CFB_BUDGET)
        plain["cases"][i]["premise"] = conv
        defonly["cases"][i]["premise"] = conv
        defrule["cases"][i]["premise"] = conv
        if tail:
            assert tail.startswith(RULE_HEAD)
            n += 1
            defonly["cases"][i]["premise"] = conv + DEF_HEAD + tail[len(RULE_HEAD):]
            defrule["cases"][i]["premise"] = conv + tail
    save("cases_cfb_x_plain.json", plain, f"conversation cut to the last {CFB_BUDGET} characters")
    save("cases_cfb_x_def.json", defonly, f"{n} cases: same conversation + definition")
    save("cases_cfb_x_defrule.json", defrule, f"{n} cases: same conversation + definition + rule sentence")


if __name__ == "__main__":
    fcrb()
    cfb()
