"""
Pre-B2 Reconciliation, Independent Validation, B3b Fire Check, Paired Cluster Bootstrap,
and Near-Miss Scale Test.
"""

import json
import re
import sys
from pathlib import Path
from collections import defaultdict, Counter
import numpy as np

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))

from server.pipeline.provenance import check_call
from server.pipeline.arg_ground import check_value, _spelled_numbers, _unit_conversions, numbers_in
from server.tools.verify_arguments import is_low_confidence_arg, run_argument_grounding
from server.models.schemas import ArgumentGroundingRequest, Decision
from benchmarks.stats import cluster_bootstrap_rate_ci, wilson_interval

SPLIT_FILE = ROOT / "benchmarks" / "data" / "mcpatlas" / "mcpatlas_tool_split.json"
CACHE_16686A4 = ROOT / "benchmarks" / "data" / "mcpatlas" / "calibration_flags_cache.json"

def paired_cluster_bootstrap_delta(diffs, clusters, n_resamples=5000, seed=42):
    arr_d = np.asarray(diffs, dtype=float)
    arr_c = np.asarray(clusters)
    point_est = float(np.mean(arr_d))
    unique_c = np.unique(arr_c)
    n_c = len(unique_c)
    cluster_indices = [np.where(arr_c == c)[0] for c in unique_c]
    cluster_sums = np.array([np.sum(arr_d[idx]) for idx in cluster_indices])
    cluster_lens = np.array([len(idx) for idx in cluster_indices])
    rng = np.random.default_rng(seed)
    boot_idx = rng.integers(0, n_c, size=(n_resamples, n_c))
    boot_means = np.sum(cluster_sums[boot_idx], axis=1) / np.sum(cluster_lens[boot_idx], axis=1)
    ci_low, ci_high = np.percentile(boot_means, [2.5, 97.5])
    return point_est, (float(ci_low), float(ci_high))


def test_b3b_firing(dev_cases, ah_cases, fcrb_clean_cases):
    """Test whether B3b (compound numbers >99, ordinals, fractions, unit conversions) fires on any real call."""
    # Original simple _ONES/_TENS regex numbers + standard numeric tokens without B3b extensions:
    _ONES = {w: i for i, w in enumerate("zero one two three four five six seven eight nine ten eleven twelve thirteen fourteen fifteen sixteen seventeen eighteen nineteen".split())}
    _TENS = {w: 10 * i for i, w in enumerate("_ _ twenty thirty forty fifty sixty seventy eighty ninety".split()) if w != "_"}
    _NUM_TOKEN = re.compile(r"(?<![\w.])(?:(?<![\w])(-)(?=\d))?(\d[\d,]*(?:\.\d+)?)\s*(k|m|million|thousand|%)?", re.I)

    def orig_numbers_in(text):
        vals = set()
        for m in _NUM_TOKEN.finditer(text):
            try:
                v = float(m.group(2).replace(",", ""))
            except ValueError:
                continue
            suf = (m.group(3) or "").lower()
            mine = {v}
            if suf in ("k", "thousand"): mine.add(v * 1e3)
            elif suf in ("m", "million"): mine.add(v * 1e6)
            if suf == "%": mine.add(v / 100.0)
            if m.group(1): mine |= {-x for x in mine if x > 0}
            vals |= mine
        # simple spelled up to 99
        toks = re.sub(r"[^\w\s]", " ", text.lower()).replace("-", " ").split()
        for i, t in enumerate(toks):
            if t in _ONES: vals.add(float(_ONES[t]))
            elif t in _TENS:
                val = _TENS[t]
                if i + 1 < len(toks) and toks[i+1] in _ONES and 0 < _ONES[toks[i+1]] < 10:
                    val += _ONES[toks[i+1]]
                vals.add(float(val))
            elif t in ("hundred", "thousand", "million"):
                mult = {"hundred": 100, "thousand": 1000, "million": 1_000_000}[t]
                base = float(_ONES.get(toks[i-1], 1)) if i > 0 and toks[i-1] in _ONES else 1.0
                vals.add(base * mult)
        return vals

    def extract_numeric_leaves(args):
        out = []
        def walk(v):
            if isinstance(v, dict):
                for x in v.values(): walk(x)
            elif isinstance(v, list):
                for x in v: walk(x)
            elif isinstance(v, (int, float)) and not isinstance(v, bool):
                out.append(float(v))
            elif isinstance(v, str) and re.fullmatch(r"\s*[\d,]+(?:\.\d+)?\s*", v):
                out.append(float(v.replace(",", "")))
        walk(args)
        return out

    b3b_hits = []
    for ds_name, cases, get_ctx in [
        ("mcpatlas_calib", dev_cases, lambda c: c["user"] + " " + " ".join(t["text"] for t in (c.get("tool_results") or []))),
        ("agenthallu", ah_cases, lambda c: c.get("premise", "")),
        ("fcrb_clean", fcrb_clean_cases, lambda c: c.get("premise", "")),
    ]:
        hits = 0
        for c in cases:
            num_leaves = extract_numeric_leaves(c["call"].get("arguments", {}))
            if not num_leaves:
                continue
            ctx = get_ctx(c)
            nums_new = numbers_in(ctx)
            nums_old = orig_numbers_in(ctx)
            for fv in num_leaves:
                in_new = fv in nums_new or any(abs(fv - x) < 1e-9 for x in nums_new)
                in_old = fv in nums_old or any(abs(fv - x) < 1e-9 for x in nums_old)
                if in_new and not in_old:
                    hits += 1
                    b3b_hits.append((ds_name, c["id"], fv))
        print(f"B3b firing count on {ds_name} (n={len(cases)}): {hits}")
    return b3b_hits


def run():
    print("=== LOADING DATASETS ===")
    split_data = json.loads(SPLIT_FILE.read_text(encoding="utf-8"))
    dev_cases = split_data["dev_cases"]
    cache_old = json.loads(CACHE_16686A4.read_text(encoding="utf-8"))
    old_block_ids = {c["id"] for c in cache_old["block_cases"]}
    old_ask_ids = {c["id"] for c in cache_old["ask_cases"]}

    # Independent datasets
    ah_file = ROOT / "benchmarks" / "data" / "agenthallu" / "cases_eval.json"
    ah_cases = json.loads(ah_file.read_text(encoding="utf-8"))["cases"]
    ah_cleans = [c for c in ah_cases if bool(c["label"])]
    ah_errors = [c for c in ah_cases if not bool(c["label"])]

    fcrb_all = {c["id"]: c for c in json.loads((ROOT / "benchmarks" / "data" / "deref" / "cases_fcrb.json").read_text(encoding="utf-8"))["cases"]}
    seed_rows = json.loads((ROOT / "benchmarks" / "llm_judge_args_fcrb_agent_sonnet_dev_v0.json").read_text(encoding="utf-8"))["rows"]
    fcrb_cleans = [fcrb_all[r["id"]] for r in seed_rows if r["label"]]

    audit_records = json.loads((ROOT / "review" / "audit3" / "error_taxonomy_audit.json").read_text(encoding="utf-8"))
    fcrb_fab_ids = {r["id"] for r in audit_records if r["category"] == "fabricated_value"}
    fcrb_fabs = [c for c in fcrb_all.values() if c["id"] in fcrb_fab_ids]

    print("\n=== 1. CHECKING B3b REAL-CALL FIRING ===")
    b3b_hits = test_b3b_firing(dev_cases, ah_cases, fcrb_cleans)

    print("\n=== 2. INDEPENDENT VALIDATION: AGENTHALLU CLEAN (n=93) & FCRB CLEAN (n=94) ===")
    for name, dataset in [("AgentHallu Clean", ah_cleans), ("DeRef fcrb Clean", fcrb_cleans)]:
        blocks, asks, allows = 0, 0, 0
        for c in dataset:
            req = ArgumentGroundingRequest(
                arguments=c["call"].get("arguments", {}),
                context=c.get("premise", ""),
                decision_mode="three_tier",
                tau_block=0.50,
                tau_ask=0.88,
            )
            res = run_argument_grounding(req)
            if res.decision == Decision.BLOCK: blocks += 1
            elif res.decision == Decision.ASK: asks += 1
            else: allows += 1
        n = len(dataset)
        any_int = blocks + asks
        print(f"Post-B3 {name} (n={n}): BLOCK={blocks}/{n} ({blocks/n*100:.1f}%), ASK={asks}/{n} ({asks/n*100:.1f}%), Any={any_int}/{n} ({any_int/n*100:.1f}%)")

    print("\n=== 3. PAIRED ANALYSIS ON MCP-ATLAS CALIBRATION (n=1,212) ===")
    # Evaluate Post-B3 (current code) across dev_cases
    new_block_ids = set()
    new_ask_ids = set()
    tools = []
    old_block_vec = []
    new_block_vec = []
    diff_block_vec = []
    diff_any_vec = []

    for c in dev_cases:
        cid = c["id"]
        tname = c["call"]["name"]
        tools.append(tname)
        was_b = 1 if cid in old_block_ids else 0
        was_a = 1 if cid in old_ask_ids else 0
        was_any = 1 if (was_b or was_a) else 0
        old_block_vec.append(was_b)

        args = c["call"].get("arguments", {})
        if not args:
            is_b, is_a = 0, 0
        else:
            ev = [("user", c["user"])] + [("tool", t["text"]) for t in (c.get("tool_results") or [])]
            cv = check_call(args, ev, config="P3")
            if not cv.blocked:
                is_b, is_a = 0, 0
            else:
                ungrounded = [lv for lv in cv.leaves if lv.verdict == "UNGROUNDED"]
                if all(is_low_confidence_arg(lv.path.split(".")[0].split("[")[0]) for lv in ungrounded):
                    is_b, is_a = 0, 1
                    new_ask_ids.add(cid)
                else:
                    is_b, is_a = 1, 0
                    new_block_ids.add(cid)
        is_any = 1 if (is_b or is_a) else 0
        new_block_vec.append(is_b)
        diff_block_vec.append(is_b - was_b)
        diff_any_vec.append(is_any - was_any)

    fixed_ids = old_block_ids - new_block_ids
    regressed_ids = new_block_ids - old_block_ids
    fixed_tools = {c["call"]["name"] for c in dev_cases if c["id"] in fixed_ids}
    regressed_tools = {c["call"]["name"] for c in dev_cases if c["id"] in regressed_ids}

    delta_b, delta_b_ci = paired_cluster_bootstrap_delta(diff_block_vec, tools)
    delta_any, delta_any_ci = paired_cluster_bootstrap_delta(diff_any_vec, tools)

    print(f"Pre-B3 (16686a4):  BLOCK={len(old_block_ids)}/1212 ({len(old_block_ids)/1212*100:.2f}%), ASK={len(old_ask_ids)}/1212 ({len(old_ask_ids)/1212*100:.2f}%)")
    print(f"Post-B3 (current): BLOCK={len(new_block_ids)}/1212 ({len(new_block_ids)/1212*100:.2f}%), ASK={len(new_ask_ids)}/1212 ({len(new_ask_ids)/1212*100:.2f}%)")
    print(f"Paired Delta BLOCK (Post vs 16686a4): {delta_b*100:+.2f} pts (Cluster CI: [{delta_b_ci[0]*100:+.2f}%, {delta_b_ci[1]*100:+.2f}%])")
    print(f"Paired Delta Any-Int (Post vs 16686a4): {delta_any*100:+.2f} pts (Cluster CI: [{delta_any_ci[0]*100:+.2f}%, {delta_any_ci[1]*100:+.2f}%])")
    print(f"Fixed blocks: {len(fixed_ids)} across {len(fixed_tools)} tools: {sorted(fixed_tools)}")
    print(f"Added token-collision tightened blocks: {len(regressed_ids)} across {len(regressed_tools)} tools: {sorted(regressed_tools)}")

    print("\n=== 4. NEAR-MISS SCALE TEST ON FCRB FABRICATED VALUES (n=362) & PERTURBED VARIANTS ===")
    # Evaluate exact recall on original 362 fcrb fabricated values + synthetic near-miss variants of clean values
    orig_caught_any, orig_caught_block = 0, 0
    for c in fcrb_fabs:
        res = run_argument_grounding(ArgumentGroundingRequest(
            arguments=c["call"].get("arguments", {}),
            context=c.get("premise", ""),
            decision_mode="three_tier", tau_block=0.50, tau_ask=0.88
        ))
        if res.decision in (Decision.BLOCK, Decision.ASK): orig_caught_any += 1
        if res.decision == Decision.BLOCK: orig_caught_block += 1

    # Perturb every numeric / string / date arg in fcrb_cleans (94 cases) to generate near-miss adversarial calls at scale!
    nm_total = 0
    nm_caught_any = 0
    nm_caught_block = 0
    for c in fcrb_cleans:
        args = dict(c["call"].get("arguments", {}))
        ctx = c.get("premise", "")
        mutated = False
        new_args = {}
        for k, v in args.items():
            if not mutated and isinstance(v, (int, float)) and not isinstance(v, bool):
                new_args[k] = v + 1 if v != 0 else 1
                mutated = True
            elif not mutated and isinstance(v, str) and re.match(r"^\d{4}-\d{2}-\d{2}", v):
                # shift day or year
                new_args[k] = "2099-01-02"
                mutated = True
            elif not mutated and isinstance(v, str) and len(v) > 2:
                # off-by-one character swap / digit perturbation
                new_args[k] = v[:-1] + ("Z" if v[-1] != "Z" else "X")
                mutated = True
            else:
                new_args[k] = v
        if mutated:
            nm_total += 1
            res = run_argument_grounding(ArgumentGroundingRequest(
                arguments=new_args, context=ctx,
                decision_mode="three_tier", tau_block=0.50, tau_ask=0.88
            ))
            if res.decision in (Decision.BLOCK, Decision.ASK): nm_caught_any += 1
            if res.decision == Decision.BLOCK: nm_caught_block += 1

    print(f"fcrb Fabricated (n={len(fcrb_fabs)}): Union Recall = {orig_caught_any}/{len(fcrb_fabs)} ({orig_caught_any/len(fcrb_fabs)*100:.1f}%), BLOCK Recall = {orig_caught_block}/{len(fcrb_fabs)} ({orig_caught_block/len(fcrb_fabs)*100:.1f}%)")
    print(f"Near-Miss Scale Perturbed (n={nm_total}): Union Recall = {nm_caught_any}/{nm_total} ({nm_caught_any/nm_total*100:.1f}%), BLOCK Recall = {nm_caught_block}/{nm_total} ({nm_caught_block/nm_total*100:.1f}%)")

    results_summary = {
        "b3b_hits": b3b_hits,
        "pre_b3_16686a4": {"block": len(old_block_ids), "ask": len(old_ask_ids)},
        "post_b3_current": {"block": len(new_block_ids), "ask": len(new_ask_ids)},
        "paired_delta_block_pts": round(delta_b * 100, 2),
        "paired_delta_block_ci": [round(delta_b_ci[0] * 100, 2), round(delta_b_ci[1] * 100, 2)],
        "fixed_count": len(fixed_ids),
        "fixed_tools": sorted(fixed_tools),
        "regressed_count": len(regressed_ids),
        "regressed_tools": sorted(regressed_tools),
        "fcrb_fab_recall_union": round(orig_caught_any / len(fcrb_fabs) * 100, 2),
        "fcrb_fab_recall_block": round(orig_caught_block / len(fcrb_fabs) * 100, 2),
        "near_miss_total": nm_total,
        "near_miss_caught_any": nm_caught_any,
        "near_miss_recall_union": round(nm_caught_any / nm_total * 100, 2) if nm_total else 0,
    }
    out = ROOT / "results" / "pre_b2_reconciliation.json"
    out.write_text(json.dumps(results_summary, indent=2), encoding="utf-8")
    print(f"\nSaved reconciliation JSON to {out}")

if __name__ == "__main__":
    run()
