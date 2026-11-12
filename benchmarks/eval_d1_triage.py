"""
Step D1 Evaluation: Frozen 3-Tier Guard + J1/J2 Judge Triage (Dev Only).

Compares on dev datasets:
1. Frozen Guard Alone (B2c alpha=0.10)
2. D1-A: Strict ASK-only -> Judge (Guard ALLOW and BLOCK final)
3. D1-B: Flagged (ASK + BLOCK) -> Judge (Guard ALLOW final; judge rescues guard false blocks)
4. J1+J2 Judge Alone (when --run-all is set)

Uses J1 (+definition + checker_hints + autorules) + J2 (cite_then_judge strict JSON with confidence).
Respects Groq rate limits sequentially with caching in benchmarks/judge_eval_cache.jsonl.
"""
from __future__ import annotations

import argparse
import datetime as dt
import json
import subprocess
import sys
from pathlib import Path
from typing import Any, Dict, List

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
sys.path.insert(0, str(ROOT / "benchmarks"))

from judge_eval import evaluate_judge, get_environment_info  # noqa: E402
from server.pipeline.arg_ground import call_argument_class  # noqa: E402
from server.pipeline.provenance import check_call, evidence_from_agenthallu_premise  # noqa: E402
from server.tools.verify_arguments import is_low_confidence_arg  # noqa: E402
from stats import check_exploratory, fast_auroc, mcnemar_exact, wilson_interval  # noqa: E402

TAU_B2C = {"control_only": 0.5526, "datetime": 0.0, "entity_or_id": 0.8158, "search_query": 0.7143}
TAU_ASK = 0.88


def guard_decision(case: Dict[str, Any], mode: str) -> str:
    call = case.get("call") or (case.get("calls") or [{}])[0]
    args = call.get("arguments") or {}
    tname = call.get("name", "")
    if not args:
        return "ALLOW"
    if mode == "mcpatlas":
        ev = [("host", "Current date and time: 2025-07-15T12:00:00Z"), ("user", case["user"])] + [
            ("tool", str(t)) for t in case.get("tool_results", [])
        ]
    elif mode == "agenthallu":
        ev = evidence_from_agenthallu_premise(case["premise"])
    else:
        ev = [("user", case["premise"])]

    cv = check_call(args, ev, config="P3", tool_name=tname)
    cls = call_argument_class(args)
    tau_block = TAU_B2C[cls]
    if cv.score >= TAU_ASK:
        return "ALLOW"
    if cv.score < tau_block:
        ungrounded = [lv.path.split(".")[0] for lv in cv.leaves if lv.verdict == "UNGROUNDED"]
        if any(not is_low_confidence_arg(a) for a in ungrounded):
            return "BLOCK"
    return "ASK"


def summarize_binary(labels: np.ndarray, blocked: np.ndarray) -> Dict[str, Any]:
    n_ok = int(labels.sum())
    n_err = int((~labels).sum())
    fb = int(blocked[labels].sum()) if n_ok else 0
    dt_cnt = int(blocked[~labels].sum()) if n_err else 0
    return {
        "n": int(len(labels)),
        "n_correct": n_ok,
        "n_wrong": n_err,
        "false_block_rate": fb / n_ok if n_ok else 0.0,
        "false_block_ci_wilson": wilson_interval(fb, n_ok) if n_ok else [0.0, 0.0],
        "detection_rate": dt_cnt / n_err if n_err else 0.0,
        "detection_ci_wilson": wilson_interval(dt_cnt, n_err) if n_err else [0.0, 0.0],
    }


def matched_fbr_catch(labels: np.ndarray, scores: np.ndarray, target_fbr: float) -> Dict[str, Any]:
    ok_scores = scores[labels]
    if len(ok_scores) == 0 or (~labels).sum() == 0:
        return {"target_fbr": target_fbr, "realized_fbr": None, "catch_rate": None}
    # Highest threshold achieving FBR <= target_fbr
    best_tau, best_fbr, best_det = 1.01, 0.0, 0.0
    for tau in np.unique(np.concatenate([scores, [1.01]])):
        b = scores >= tau
        fbr = float(b[labels].mean())
        det = float(b[~labels].mean())
        if fbr <= target_fbr and (det > best_det or (det == best_det and fbr < best_fbr)):
            best_tau, best_fbr, best_det = float(tau), fbr, det
    return {"target_fbr": target_fbr, "tau": best_tau, "realized_fbr": best_fbr, "catch_rate": best_det}


def run_d1_on_dataset(
    name: str,
    cases: List[Dict[str, Any]],
    mode: str,
    provider: str,
    model: str,
    rpm: float,
    run_all: bool,
    use_checker_hints: bool = True,
) -> Dict[str, Any]:
    # Ensure unified 'premise' and 'call' for mcpatlas
    norm_cases = []
    for c in cases:
        nc = dict(c)
        if mode == "mcpatlas":
            nc["premise"] = f"User: {c['user']}\n" + "\n".join(f"[tool result] {t}" for t in c.get("tool_results", []))
            nc["label"] = True
        norm_cases.append(nc)

    decisions = [guard_decision(c, mode) for c in norm_cases]
    eval_indices = [i for i, d in enumerate(decisions) if (run_all or d in ("ASK", "BLOCK"))]
    to_eval = [norm_cases[i] for i in eval_indices]

    j1_factors = {
        "definition": True,
        "other_tools": False,
        "source_labelled": True,
        "checker_hints": use_checker_hints,
        "checker_score": False,
        "checker_scope": "frozen_pre_b3" if mode == "agenthallu" else "in_sample",
        "autorules": True,
    }

    judge_rows_by_id: Dict[str, Dict[str, Any]] = {}
    if to_eval:
        res_j = evaluate_judge(
            to_eval,
            provider=provider,
            model=model,
            condition="D_def_autorule",
            schema_mode="j2",
            variant="cite_then_judge",
            j2_order="cite_then_judge",
            j1_factors=j1_factors,
            temperature=0.0,
            rpm=rpm,
            max_context_chars=6000,
        )
        for r in res_j["rows"]:
            judge_rows_by_id[r["id"]] = r

    labels = np.array([bool(c["label"]) for c in norm_cases], dtype=bool)
    guard_blk = np.array([d == "BLOCK" for d in decisions], dtype=bool)
    guard_any = np.array([d in ("ASK", "BLOCK") for d in decisions], dtype=bool)

    d1a_blk = np.zeros(len(norm_cases), dtype=bool)
    d1b_blk = np.zeros(len(norm_cases), dtype=bool)
    d1b_scores = np.zeros(len(norm_cases), dtype=float)
    judge_blk = np.zeros(len(norm_cases), dtype=bool)
    judge_scores = np.zeros(len(norm_cases), dtype=float)

    for i, c in enumerate(norm_cases):
        d = decisions[i]
        jr = judge_rows_by_id.get(c["id"])
        j_b = ((not jr["supported"]) if jr["parsed"] else False) if jr else False
        j_s = jr["error_score"] if jr else 0.0

        # D1-A: ALLOW/BLOCK final by guard, ASK -> judge
        if d == "BLOCK":
            d1a_blk[i] = True
        elif d == "ASK":
            d1a_blk[i] = j_b

        # D1-B: ALLOW final by guard, ASK+BLOCK -> judge
        if d in ("ASK", "BLOCK"):
            d1b_blk[i] = j_b
            d1b_scores[i] = j_s

        judge_blk[i] = j_b
        judge_scores[i] = j_s

    toks = [
        (r.get("prompt_tokens") or 0) + (r.get("completion_tokens") or 0)
        for r in judge_rows_by_id.values()
        if r.get("prompt_tokens") is not None
    ]
    lats = [r["latency_s"] for r in judge_rows_by_id.values() if r.get("latency_s") is not None]

    out: Dict[str, Any] = {
        "dataset": name,
        "n_total": len(norm_cases),
        "exploratory": check_exploratory(int((~labels).sum()), label=name),
        "guard_counts": {k: decisions.count(k) for k in ("ALLOW", "ASK", "BLOCK")},
        "guard_block_only": summarize_binary(labels, guard_blk),
        "guard_any_intervention": summarize_binary(labels, guard_any),
        "d1a_ask_only_judge": summarize_binary(labels, d1a_blk),
        "d1b_flagged_judge": summarize_binary(labels, d1b_blk),
        "judge_calls_d1a": decisions.count("ASK"),
        "judge_calls_d1b": decisions.count("ASK") + decisions.count("BLOCK"),
        "mean_tokens_per_judge_call": float(np.mean(toks)) if toks else None,
        "effective_tokens_per_call_d1b": (float(np.mean(toks)) * (decisions.count("ASK") + decisions.count("BLOCK")) / len(norm_cases)) if toks else None,
        "latency_p50_s": float(np.median(lats)) if lats else None,
    }
    if (~labels).any() and labels.any():
        out["d1b_auroc"] = float(fast_auroc(~labels, d1b_scores))
        out["d1b_at_5pct_fbr"] = matched_fbr_catch(labels, d1b_scores, 0.05)
        out["d1b_at_10pct_fbr"] = matched_fbr_catch(labels, d1b_scores, 0.10)
    if run_all:
        out["judge_alone"] = summarize_binary(labels, judge_blk)
        if (~labels).any() and labels.any():
            out["judge_alone_auroc"] = float(fast_auroc(~labels, judge_scores))
            out["judge_alone_at_5pct_fbr"] = matched_fbr_catch(labels, judge_scores, 0.05)
            out["judge_alone_at_10pct_fbr"] = matched_fbr_catch(labels, judge_scores, 0.10)
            out["mcnemar_d1b_vs_guard_block"] = mcnemar_exact(
                [(not b) == l for b, l in zip(d1b_blk, labels)],
                [(not b) == l for b, l in zip(guard_blk, labels)],
            )
    return out


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--dataset", required=True, choices=["agenthallu", "mcpatlas_dev_200", "fcrb_tool_val"])
    ap.add_argument("--provider", default="groq")
    ap.add_argument("--model", default="openai/gpt-oss-120b")
    ap.add_argument("--rpm", type=float, default=15.0)
    ap.add_argument("--run-all", action="store_true", help="Also run judge on guard-ALLOW calls for Judge-Alone baseline.")
    ap.add_argument("--no-checker-hints", action="store_true", help="Omit deterministic checker hints to avoid anchoring the judge.")
    ap.add_argument("--exclude-dev200", action="store_true", help="For fcrb_tool_val, exclude in_dev200 cases.")
    ap.add_argument("--limit", type=int, default=None, help="Optional cap on number of cases.")
    ap.add_argument("--out", required=True)
    a = ap.parse_args()

    if a.dataset == "agenthallu":
        cases = json.loads((ROOT / "benchmarks/data/agenthallu/cases_eval.json").read_text(encoding="utf-8"))["cases"]
        mode = "agenthallu"
    elif a.dataset == "mcpatlas_dev_200":
        cases = json.loads((ROOT / "benchmarks/data/mcpatlas/mcpatlas_tool_split.json").read_text(encoding="utf-8"))["dev_cases"][:200]
        mode = "mcpatlas"
    else:
        cases = json.loads((ROOT / "benchmarks/data/deref/fcrb_tool_split.json").read_text(encoding="utf-8"))["dev_val_cases"]
        if a.exclude_dev200:
            cases = [c for c in cases if not c.get("in_dev200")]
        mode = "fcrb"
    if a.limit:
        cases = cases[: a.limit]

    res = run_d1_on_dataset(a.dataset, cases, mode, a.provider, a.model, a.rpm, a.run_all, use_checker_hints=not a.no_checker_hints)
    git_hash = subprocess.check_output(["git", "rev-parse", "--short", "HEAD"], cwd=ROOT).decode().strip()
    payload = {
        "timestamp": dt.datetime.now().isoformat(),
        "git_hash": git_hash,
        "environment": get_environment_info(),
        "provider": a.provider,
        "model": a.model,
        **res,
    }
    out_path = Path(a.out)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    out_path.write_text(json.dumps(payload, indent=1), encoding="utf-8")
    print(json.dumps({k: v for k, v in payload.items() if k != "environment"}, indent=2))


if __name__ == "__main__":
    main()
