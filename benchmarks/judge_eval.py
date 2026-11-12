"""
Unified LLM Judge Evaluation Harness (J0-J8).

Takes (dataset, judge config) -> per-call verdict, continuous score, tokens, latency,
with JSONL caching keyed by (provider, model, prompt_hash, temperature, condition, messages).

Enforces Ground Rules:
- No gold labels, error-type labels, or dataset names in any prompt.
- Temperature pinned (default 0.0).
- Sequential execution respecting --rpm.
- Saves raw outputs, command, git hash, seed, prompt hash, and model under results/judge_<date>_<exp>/.
"""
from __future__ import annotations

import argparse
import datetime as dt
import hashlib
import json
import re
import subprocess
import sys
import time
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
sys.path.insert(0, str(ROOT / "benchmarks"))

from llm_judge_args import Cache, PROVIDERS, chat, parse_verdict, trim_context  # noqa: E402
from keys import get_key  # noqa: E402
from stats import check_exploratory, cluster_bootstrap_rate_ci, fast_auroc, mcnemar_exact, wilson_interval  # noqa: E402

SYSTEM_PROMPT_PLAIN = (
    "You are an impartial tool-call verifier. Given the user request, conversation context, "
    "and a proposed tool call, decide whether every argument value and the tool call are "
    "supported by the context (and tool definition, if provided).\n"
    'Respond ONLY with valid JSON: {"supported": true or false, "unsupported_arguments": [...], "confidence": 0-100}.'
)


def extract_auto_format_rules(tool_def_text: str) -> List[str]:
    """Automatically extract format conventions from a tool definition (no hand-written dataset rules)."""
    if not tool_def_text:
        return []
    rules: List[str] = []
    if re.search(r"City,\s*State|city\s+and\s+state|e\.g\.?\s*[A-Z][a-z]+,\s*[A-Z]{2}", tool_def_text):
        rules.append("Format convention from tool definition: location values requesting 'City, State' must include both city and state separated by a comma.")
    if re.search(r"YYYY-MM-DD", tool_def_text):
        rules.append("Format convention from tool definition: date parameters specifying YYYY-MM-DD must use ISO-8601 YYYY-MM-DD format.")
    if re.search(r"HH:MM", tool_def_text):
        rules.append("Format convention from tool definition: time parameters specifying HH:MM must use 24-hour HH:MM format.")
    if re.search(r'"enum"\s*:', tool_def_text):
        rules.append("Schema convention from tool definition: parameters with an 'enum' list must use one of the exact allowed enum values.")
    return rules


def build_judge_messages(
    case: Dict[str, Any],
    condition: str = "A_plain",
    max_context_chars: Optional[int] = None,
) -> Tuple[List[Dict[str, str]], str]:
    """Build (messages, prompt_hash) with zero gold labels, error types, or dataset names."""
    if condition == "A_plain":
        premise = case.get("premise", "")
    elif condition in ("C_def", "D_def_autorule"):
        premise = case.get("premise_with_schema") or case.get("premise", "")
    else:
        raise ValueError(f"Unknown condition: {condition}")

    premise = trim_context(premise, max_context_chars)
    call = case.get("calls") or case.get("call")
    call_str = json.dumps(call, ensure_ascii=False)

    extra_rules = ""
    if condition == "D_def_autorule":
        def_part = premise.split("Tool definition", 1)[1] if "Tool definition" in premise else ""
        auto_rules = extract_auto_format_rules(def_part)
        if auto_rules:
            extra_rules = "\n\nExtracted conventions:\n- " + "\n- ".join(auto_rules)

    user_msg = f"Context:\n{premise}{extra_rules}\n\nProposed tool call:\n{call_str}"
    msgs = [
        {"role": "system", "content": SYSTEM_PROMPT_PLAIN},
        {"role": "user", "content": user_msg},
    ]
    prompt_hash = hashlib.sha256(SYSTEM_PROMPT_PLAIN.encode("utf-8")).hexdigest()[:16]
    return msgs, prompt_hash


def judge_cache_key(
    provider: str,
    model: str,
    prompt_hash: str,
    temperature: float,
    condition: str,
    messages: List[Dict[str, str]],
) -> str:
    payload = [provider, model, prompt_hash, float(temperature), condition, messages]
    return hashlib.sha256(json.dumps(payload, sort_keys=True).encode("utf-8")).hexdigest()


def parse_judge_output(text: str) -> Dict[str, Any]:
    parsed_basic = parse_verdict(text)
    if parsed_basic is None:
        return {"parsed": False, "supported": None, "unsupported_arguments": [], "confidence": None, "error_score": 0.5}
    sup, un = parsed_basic
    conf = None
    m = re.search(r"\{.*\}", text, re.S)
    if m:
        try:
            obj = json.loads(m.group(0))
            if isinstance(obj.get("confidence"), (int, float)):
                conf = max(0.0, min(100.0, float(obj["confidence"])))
        except Exception:
            pass
    if conf is not None:
        p_err = (100.0 - conf) / 100.0 if sup else conf / 100.0
    else:
        p_err = 0.0 if sup else 1.0
    return {
        "parsed": True,
        "supported": sup,
        "unsupported_arguments": un,
        "confidence": conf,
        "error_score": float(p_err),
    }


def load_dataset(dataset_name: str, split: str = "dev", limit: Optional[int] = None, seed: int = 0) -> List[Dict[str, Any]]:
    if dataset_name == "fcrb_dev200":
        plain = json.loads((ROOT / "benchmarks/data/deref/cases_fcrb.json").read_text(encoding="utf-8"))["cases"]
        schema = {c["id"]: c["premise"] for c in json.loads((ROOT / "benchmarks/data/deref/cases_fcrb_schema.json").read_text(encoding="utf-8"))["cases"]}
        pool = [c for c in plain if c.get("split") == "dev"]
        idx = np.random.default_rng(seed).permutation(len(pool))[: (limit or 200)]
        out = []
        for i in sorted(idx):
            c = dict(pool[i])
            c["premise_with_schema"] = schema.get(c["id"], c["premise"])
            out.append(c)
        return out
    if dataset_name == "fcrb_tool_dev":
        d = json.loads((ROOT / "benchmarks/data/deref/fcrb_tool_split.json").read_text(encoding="utf-8"))
        cases = d["dev_cases"]
        return cases[:limit] if limit else cases
    if dataset_name == "agenthallu":
        cases = json.loads((ROOT / "benchmarks/data/agenthallu/cases_eval.json").read_text(encoding="utf-8"))["cases"]
        return cases[:limit] if limit else cases
    if dataset_name == "mcpatlas_dev":
        d = json.loads((ROOT / "benchmarks/data/mcpatlas/mcpatlas_tool_split.json").read_text(encoding="utf-8"))
        cases = d["dev_cases"]
        return cases[:limit] if limit else cases
    raise ValueError(f"Unsupported dataset: {dataset_name} (held-out splits require explicit J8 runner).")


def evaluate_judge(
    cases: List[Dict[str, Any]],
    provider: str,
    model: str,
    condition: str = "A_plain",
    temperature: float = 0.0,
    rpm: float = 10.0,
    max_context_chars: Optional[int] = None,
    cache_path: Path = ROOT / "benchmarks" / "judge_eval_cache.jsonl",
    dry_run: bool = False,
    chat_fn=chat,
) -> Dict[str, Any]:
    if not cases:
        raise ValueError("evaluate_judge received 0 cases.")

    key = ""
    if not dry_run:
        key_env = PROVIDERS[provider]["key_env"]
        key = get_key(key_env) or ""
        if not key:
            raise RuntimeError(f"Missing API key {key_env} for provider {provider}.")

    cache = Cache(cache_path)
    gap = 60.0 / rpm if rpm else 0.0
    last_call = 0.0
    rows = []

    for i, c in enumerate(cases):
        msgs, phash = build_judge_messages(c, condition=condition, max_context_chars=max_context_chars)
        if dry_run:
            if i < 2:
                print(f"[DRY RUN item {i}] prompt_hash={phash} msgs={json.dumps(msgs)[:500]}")
            continue
        ck = judge_cache_key(provider, model, phash, temperature, condition, msgs)
        hit = cache.get(ck)
        if hit is None:
            wait = gap - (time.time() - last_call)
            if wait > 0:
                time.sleep(wait)
            last_call = time.time()
            out, lat = chat_fn(provider, key, model, msgs, max_tokens=220, temperature=temperature)
            text = (out.get("choices") or [{}])[0].get("message", {}).get("content", "") or ""
            usage = out.get("usage") or {}
            hit = {
                "text": text,
                "latency_s": round(float(lat), 4),
                "prompt_tokens": usage.get("prompt_tokens"),
                "completion_tokens": usage.get("completion_tokens"),
            }
            cache.put(ck, hit)
        parsed = parse_judge_output(hit.get("text", ""))
        call_obj = c.get("call") or (c.get("calls") or [{}])[0]
        rows.append({
            "id": c["id"],
            "tool_name": call_obj.get("name", "unknown"),
            "label": bool(c["label"]),
            "parsed": parsed["parsed"],
            "supported": parsed["supported"],
            "unsupported_arguments": parsed["unsupported_arguments"],
            "confidence": parsed["confidence"],
            "error_score": parsed["error_score"],
            "latency_s": hit.get("latency_s"),
            "prompt_tokens": hit.get("prompt_tokens"),
            "completion_tokens": hit.get("completion_tokens"),
        })

    if dry_run:
        return {"dry_run": True, "n_cases": len(cases)}

    labels = np.array([r["label"] for r in rows], dtype=bool)
    blocked = np.array([(not r["supported"]) if r["parsed"] else bool(r["label"]) for r in rows], dtype=bool)
    scores = np.array([r["error_score"] for r in rows], dtype=float)
    n_pos = int((~labels).sum())
    n_neg = int(labels.sum())
    fb = int(blocked[labels].sum())
    dt_cnt = int(blocked[~labels].sum())
    fbr = fb / n_neg if n_neg else 0.0
    det = dt_cnt / n_pos if n_pos else 0.0
    lats = [r["latency_s"] for r in rows if r["latency_s"] is not None]
    toks = [(r["prompt_tokens"] or 0) + (r["completion_tokens"] or 0) for r in rows if r["prompt_tokens"] is not None]

    return {
        "n": len(rows),
        "n_correct": n_neg,
        "n_wrong": n_pos,
        "exploratory": check_exploratory(n_pos, label="judge_eval"),
        "false_block_rate": fbr,
        "false_block_ci_wilson": wilson_interval(fb, n_neg),
        "detection_rate": det,
        "detection_ci_wilson": wilson_interval(dt_cnt, n_pos),
        "youden_j": det - fbr,
        "auroc": float(fast_auroc(~labels, scores)) if (n_pos > 0 and n_neg > 0) else None,
        "n_unparseable": sum(1 for r in rows if not r["parsed"]),
        "latency_p50_s": float(np.median(lats)) if lats else None,
        "latency_p95_s": float(np.percentile(lats, 95)) if lats else None,
        "tokens_mean": float(np.mean(toks)) if toks else None,
        "rows": rows,
    }


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--dataset", required=True, choices=["fcrb_dev200", "fcrb_tool_dev", "agenthallu", "mcpatlas_dev"])
    ap.add_argument("--provider", default="groq")
    ap.add_argument("--model", required=True)
    ap.add_argument("--condition", default="A_plain", choices=["A_plain", "C_def", "D_def_autorule"])
    ap.add_argument("--temperature", type=float, default=0.0)
    ap.add_argument("--rpm", type=float, default=10.0)
    ap.add_argument("--limit", type=int, default=None)
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--max-context-chars", type=int, default=None)
    ap.add_argument("--out", required=True)
    ap.add_argument("--dry-run", action="store_true")
    a = ap.parse_args()

    cases = load_dataset(a.dataset, limit=a.limit, seed=a.seed)
    res = evaluate_judge(
        cases,
        provider=a.provider,
        model=a.model,
        condition=a.condition,
        temperature=a.temperature,
        rpm=a.rpm,
        max_context_chars=a.max_context_chars,
        dry_run=a.dry_run,
    )
    git_hash = subprocess.check_output(["git", "rev-parse", "--short", "HEAD"], cwd=ROOT).decode().strip()
    meta = {
        "timestamp": dt.datetime.now().isoformat(),
        "git_hash": git_hash,
        "command": " ".join(sys.argv),
        "dataset": a.dataset,
        "provider": a.provider,
        "model": a.model,
        "condition": a.condition,
        "temperature": a.temperature,
        "seed": a.seed,
        **res,
    }
    out_path = Path(a.out)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    out_path.write_text(json.dumps(meta, indent=1), encoding="utf-8")
    print(f"Wrote {out_path}")


if __name__ == "__main__":
    main()
