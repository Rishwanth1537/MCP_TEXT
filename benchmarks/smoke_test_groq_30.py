"""
30-case Live Smoke Test on Groq (openai/gpt-oss-120b) vs Cached Baseline (benchmarks/llm_judge_args_cache.jsonl).

Runs strictly 30 cases from fcrb_dev200 (seed=0) with a fresh cache under results/judge_2026-11-12_smoke30/
so all 30 calls hit the live Groq API, then reports per-case agreement with the existing cache.
"""
from __future__ import annotations

import datetime as dt
import json
import subprocess
import sys
import time
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
sys.path.insert(0, str(ROOT / "benchmarks"))

from judge_eval import get_environment_info, load_dataset  # noqa: E402
from judge_prompts import PROMPT_HASH, build_messages  # noqa: E402
from keys import get_key  # noqa: E402
from llm_judge_args import Cache, cache_key, chat, parse_verdict, trim_context  # noqa: E402


def main():
    out_dir = ROOT / "results" / "judge_2026-11-12_smoke30"
    out_dir.mkdir(parents=True, exist_ok=True)

    provider = "groq"
    model = "openai/gpt-oss-120b"
    key = get_key("GROQ_API_KEY") or ""
    if not key:
        raise RuntimeError("GROQ_API_KEY missing.")

    cases = load_dataset("fcrb_dev200", seed=0, limit=200)[:30]
    old_cache = Cache(ROOT / "benchmarks" / "llm_judge_args_cache.jsonl")
    live_cache = Cache(out_dir / "live_cache.jsonl")

    rpm = 10.0
    gap = 60.0 / rpm
    last_call = 0.0

    rows = []
    for c in cases:
        msgs = build_messages(0, trim_context(c["premise"], None), c.get("calls") or c["call"])
        ck = cache_key(provider, model, 0, msgs)

        old_hit = old_cache.get(ck)
        if old_hit is None:
            raise KeyError(f"Missing baseline cache entry for {c['id']}")
        old_parsed = parse_verdict(old_hit.get("text", ""))

        live_hit = live_cache.get(ck)
        if live_hit is None:
            wait = gap - (time.time() - last_call)
            if wait > 0:
                time.sleep(wait)
            last_call = time.time()
            out, lat = chat(provider, key, model, msgs, max_tokens=120, temperature=0)
            text = (out.get("choices") or [{}])[0].get("message", {}).get("content", "") or ""
            usage = out.get("usage") or {}
            live_hit = {
                "text": text,
                "latency_s": round(float(lat), 4),
                "prompt_tokens": usage.get("prompt_tokens"),
                "completion_tokens": usage.get("completion_tokens"),
            }
            live_cache.put(ck, live_hit)

        live_parsed = parse_verdict(live_hit.get("text", ""))
        old_sup = old_parsed[0] if old_parsed else None
        live_sup = live_parsed[0] if live_parsed else None
        agree = (old_sup == live_sup) and (old_parsed is not None) and (live_parsed is not None)
        rows.append({
            "id": c["id"],
            "label": bool(c["label"]),
            "cache_parsed": old_parsed is not None,
            "cache_supported": old_sup,
            "live_parsed": live_parsed is not None,
            "live_supported": live_sup,
            "agree": agree,
            "latency_s": live_hit.get("latency_s"),
            "prompt_tokens": live_hit.get("prompt_tokens"),
            "completion_tokens": live_hit.get("completion_tokens"),
        })

    n_agree = sum(r["agree"] for r in rows)
    n_unpars_live = sum(not r["live_parsed"] for r in rows)
    n_unpars_cache = sum(not r["cache_parsed"] for r in rows)

    def _rates(sup_key, parsed_key):
        ok = [r for r in rows if r["label"]]
        err = [r for r in rows if not r["label"]]
        fb = sum(1 for r in ok if (not r[parsed_key]) or (not r[sup_key]))
        det = sum(1 for r in err if r[parsed_key] and (not r[sup_key]))
        return {"fbr": fb / len(ok), "det": det / len(err), "n_ok": len(ok), "n_err": len(err)}

    git_hash = subprocess.check_output(["git", "rev-parse", "--short", "HEAD"], cwd=ROOT).decode().strip()
    report = {
        "timestamp": dt.datetime.now().isoformat(),
        "git_hash": git_hash,
        "environment": get_environment_info(),
        "provider": provider,
        "model": model,
        "prompt_hash": PROMPT_HASH,
        "n_cases": len(rows),
        "n_agree": n_agree,
        "agreement_rate": n_agree / len(rows),
        "n_unparseable_live": n_unpars_live,
        "n_unparseable_cache": n_unpars_cache,
        "cache_rates": _rates("cache_supported", "cache_parsed"),
        "live_rates": _rates("live_supported", "live_parsed"),
        "discordant_cases": [r for r in rows if not r["agree"]],
        "rows": rows,
    }
    (out_dir / "smoke_report.json").write_text(json.dumps(report, indent=1), encoding="utf-8")
    print(json.dumps({k: v for k, v in report.items() if k != "rows"}, indent=2))


if __name__ == "__main__":
    main()
