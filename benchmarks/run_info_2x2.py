"""
Run the clean information ablation (Deviation 22). Works in PowerShell, cmd and bash; resumable (every reply is cached).

  python benchmarks/run_info_2x2.py --dry-run          show what would run, call no API
  python benchmarks/run_info_2x2.py                     NVIDIA Nemotron 4B and 550B, fcrb then cfb
  python benchmarks/run_info_2x2.py --models 4b         only the small, fast model
  python benchmarks/run_info_2x2.py --bench fcrb        only fcrb

Needs NVIDIA_API_KEY in the environment or in ~/.mcp_checker_keys.env (python benchmarks/keys.py shows which keys are visible).
Build the case files first: python -I benchmarks/make_info_2x2_benches.py
Then: python -I benchmarks/analyze_info_2x2.py
"""
import argparse
import os
import subprocess
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
MODELS = {"4b": ("nvidia/nemotron-3.5-bolt-e4b", 20), "550b": ("nvidia/nemotron-3-ultra-550b-a55b", 8),
          "120b": ("nvidia/nemotron-3.5-super-vl-120b-a12b", 20), "30b": ("nvidia/nemotron-3.5-lightning-30b-a3b", 20)}
# condition -> (case file, limit); fcrb A and D reuse the files of the earlier ladder so their cached replies are reused
CONDITIONS = {
    "fcrb": [("A_plain", "cases_fcrb.json"), ("B_rule", "cases_fcrb_x_rule.json"), ("C_def", "cases_fcrb_schema.json"),
             ("D_defrule", "cases_fcrb_schema_note.json"), ("E_placebo", "cases_fcrb_x_placebo.json")],
    "cfb": [("A_plain", "cases_cfb_x_plain.json"), ("C_def", "cases_cfb_x_def.json"), ("D_defrule", "cases_cfb_x_defrule.json")],
}
LIMIT = {"fcrb": 200, "cfb": 100}


def out_name(bench, cond, tag):
    return ROOT / "benchmarks" / f"info2x2_{bench}_{cond}_{tag}.json"


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--models", default="4b,550b")
    ap.add_argument("--bench", default="fcrb,cfb")
    ap.add_argument("--skip", default="", help="comma-separated conditions to leave out, e.g. E_placebo")
    ap.add_argument("--dry-run", action="store_true")
    a = ap.parse_args()
    env = dict(os.environ, PYTHONPATH=str(ROOT), PYTHONIOENCODING="utf-8")
    (ROOT / "benchmarks" / "logs_info2x2").mkdir(exist_ok=True)
    for tag in a.models.split(","):
        model, rpm = MODELS[tag]
        for bench in a.bench.split(","):
            for cond, fname in CONDITIONS[bench]:
                if cond in a.skip.split(","):
                    continue
                case_file = ROOT / "benchmarks" / "data" / "deref" / fname
                if not case_file.exists():
                    sys.exit(f"missing {case_file}: run  python -I benchmarks/make_info_2x2_benches.py  first")
                out = out_name(bench, cond, tag)
                cmd = [sys.executable, "-u", str(ROOT / "benchmarks" / "llm_judge_args.py"), "--provider", "nvidia", "--model", model,
                       "--bench", str(case_file), "--split", "dev", "--limit", str(LIMIT[bench]), "--rpm", str(rpm), "--out", str(out)]
                if a.dry_run:
                    print("would run:", bench, cond, tag, "->", out.name)
                    continue
                if out.exists():
                    print("done already:", out.name)
                    continue
                print("running:", bench, cond, tag, flush=True)
                log = ROOT / "benchmarks" / "logs_info2x2" / f"{bench}_{cond}_{tag}.log"
                with open(log, "w", encoding="utf-8") as f:
                    rc = subprocess.run(cmd, env=env, cwd=ROOT, stdout=f, stderr=subprocess.STDOUT).returncode
                if rc != 0:
                    print(f"  FAILED (exit {rc}); see {log}. Re-run this command to resume from the cache.")
                    return 1
    return 0


if __name__ == "__main__":
    sys.exit(main())
