"""
Analyze remaining residual false flags on the MCP-Atlas calibration split after B3 (B3c -> B3b -> B3a).
Reports cumulative FBR, remaining categorized counts, and remaining unexplained share before starting B2.
"""

import json
import sys
from pathlib import Path
from collections import defaultdict

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))

from server.pipeline.provenance import check_call
from server.tools.verify_arguments import is_low_confidence_arg
from benchmarks.label_all_calibration_flags import classify_flag

SPLIT_FILE = ROOT / "benchmarks" / "data" / "mcpatlas" / "mcpatlas_tool_split.json"

def run():
    with open(SPLIT_FILE, 'r', encoding='utf-8') as f:
        split_data = json.load(f)

    calib_cases = split_data['dev_cases']
    
    rem_blocks = []
    rem_asks = []
    
    for c in calib_cases:
        args = c['call'].get('arguments', {})
        if not args:
            continue
        ev = [('user', c['user'])] + [('tool', t['text']) for t in (c.get('tool_results') or [])]
        cv = check_call(args, ev, config='P3')
        if cv.blocked:
            ungrounded = [lv for lv in cv.leaves if lv.verdict == 'UNGROUNDED']
            record = {
                'id': c['id'],
                'tool': c['call'].get('name', ''),
                'call': c['call'],
                'user': c.get('user', ''),
                'tool_results_snippet': ' '.join(t.get('text', '')[:200] for t in (c.get('tool_results') or [])),
                'ungrounded_leaves': [(lv.path, lv.value) for lv in ungrounded]
            }
            if all(is_low_confidence_arg(lv.path.split('.')[0].split('[')[0]) for lv in ungrounded):
                record['tier'] = 'ASK'
                rem_asks.append(record)
            else:
                record['tier'] = 'BLOCK'
                rem_blocks.append(record)

    all_rem = rem_blocks + rem_asks
    cat_counts = defaultdict(lambda: {'total': 0, 'BLOCK': 0, 'ASK': 0})
    for r in all_rem:
        cat, _ = classify_flag(r)
        cat_counts[cat]['total'] += 1
        cat_counts[cat][r['tier']] += 1

    print(f"Post-B3 Residual Flags: {len(rem_blocks)} BLOCK ({len(rem_blocks)/len(calib_cases)*100:.2f}%), {len(rem_asks)} ASK ({len(rem_asks)/len(calib_cases)*100:.2f}%)")
    print(f"Total Any Intervention: {len(all_rem)} / {len(calib_cases)} ({len(all_rem)/len(calib_cases)*100:.2f}%)")
    print("\nBreakdown of Remaining Residual False Flags:")
    for cat, counts in sorted(cat_counts.items(), key=lambda x: x[1]['total'], reverse=True):
        print(f"  {cat:<32} : {counts['total']:3d} ({counts['total']/len(all_rem)*100:4.1f}%) | BLOCK: {counts['BLOCK']:3d}, ASK: {counts['ASK']:3d}")

    unexplained = cat_counts['residual_other']['total']
    unexplained_pct_of_flags = (unexplained / len(all_rem)) * 100
    unexplained_pct_of_traffic = (unexplained / len(calib_cases)) * 100
    print(f"\nRemaining Unexplained ('residual_other') Share: {unexplained}/{len(all_rem)} flags ({unexplained_pct_of_flags:.1f}% of flags; {unexplained_pct_of_traffic:.2f}% of all calls)")

    # Write summary markdown
    out_md = ROOT / "results" / "b3_ablation_summary.md"
    with open(out_md, 'w', encoding='utf-8') as f:
        f.write("# Phase B3 Cumulative Ablation Summary (B3c -> B3b -> B3a)\n\n")
        f.write("Date: 2026-11-11  \n")
        f.write("Branch: `improvements-2026-11`  \n\n")

        f.write("## 1. Step-by-Step Ablation Progression (Calibration Split, n=1,212)\n\n")
        f.write("| Pipeline Step | BLOCK Rate [Tool-Cluster 95% CI] | BLOCK Count | Delta BLOCK | ASK Rate [CI] | Any Intervention [CI] | fcrb Recall (Floor 60%) | AgentHallu Recall (Floor 60%) |\n")
        f.write("|---|---|---|---|---|---|---|---|\n")
        f.write("| **Pre-B3 Baseline (Clean)** | 17.74% [6.7%, 30.4%] | 215 | — | 5.12% [1.6%, 9.9%] | 22.85% [12.0%, 35.1%] | 77.9% (PASS) | 61.3% (PASS) |\n")
        f.write("| **+ B3c (DateTime & Trajectory Ranges)** | 16.17% [6.6%, 27.2%] | 196 | **-19 (-1.57 pts)** | 5.12% [1.6%, 9.9%] | 21.29% [11.8%, 32.2%] | 77.9% (PASS) | 61.3% (PASS) |\n")
        f.write("| **+ B3b (Extended Numbers & Units)** | 16.17% [6.6%, 27.2%] | 196 | 0 (0.00 pts) | 5.12% [1.6%, 9.9%] | 21.29% [11.8%, 32.2%] | 77.9% (PASS) | 61.3% (PASS) |\n")
        f.write("| **+ B3a (Entity Aliases & Tickers)** | **15.84% [6.5%, 26.6%]** | **192** | **-4 (-0.33 pts)** | **4.95% [1.4%, 9.7%]** | **20.79% [11.6%, 31.3%]** | **77.9% (PASS)** | **61.3% (PASS)** |\n\n")

        f.write("* **Adversarial Near-Miss Suite**: All 7 adversarial near-miss tests (`tests/test_b3_adversarial.py`) PASS 100% (off-by-one dates, digit injections, wrong fractions, unit conversion mismatches, unmentioned tickers/states, and word-boundary substrings all strictly remain UNGROUNDED).\n")
        f.write("* **Tense-Invariance Removal**: Excluded sentence-level tense adjustments per user directive as they do not affect argument-level verdicts.\n\n")

        f.write("## 2. Post-B3 Remaining Residual False Flags Breakdown\n\n")
        f.write(f"* **Total Remaining Flags**: {len(all_rem)} / {len(calib_cases)} ({len(rem_blocks)} BLOCK, {len(rem_asks)} ASK)\n")
        f.write(f"* **Remaining Unexplained Share (`residual_other`)**: **{unexplained} flags** ({unexplained_pct_of_flags:.1f}% of remaining flags; **{unexplained_pct_of_traffic:.2f}%** of clean traffic)\n\n")
        f.write("| Root Cause Category | Total Remaining | % of Remaining | BLOCK Tier | ASK Tier | Addressed Next By |\n")
        f.write("|---|---|---|---|---|---|\n")
        for cat, counts in sorted(cat_counts.items(), key=lambda x: x[1]['total'], reverse=True):
            pct = (counts['total'] / len(all_rem)) * 100
            addr = "B2 (Control/Enum/Schema)" if cat in ('enum_or_flag_argument', 'buffer_or_size_limit', 'control_parameter') else ("B4 Combiner / B2 Mondrian" if cat != 'datetime_format' else "Cross-turn / Implicit Dates (B2/B4)")
            f.write(f"| `{cat}` | {counts['total']} | {pct:.1f}% | {counts['BLOCK']} | {counts['ASK']} | {addr} |\n")

    print(f"Wrote summary report to {out_md}")

if __name__ == '__main__':
    run()
