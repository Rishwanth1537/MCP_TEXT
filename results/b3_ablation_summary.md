# Phase B3 Cumulative Ablation Summary (B3c -> B3b -> B3a)

Date: 2026-11-11  
Branch: `improvements-2026-11`  

## 1. Step-by-Step Ablation Progression (Calibration Split, n=1,212)

| Pipeline Step | BLOCK Rate [Tool-Cluster 95% CI] | BLOCK Count | Delta BLOCK | ASK Rate [CI] | Any Intervention [CI] | fcrb Recall (Floor 60%) | AgentHallu Recall (Floor 60%) |
|---|---|---|---|---|---|---|---|
| **Pre-B3 Baseline (Clean)** | 17.74% [6.7%, 30.4%] | 215 | — | 5.12% [1.6%, 9.9%] | 22.85% [12.0%, 35.1%] | 77.9% (PASS) | 61.3% (PASS) |
| **+ B3c (DateTime & Trajectory Ranges)** | 16.17% [6.6%, 27.2%] | 196 | **-19 (-1.57 pts)** | 5.12% [1.6%, 9.9%] | 21.29% [11.8%, 32.2%] | 77.9% (PASS) | 61.3% (PASS) |
| **+ B3b (Extended Numbers & Units)** | 16.17% [6.6%, 27.2%] | 196 | 0 (0.00 pts) | 5.12% [1.6%, 9.9%] | 21.29% [11.8%, 32.2%] | 77.9% (PASS) | 61.3% (PASS) |
| **+ B3a (Entity Aliases & Tickers)** | **15.84% [6.5%, 26.6%]** | **192** | **-4 (-0.33 pts)** | **4.95% [1.4%, 9.7%]** | **20.79% [11.6%, 31.3%]** | **77.9% (PASS)** | **61.3% (PASS)** |

* **Adversarial Near-Miss Suite**: All 7 adversarial near-miss tests (`tests/test_b3_adversarial.py`) PASS 100% (off-by-one dates, digit injections, wrong fractions, unit conversion mismatches, unmentioned tickers/states, and word-boundary substrings all strictly remain UNGROUNDED).
* **Tense-Invariance Removal**: Excluded sentence-level tense adjustments per user directive as they do not affect argument-level verdicts.

## 2. Post-B3 Remaining Residual False Flags Breakdown

* **Total Remaining Flags**: 252 / 1212 (192 BLOCK, 60 ASK)
* **Remaining Unexplained Share (`residual_other`)**: **20 flags** (7.9% of remaining flags; **1.65%** of clean traffic)

| Root Cause Category | Total Remaining | % of Remaining | BLOCK Tier | ASK Tier | Addressed Next By |
|---|---|---|---|---|---|
| `datetime_format` | 93 | 36.9% | 89 | 4 | Cross-turn / Implicit Dates (B2/B4) |
| `enum_or_flag_argument` | 67 | 26.6% | 37 | 30 | B2 (Control/Enum/Schema) |
| `number_unit_format` | 22 | 8.7% | 22 | 0 | B4 Combiner / B2 Mondrian |
| `residual_other` | 20 | 7.9% | 14 | 6 | B4 Combiner / B2 Mondrian |
| `search_query_reformulation` | 17 | 6.7% | 2 | 15 | B4 Combiner / B2 Mondrian |
| `buffer_or_size_limit` | 13 | 5.2% | 13 | 0 | B2 (Control/Enum/Schema) |
| `world_knowledge` | 7 | 2.8% | 7 | 0 | B4 Combiner / B2 Mondrian |
| `url_or_endpoint_construction` | 6 | 2.4% | 6 | 0 | B4 Combiner / B2 Mondrian |
| `control_parameter` | 6 | 2.4% | 1 | 5 | B2 (Control/Enum/Schema) |
| `intermediate_step_dependency` | 1 | 0.4% | 1 | 0 | B4 Combiner / B2 Mondrian |
