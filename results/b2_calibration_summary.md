# Phase B2 Summary: Schema Validation, Host Clock Evidence, and Call-Level Mondrian C1 Calibration

Date: 2026-11-11  
Branch: `improvements-2026-11`  

## 1. Step-by-Step Ablation Progression (MCP-Atlas Calibration Split, n=1,212)

| Step | Calibration `BLOCK` [Tool-Cluster 95% CI] | `ASK` [CI] | Any Intervention [CI] | `fcrb` Fabricated (`n=362`) BLOCK / Union | `AgentHallu` Err (`n=31`) / Clean `BLOCK` (`n=93`) |
|---|---|---|---|---|---|
| **Post-B3 Baseline** | 15.84% `[6.5%, 26.6%]` (192) | 4.95% `[1.4%, 9.7%]` (60) | 20.79% `[11.6%, 31.3%]` (252) | 49.7% / 77.3% | 61.3% / 11.8% (11) |
| **+ B2a (Schema Validation -> ALLOW)** | **13.12%** `[5.7%, 21.7%]` (159) | **3.38%** `[1.1%, 6.2%]` (41) | **16.50%** `[9.3%, 24.5%]` (200) | 48.9% / 76.5% | 61.3% / 11.8% (11) |
| **+ B2b (Host DateTime + ISO/Split Dates)** | **12.29%** `[5.2%, 20.4%]` (149) | **3.14%** `[1.1%, 5.7%]` (38) | **15.43%** `[8.6%, 23.1%]` (187) | 48.9% / 76.5% | 61.3% / 11.8% (11) |
| **+ B2c (OOF Mondrian C1, $\alpha=0.05$)** | **1.82%** `[1.0%, 2.8%]` (22) | 13.61% `[11.3%, 15.7%]` | 15.43% `[13.3%, 17.6%]` | 0.0% / 76.5% | — |
| **+ B2c (OOF Mondrian C1, $\alpha=0.10$)** | **3.14%** `[2.0%, 4.5%]` (38) | **12.29%** `[10.3%, 14.1%]` | **15.43%** `[13.3%, 17.6%]` | **32.3% / 76.5%** | **61.3% / 9.7% (9)** |

### Key Takeaways by Sub-Step
1. **B2a (Schema Validation -> `ALLOW`)**:
   - Validating `enum`, `csv_enum`, integer bounds (`minimum`/`maximum`), and patterns against `server/data/tool_schemas.json` (`interval`, `channel_types`, `max_length`, `mode`, `output_format`, `pages`, `maxResults`, `overallStatus`, etc.) while blocking explicit enum contradictions and out-of-bounds values cut **52 false flags** (-33 `BLOCK`, -19 `ASK`), dropping hard `BLOCK` from `15.84% -> 13.12%` and Any-Intervention from `20.79% -> 16.50%`.
   - Safety preserved: `fcrb` fabricated Union recall = **76.5%** (+16.5 pts above 60% floor).
2. **B2b (Host Current Date/Time Evidence & ISO Extensions)**:
   - Adding `("host", ...)` current clock evidence for relative phrases (`today`, `yesterday`, `tomorrow`, `this week`, `this month`), supporting `.000Z` / lowercase `z` timestamps, and supporting split month+year & month+day references (`"July 2025 ... July 15th"`) while preserving strict off-by-one adversarial blocking cut an additional **13 false flags** (-10 `BLOCK`, -3 `ASK`), reaching **12.29%** `BLOCK` / **15.43%** `Any Intervention` with **zero recall change** (`76.5%` fcrb, `61.3%` AgentHallu).
3. **B2c (5-Fold Tool-Grouped Call-Level Mondrian Conformal Calibration with C1 Clopper-Pearson UCB)**:
   - Deterministic call groups (`control_only`, `datetime`, `entity_or_id`, `search_query`):
     - `search_query` (`n=196`): 0.5% zero scores (`UCB=2.0%`), feasible at $\alpha=0.05$ with $\tau=0.5758$.
     - `entity_or_id` (`n=676`): 4.3% zero scores (`UCB=5.5%`), feasible at $\alpha=0.08..0.10$ with $\tau=0.8158$.
     - `control_only` (`n=130`): 6.2% zero scores (`UCB=9.8%`), feasible at $\alpha=0.10$ with $\tau=0.5526$.
     - `datetime` (`n=210`): **46.7% (`98/210`)** score `0.0` (`UCB=51.3%`) due to world-knowledge year lookups and implicit trajectory dates; C1 UCB sets $\tau=0.0000$, routing ungrounded dates to `ASK` rather than hard `BLOCK`.
   - At $\alpha = 0.10$ out-of-fold: achieves **3.14% `[2.0%, 4.5%]`** hard `BLOCK` rate (meeting `<= 5.0%` target), **12.29% `ASK`**, **15.43% Any Intervention**, **76.5%** `fcrb` Union recall (32.3% `BLOCK`-only), and **61.3%** AgentHallu detection (9.7% clean `BLOCK`).

## 2. Target Assessment & Logged Deviation Proposal

* **Stated Goal Maintained**: Keep **Hard `BLOCK` False-Block Rate $\le 5.0\%$** as the official goal on the frozen held-out split (`<= 5%` achieved out-of-fold at `3.14%` under Mondrian C1 $\alpha=0.10$).
* **Irreducible Pre-B4 Residual (> 8% Notice)**:
  * Prior to Mondrian `BLOCK -> ASK` routing, the deterministic uncalibrated `BLOCK` rate after B2a+B2b stands at **12.29%** (`> 8%`) and Total `Any-Intervention` (`BLOCK + ASK`) stands at **15.43%**.
  * **Root Causes**:
    1. *World-knowledge entity/year references (`46.7%` of `datetime` group)*: Prompts reference events via historical trivia rather than explicit dates, yielding exact score `0.0` before search retrieval.
    2. *URL construction / query reformulation*: Non-verbatim domain arguments score `0.0` on deterministic token/fuzzy matching.
    3. *Threshold invariance on discrete `0.0` mass*: Conformal score thresholds alone cannot convert `0.0`-scoring legitimate calls to `ALLOW` without disabling blocking; they only trade `BLOCK` for `ASK`.
* **Proposed Revised Operational Targets & Phase B4 Objective (Deviation Logged)**:
  1. Keep **`BLOCK <= 5.0%`** as the primary hard-block goal (with Mondrian C1 achieving `3.14%` OOF).
  2. Acknowledge the deterministic pre-ML **`Any-Intervention` floor of `15.4%`** after B2 and set a realistic joint Phase B4 target for the learned cross-fitted combiner: **`BLOCK <= 5.0%`**, **`ASK <= 8.0%`** (total `Any-Intervention <= 12.0%`), **`BLOCK`-only fabricated recall $\ge 45\%$**, and **Union recall $\ge 60\%$**.
