# Phase 1 & Phase 2 Results Draft (Branch `improvements-2026-11`)

**Date:** 2026-11-12  
**Status:** Development-half & Exploratory (`n_pos < 100` on `AgentHallu` and `fcrb_val100_d1`; locked held-out splits remain unopened for final evaluation).

---

## 1. Phase 1: Guard Freeze, 550B Verification & Split Hygiene

### 1.1 Frozen Guard Configuration
The deterministic 3-tier guard (`ALLOW` / `ASK` / `BLOCK`) was frozen (`3e1df43`) with no further rule changes:
- `server/pipeline/arg_ground.py` (SHA-256: `d8d66f1dc16c9c0a54219c1699972661e3fea2c3e6f5a5650c36bdf98165bff2`)
- `server/pipeline/provenance.py` (SHA-256: `98218e845d204b57553efdb1f8ea3ca2753623a2df8e87dda78eee9f6cfbb2a7`)
- `server/pipeline/schema_validator.py` (SHA-256: `a551122906b0b9c4a677d30f0679d65559738e3501a9d1559fc3ba83ec9ed494`)
- `server/tools/verify_arguments.py` (SHA-256: `0e4784e8a2cc0733de568fd627ae6b3a925f9b447f0902b301864bacd4850f5c`)
- **Fast unit test suite**: `175 passed, 27 deselected in 48.10s` (`pytest tests/ -m "not integration"`).

On MCP-Atlas calibration (`n = 1,212`, `53` tools), the frozen guard achieves:
- **Uncalibrated (`B2a+B2b`)**: `12.29%` `BLOCK`, `3.14%` `ASK`, `15.43%` Any-Intervention.
- **Mondrian C1 calibrated (`B2c`, $\alpha = 0.10$, 5-fold tool-grouped OOF)**: **`3.14%` `BLOCK`** `[2.0%, 4.5%]`, **`12.29%` `ASK`** `[10.3%, 14.1%]`, **`15.43%` Any-Intervention**. (Conformal calibration shifts ungrounded `datetime` calls — `46.7%` zero scores — from `BLOCK` into `ASK`, rather than resolving them.)

### 1.2 Verified Nemotron 550B Tool-Definition Ablation (`benchmarks/info2x2_summary.json`)
Evaluated on `n = 168` `fcrb` dev calls carrying tool definitions (`82` correct, `86` wrong), `temperature = 0` (`E_placebo` marked invalid and excluded):

| Condition | FBR [95% Wilson] | Detection [95% Wilson] | Youden $J$ | Format-Conv. (`n=14`) | Non-Format (`n=72`) | Unpars. |
|---|---|---|---|---|---|---|
| `A_plain` | `29.3%` `[20.5, 39.9]` | `57.0%` `[46.4, 66.9]` | `+27.7%` | `0 / 14` | `49 / 72` | `1` |
| `B_rule` | `43.9%` `[33.7, 54.7]` | `62.8%` `[52.2, 72.3]` | `+18.9%` | `1 / 14` | `53 / 72` | `0` |
| `C_def` | `29.3%` `[20.5, 39.9]` | `66.3%` `[55.8, 75.4]` | `+37.0%` | `5 / 14` | `52 / 72` | `0` |
| `D_defrule` | `32.9%` `[23.7, 43.7]` | `77.9%` `[68.1, 85.4]` | `+45.0%` | `12 / 14` | `55 / 72` | `0` |

- **Key negative/null finding**: Giving `550B` the tool definition alone (`C_def` vs `A_plain`) yields `+9.3` pts Youden $J$ with **exact McNemar `p = 0.243` (`22` vs `14`, not statistically significant)**. Most of the `D_defrule` gain (`77.9%`) is driven by one formatting rule (`City, State`: `0/14 -> 12/14`), while adding the rule without the definition (`B_rule`) spikes false blocks to `43.9%` (`-8.8` pts $J$).

### 1.3 Split Hygiene & Contamination Warnings
- **`fcrb_tool_split.json`**: `dev` = `89` tools (`856` cases: `dev_train` `49` tools / `432` cases; `dev_val` `40` tools / `424` cases); `frozen_heldout` = `153` tools (`404` cases, `sha256: 21887c2a...`). Because `132` of `404` held-out cases (`56` tools) appeared in the checker's earlier `630`-case `split == "dev"` set, the held-out set is **"judge-fresh, checker-partly-seen"**, and **`heldout_fresh_272` (`n = 272`, `sha256: 2bea694c...`)** is the headline set for judge-vs-checker comparisons.
- **`mcpatlas_tool_split.json`**: `53` dev tools (`1,212` cases) / `53` held-out tools (`1,211` cases). **Overlap warning**: the 9 Nov Deviation 21 provenance held-out run (`19.5%` block rate on `265` tasks / `1,113` calls by `sha256(TASK)` parity) overlaps this locked held-out split; it is **not fully unseen** for provenance rules.
- **Groq `openai/gpt-oss-120b` 30-case smoke test**: `28 / 30 (93.3%)` per-case agreement with cache (`0` unparseable; live `FBR 30.8% / det 58.8%` vs cache `23.1% / 64.7%`).

---

## 2. Phase 2 (Step D1): Guard-First + Judge Triage (Dev)

Step D1 evaluates routing `B2c` ($\alpha = 0.10$) guard decisions to a J1+J2 `cite_then_judge` LLM judge (`+def + source_labelled + checker_hints + autorules`, `temperature = 0`):
- **D1-A (`ASK`-only -> Judge)**: Guard `ALLOW` and `BLOCK` are final; only `ASK` calls go to the judge.
- **D1-B (`ASK + BLOCK` -> Judge)**: Guard `ALLOW` is final; all flagged calls go to the judge so it can rescue guard false blocks.

### 2.1 Quantitative Results (`results/judge_2026-11-12_d1/`)

| Dataset & Model | Guard `BLOCK` (FBR / Det) | Guard `Any` (FBR / Det) | D1-A (`ASK` -> Judge) | D1-B (`ASK+BLOCK` -> Judge) | Judge Calls (D1-A / D1-B) | Eff. Tokens/Call (D1-B) |
|---|---|---|---|---|---|---|
| **`AgentHallu`** (`n=124`, `[EXPLORATORY]`, `550B`, `frozen_pre_b3` hints) | **`9.7%` / `58.1%`** | `15.1%` / `61.3%` | `14.0%` / `61.3%` | `11.8%` / `54.8%` (`AUROC 0.541`) | `6` / `33` | `464` (`p50 12.8s`) |
| **`AgentHallu`** (`n=124`, `[EXPLORATORY]`, `gpt-oss-120b`, w/ hints) | `9.7%` / `58.1%` | `15.1%` / `61.3%` | `14.0%` / `61.3%` | `11.8%` / `58.1%` (`AUROC 0.563`) | `6` / `33` | `299` (`p50 1.0s`) |
| **`AgentHallu`** (`n=124`, `[EXPLORATORY]`, `gpt-oss-120b`, **no hints**) | `9.7%` / `58.1%` | `15.1%` / `61.3%` | `14.0%` / `61.3%` | **`10.8%` / `29.0%`** (`AUROC 0.489`) | `6` / `33` | `288` (`p50 1.1s`) |
| **`mcpatlas_dev_200`** (`n=200` clean, `550B`) | **`7.0%` / —** | `21.5%` / — | `19.0%` / — | `18.0%` / — (`36/200`) | `29` / `43` | `681` (`p50 17.6s`) |
| **`fcrb_val100_d1`** (`n=100` `excluding_dev200`, `[EXPLORATORY]`, `550B`, `[in-sample]` hints) | `19.6%` / `30.6%` | `76.5%` / `75.5%` | `64.7%` / `75.5%` | `58.8%` / `71.4%` (`AUROC 0.396`) | `51` / `76` | `1,470` (`p50 21.8s`) |

### 2.2 Honest Diagnosis of Phase 2 (Clean Negative Result)
1. **Checker hints anchor the judge**: Sending `checker_hints` (`UNGROUNDED`) causes `550B` and `gpt-oss-120b` to agree with the guard's false blocks rather than rescuing them (`18.0%` FBR on `mcpatlas_dev_200` vs `7.0%` guard `BLOCK` alone; `58.8%` FBR on `fcrb_val100_d1`).
2. **Removing hints collapses fabricated-value recall**: On `AgentHallu` without `checker_hints`, `gpt-oss-120b` D1-B detection drops from `58.1%` to **`29.0%`**, missing half of the invented values that the deterministic guard already caught.
3. **Guard `ALLOW` leaks `wrong_selection`**: On `fcrb_val100_d1`, `24.5%` of errors (`12/49`) were `ALLOW`ed by the guard because the wrong value appeared verbatim in the prompt, so escalating only `ASK`/`BLOCK` calls cannot catch them.

---

## 3. Open Items Before Workshop Paper
1. **98-Item Human Audit**: Both authors independently annotate `review/audit2/audit_sample_blind.csv` (1 hour each, separately) to replace AI-audited `fcrb`/`BFCL` labels.
2. **Final Locked Evaluation**: Run once on `fcrb` `heldout_fresh_272` (headline) / `404` (secondary), `mcpatlas` `heldout` (noting Deviation 21 overlap), and `AgentHallu`, reporting guard-alone, judge-alone, and D1 honestly.
