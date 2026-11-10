# Experiments Ledger (2026-11 Improvement & Hardening)

This ledger records every experiment, benchmark run, calibration, and ablation conducted during the November 2026 hardening sprint on branch `improvements-2026-11`.

---

## Phase A: Baseline Verification & Protocol Rigor

### Baseline Verification (2026-11-10)
- **Fast Test Suite**: 143 passed, 27 deselected (fast suite green in 54s).
- **Headline 1 (AgentHallu BFCL Provenance)**:
  - P3_E1: Detection = 61.3% (19/31), False-Block Rate = 17.2% (16/93).
  - P0 Context: Detection = 58.1% (18/31), False-Block Rate = 12.9% (12/93).
  - Verified from `benchmarks/provenance_holdout.json`.
- **Headline 2 (MCP-Atlas Clean Traffic False Blocks)**:
  - User-only (P0): 78.0% false-block rate.
  - User + Tool Outputs (P1): 29.4% false-block rate.
  - Typed Rules (P3): 19.5% false-block rate.
- **Headline 3 (Tool-Definition Ablation Clean 2x2)**:
  - Nemotron 550B: +9.3 points gain on Youden J (p = 0.24, not statistically significant).
  - Driven almost entirely by 1 "City, State" formatting heuristic (12/14 cases).
  - Nemotron 4B: +10.8 points (not statistically significant).

---

### Task A1: Reconcile Claims & Module Limitations (Commit `6a2f2fa`)
- **Scope**: Documentation audit across `README.md`, `STATUS.md`, `review/INFORMATION_ABLATION_FINDINGS.md`, `review/PAPER_PLAN_V2.md`, `review/EXPERIMENTS_2026-10-30.md`, and `review/PROJECT_BRIEF.md`.
- **Actions**:
  - Replaced all "+16 to 20 points" claims with verified numbers (+9.3 points, p=0.24).
  - Added dedicated "Claims We Do NOT Make" section.
  - Reconciled limitations: spelled-out numbers supported up to 99 in `arg_ground.py` but absent in `value_check.py`.

---

### Task A2: Statistical Rigor Module (Commit `6ef1604`)
- **Script**: `benchmarks/stats.py` with test suite `tests/test_stats.py`.
- **Capabilities**:
  - Exact Wilson score intervals for binomial proportions.
  - Stratified bootstrap confidence intervals (>= 10,000 resamples).
  - Fast vectorized AUROC computation and bootstrap AUROC CIs.
  - Exact McNemar tests for paired binary classification discordance.
  - Small sample size guardrail: auto-flags `[EXPLORATORY]` when $n_{pos} < 100$.
- **Wired Into**: `benchmarks/eval_provenance.py`, `benchmarks/analyze_agenthallu.py`, `benchmarks/eval_argground.py`, `benchmarks/mcpevol_runner.py`.

---

### Task A3: Matched False-Block Rate Evaluation (Commit `c25befd`)
- **Script**: `benchmarks/matched_fpr.py` -> `benchmarks/matched_fpr_results.json`.
- **Findings**:
  - Evaluated continuous checker against binary LLM judges (Claude Opus, Sonnet, Haiku, Nemotron 550B, 4B, gpt-oss-120b).
  - On natural errors (DeRef fcrb, n=200):
    - At matched ~10% FBR: Claude Sonnet detects **53.8%** [43.8, 63.5] vs Checker's **22.6%** [15.1, 31.8] ($p < 0.0001$, exact McNemar discordant 36 vs 3).
    - At 5% FBR: Checker catches only **11.3%** [6.0, 19.3] of natural errors.
  - Demonstrates that deterministic context grounding cannot stand alone as a drop-in replacement for semantic LLM judges on natural selection errors.

---

### Task A4: Error Taxonomy Breakdown with Independent Labels (Commit `cc862e7`, updated for label independence)
- **Script**: `benchmarks/error_taxonomy_eval.py` -> `results/error_type_table.md`, `review/audit3/error_taxonomy_audit.json`.
- **Methodological Fix**: Error taxonomy labels were decoupled entirely from `check_arguments` / `call_score`. DeRef fcrb labels are assigned 100% from DeRef ground-truth metadata (`args_meta` paths `no_source`, `format`, `visible`); AgentHallu errors are labeled by independent ground-truth inspection.
- **Results on fcrb (seeded n=200 dev sample)**:
  - `fabricated_value` (n=49): Context Checker = **75.5%** [61.9, 85.4] (37/49) vs Claude Sonnet = **63.3%** [49.3, 75.3] (31/49).
  - `wrong_selection` (n=43): Context Checker = **25.6%** [14.9, 40.2] (11/43) vs Claude Sonnet = **41.9%** [28.4, 56.7] (18/43).
  - `wrong_format` (n=5): Context Checker = **80.0%** [37.5, 96.4] (4/5) vs Claude Sonnet = **40.0%** [11.8, 76.9] (2/5).

---

### Task A5: Demote Circular Benchmarks (Commit `0544156`)
- **Action**: Demoted ArgGround-Bench and `mcpevol_v2/v3` from headline performance tables in `README.md`.
- **Circularity Note Added**: Explains that synthetic injected absent-value errors match the exact heuristic tested by the checker, artificially flattering the checker. Kept strictly as unit-level regression tests.

---

### Task A6: Finish Label Audit & Cleaned AUROC (Commit `d62abff`)
- **Script**: `benchmarks/finalize_label_audit.py` -> `results/label_audit_report.md`, `review/audit2/label_audit_summary.json`.
- **Dataset Cleaned Files**:
  - `benchmarks/data/bfcl/cases_bfcl_cleaned.json`
  - `benchmarks/data/deref/cases_fcrb_cleaned.json`
- **Audit Findings**:
  - Multi-annotator AI consensus (6 Claude runs: Opus, Sonnet, Haiku x 2 directions) + rule adjudication on 98 sampled items.
  - In BFCL, **64.2% (43/67)** of benchmark "wrong" calls were actually harmless format variants (e.g. `5:00 PM` vs `17:00`) or acceptable alternatives.
- **Dual AUROC Headline Reporting**:
  - **Raw benchmark labels**: Consensus AUROC = **0.843** [0.756, 0.904] (dev 56 cases, n=491) / **0.889** [0.825, 0.941] (200 cases) / **0.930** [0.858, 0.979] (all 19,443 argument instances).
  - **Cleaned high-confidence subset**: Consensus AUROC = **0.947** [0.915, 0.974] (dev 56 cases, n=448, n_wrong=24).
  - Both figures reported to distinguish genuine hallucination detection from format-convention mismatches.

---

## Phase B: Cutting False Blocks (Registered Protocol)

### Experimental Protocol & Commitments:
1. **Tool-Grouped Split on MCP-Atlas**:
   - Split clean MCP-Atlas traffic strictly by tool (grouped split), NEVER by row or call, into calibration/dev and held-out test.
   - Freeze the held-out split immediately and touch it strictly ONCE at the end of Phase B.
2. **Threshold Setting Rule**:
   - Define all ALLOW / ASK / BLOCK thresholds strictly on calibration data.
   - Target metrics reported with Wilson / Bootstrap 95% CIs:
     - BLOCK-tier false-block rate (target: <= 5%)
     - ASK rate on correct calls
     - Detection recall on fabricated values (floor: >= 60%)
3. **Step-by-Step Ablation Ladder**:
   - B1: Three-outcome decision (`ALLOW | ASK | BLOCK`) in proxy and verifier.
   - B2: Mondrian conformal calibration grouped by argument class.
   - B3: Entity alias table (`server/data/aliases.json`), extended number parsing, tense-invariant grounding.
   - B4: Learned combiner (logistic regression vs GBDT) with grouped cross-fitting by tool.
   - Evaluate each component independently, then cumulatively, logging delta FBR and delta recall.
4. **Safety & Trade-Off Floor**:
   - False-block reductions must not compromise detection recall on AgentHallu and fabricated values below 60%.
   - Full trade-off curve logged if tension occurs.

### MCP-Atlas Tool-Grouped Split (Frozen for Phase B)
- **Date**: 2026-11-10
- **Script**: enchmarks/split_mcpatlas_by_tool.py
- **Output File**: enchmarks/data/mcpatlas/mcpatlas_tool_split.json
- **Split Strategy**: Grouped strictly by tool (call['name']), zero tool overlap.
- **Calibration/Dev**: 53 tools, 1,212 calls (50.0%).
- **Frozen Held-out**: 53 tools, 1,211 calls (50.0%).
- **Held-out SHA256**: 64848214a38714bec0f4bf2aeda56b29f064b866332e5013ffcb7b60c1c06106
- **Rule**: Calibration and threshold tuning strictly on Calibration/Dev. Frozen held-out will be evaluated ONCE at the end of Phase B.
