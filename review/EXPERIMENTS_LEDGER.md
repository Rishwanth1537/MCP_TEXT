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

---

### Task B1: Three-Outcome Decision (ALLOW | ASK | BLOCK)
- **Status**: Completed and verified.
- **Code Changes**:
  - `server/models/schemas.py`: Added `Decision` (`ALLOW`, `ASK`, `BLOCK`) and `BlockReason` enums; updated `ArgumentGroundingRequest` and `ArgumentGroundingResponse` with `decision`, `block_reason`, `tau_block=0.50`, `tau_ask=0.88`, and `decision_mode="three_tier"`.
  - `server/tools/verify_arguments.py`: Implemented three-tier decision routing. High-confidence ungrounded arguments trigger `BLOCK`; low-confidence control parameters (`limit`, `offset`, `sort`, `filter`, `query`) and medium-confidence scores trigger `ASK`.
  - `server/proxy/interceptor.py`: Updated `InterceptingToolProxy` to route `ALLOW` to execution, `ASK` to human confirmation (`requires_confirmation=True`), and `BLOCK` to hard halt. Backward compatibility preserved via `decision_mode="binary"`.
  - Tests added: 4 new unit tests in `tests/test_proxy.py` and 4 in `tests/test_arg_ground.py` (all passing).
- **Evaluation on Calibration Data (`benchmarks/eval_phase_b.py`)**:
  - **MCP-Atlas Calibration Clean Traffic (n=1,212 calls across 53 tools)**:
    - Baseline Binary (P3 tau=0.88): Hard False-Block Rate = **20.2%** [18.1, 22.6]
    - B1 Three-Tier:
      - Hard False Blocks (`BLOCK` tier): **15.1%** [13.2, 17.2] (-5.1 points)
      - Confirmation Rate (`ASK` tier): **5.1%** [4.0, 6.5]
      - Pass Rate (`ALLOW` tier): **79.8%** [77.4, 82.0]
  - **Fabricated Value Recall (DeRef fcrb, n=362)**:
    - Combined Recall (`BLOCK + ASK`): **77.9%** [73.4, 81.9] (282/362) -> Zero recall lost, well above 60% floor.
  - **AgentHallu (n=31 errors, n=93 clean calls)**:
    - Detection (`BLOCK + ASK`): **61.3%** [43.8, 76.3] `[EXPLORATORY]` -> Exact detection preserved.
    - Clean Hard Blocks: **14.0%** [8.3, 22.5] (down from 17.2%).
    - Clean ASK Rate: **3.2%** [1.1, 9.1]%.

---

### Task B1b: Residual False-Block Analysis & Cluster Rigor (2026-11-10)
- **Status**: Completed and recorded.
- **MCP-Atlas Calibration Clean False Blocks (B1 Three-Tier, n=1,212 calls across 53 tools)**:
  - Hard False-Block Rate (`BLOCK` tier): **15.1%**
    - **Primary (Cluster-Bootstrap by Tool, n=53 tools)**: **[6.5%, 25.9%]**
    - Secondary (Cluster-Bootstrap by Task, n=360 tasks): **[12.7%, 17.5%]**
    - Secondary (Exact Wilson): **[13.2%, 17.2%]**
  - Confirmation Rate (`ASK` tier): **5.1%** (Primary Tool Cluster-Bootstrap: [1.6%, 9.9%], Wilson: [4.0%, 6.5%])
  - Any Intervention Rate on Clean Calls (`BLOCK + ASK`): **20.2%** [18.0%, 22.6%]
  - Out-of-Fold 5-Fold Tool-Grouped Cross-Fitting BLOCK Rate: **15.1%** [6.5%, 25.9%] (n=1,212)
  - Largest Tool Share:
    - Calibration split: `airtable_search_records` = 97/1,212 (8.0%)
    - Frozen held-out split: `cli-mcp-server_run_command` = 95/1,211 (7.8%)

- **Root Cause Breakdown (100 False Flags Stratified across 30 Tools)**:
  - `datetime_format`: 24% (BLOCK: 21, ASK: 3) -> Projected removal: **~4.23 percentage points**
  - `number_unit_format`: 17% (BLOCK: 16, ASK: 1) -> Projected removal: **~3.22 percentage points**
  - `control_parameter`: 13% (BLOCK: 3, ASK: 10) -> Projected removal: **~0.60 percentage points**
  - `alias_entity_naming`: 6% (BLOCK: 4, ASK: 2) -> Projected removal: **~0.81 percentage points**
  - `world_knowledge`: 3% (BLOCK: 2, ASK: 1) -> Projected removal: **~0.40 percentage points**
  - `other` (residual entity mismatches, complex params): 37% (BLOCK: 29, ASK: 8)
  - **Key Finding**: Surface formatting and alias mismatches (B3 scope) account for **47%** of all false flags (~8.26 points of potential BLOCK-tier reduction).

- **Tiered Recall Breakdown**:
  - **DeRef fcrb Fabricated Values (n=362)**:
    - BLOCK-only: **49.7%** [44.6%, 54.9%] (180/362)
    - ASK-only: **28.2%** [23.8%, 33.0%] (102/362)
    - Union (`BLOCK + ASK`): **77.9%** [73.4%, 81.9%] (282/362)
  - **AgentHallu Errors (n=31)**:
    - Union Detection: **61.3%** [43.8%, 76.3%] `[EXPLORATORY]` (19/31)
  - **AgentHallu Clean Calls (n=93)**:
    - Hard False Blocks: **14.0%** [8.3%, 22.5%] (13/93)
    - Asks: **3.2%** [1.1%, 9.1%] (3/93)
    - Any Intervention Rate: **17.2%** [10.9%, 26.1%] (16/93)

- **Redone A4 Matched-FPR Comparison on `wrong_selection` (fcrb seeded n=200)**:
  - N = 200: 94 clean control calls, 106 errors (49 fabricated, 43 wrong selection, 5 wrong format, 9 other).
  - Clean Call FBR: Checker = 10.6% [5.9%, 18.5%], Claude Sonnet = 9.6% [5.1%, 17.2%].
  - Catch Rate on `wrong_selection` (n=43): Checker = **25.6%** [14.9%, 40.2%] (11/43) vs Claude Sonnet = **41.9%** [28.4%, 56.7%] (18/43).
  - Exact McNemar Test: 10 discordant favoring Sonnet vs 3 favoring Checker ($p = 0.0923$, not statistically significant at n=43).

- **A6 Headline Realignment & 50-Item Spot-Check Plan**:
  - **Headline Metric**: Full-data raw consensus AUROC = **0.889** [0.825, 0.941] (200 cases) / **0.930** [0.858, 0.979] (all 19,443 instances) / **0.843** [0.756, 0.904] (dev 56 cases).
  - **Labelled Sensitivity Analysis**: Cleaned high-confidence subset AUROC = **0.947** [0.915, 0.974] (dev 56 cases, n=448, n_wrong=24).
  - **50-Item Human Spot-Check Plan**: 50 items stratified from `review/audit2/` (25 where AI consensus agreed with gold, 25 where AI consensus disagreed with gold) will be reviewed by a single blinded human adjudicator without seeing model identities to establish inter-rater reliability (Cohen's kappa).

## Experiment 11: Phase B - Before B3 Audit & Protocol Hardening
- **Date**: 2026-11-10
- **Branch**: improvements-2026-11
- **Scope**: Rigorous audit of all 245 calibration false flags, micro/macro per-tool rates, sample size lineage, recall trade-off curves, blinded human audit file, and success criterion definition.

- **Breakdown of All 245 Calibration False Flags (183 BLOCK, 62 ASK)**:
  - datetime_format: 32.7% (80 flags: 76 BLOCK, 4 ASK)
  - enum_or_flag_argument: 27.3% (67 flags: 37 BLOCK, 30 ASK) — former 'other' decomposed
  - 
umber_unit_format: 9.0% (22 flags: 22 BLOCK, 0 ASK)
  - 
esidual_other: 8.6% (21 flags: 15 BLOCK, 6 ASK) — down from 37% to 8.6%
  - search_query_reformulation: 6.5% (16 flags: 2 BLOCK, 14 ASK)
  - uffer_or_size_limit: 5.3% (13 flags: 13 BLOCK, 0 ASK)
  - world_knowledge: 2.9% (7 flags: 7 BLOCK, 0 ASK)
  - control_parameter: 2.9% (7 flags: 1 BLOCK, 6 ASK)
  - url_or_endpoint_construction: 2.4% (6 flags: 6 BLOCK, 0 ASK)
  - lias_entity_naming: 2.0% (5 flags: 3 BLOCK, 2 ASK)
  - intermediate_step_dependency: 0.4% (1 flag: 1 BLOCK, 0 ASK)

- **20-Item Clean Call Benchmark Spot-Check**:
  - Sampled 10 BLOCK and 10 ASK across 6 tool families.
  - Finding: 0 / 20 (0.0%) mislabeled clean calls. All 20 were valid tool executions following user intent where strict string matching failed.

- **Out-of-Fold vs In-Sample Equivalence Clarification**:
  - Out-of-fold BLOCK rate equals in-sample (15.1% [6.5, 25.9]) because B1 routing used fixed, deterministic threshold rules (tau_block=0.50, tau_ask=0.15, control param schema checks).
  - Explicit confirmation: No mathematical parameter was fitted. Fixed rule thresholding will not be described as 'cross-fitted' until Phase B4/B2.

- **Sample Size (N) Lineage**:
  - MCP-Atlas: Calibration n=1,212 (53 tools); Held-Out n=1,211 (53 tools, frozen SHA256).
  - DeRef fcrb: 362 fabricated leaves; Dev eval n=200 calls (94 clean, 49 fabricated, 43 wrong_selection, 5 format, 9 other).
  - AgentHallu: 31 trajectory error steps; 93 clean control steps.
  - BFCL v3: 56 dev cases (448 args); 200 audit cases; 19,443 corpus instances.

- **Recall vs Intervention Trade-off Curves & 60% Floor**:
  - fcrb Fabricated Values: At tau=0.50, recall is 77.9% (+17.9 pts above 60% floor) at 20.2% clean intervention.
  - AgentHallu: At tau=0.50, detection is 61.3% (+1.3 pts above 60% floor) at 17.2% clean intervention.
  - Relaxing tau beyond 0.50 drops AgentHallu below the 60% floor (51.6% at tau=0.65).

- **A6 Reference Mismatch Note & Blinded Audit**:
  - Note: Consensus predicts mismatch with benchmark reference values rather than absolute truth.
  - Generated blinded 50-item audit in 
eview/audit3/blinded_human_audit_50.json.

- **Success Criterion Confirmation**:
  - Official success criterion: BLOCK-tier false-block rate <= 5.0% as a point estimate on the frozen held-out split.
  - Tool-cluster bootstrap CI will be fully reported for uncertainty quantification, but point estimate <= 5.0% serves as the pass/fail bar.
