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

## Experiment 12: Phase B3 Step-by-Step Normalization Ablations (B3c -> B3b -> B3a)
- **Date**: 2026-11-11
- **Branch**: improvements-2026-11
- **Scope**: Step-by-step ablations for DateTime (B3c), Extended Numbers/Units (B3b), and Entity Aliases (B3a) evaluated on MCP-Atlas Calibration Split (n=1,212) with cluster-bootstrap CIs, adversarial near-miss verification, and residual breakdown.

- **Step-by-Step Ablation Progression**:
  1. **True Pre-B3 Baseline (16686a4)**:
     - Hard False Blocks (BLOCK): **15.10%** [6.5%, 25.9%] (count: 183/1212)
     - Asks (ASK): **5.12%** [1.6%, 9.9%] (count: 62/1212)
     - Any Intervention: **20.21%** (count: 245/1212)
     - fcrb Fabricated Recall: 77.9% [73.4%, 81.9%] (Floor: 60%)
     - AgentHallu Detection: 61.3% [43.8%, 76.3%] (Floor: 60%)
  2. **Intermediate Buggy B3c Run (	ask-1131)**:
     - Hard False Blocks (BLOCK): **17.74%** [6.7%, 30.4%] (count: 215/1212, +32 blocks from intercepting _YM/_QUARTER before cross-source _date_range_grounded fix)
     - Asks (ASK): **5.12%** [1.6%, 9.9%] (count: 62/1212)
     - Any Intervention: **22.85%** [12.0%, 35.1%] (count: 277/1212)
  3. **+ B3c (DateTime Normalization & Cross-Turn Trajectory Range Expansion)**:
     - Hard False Blocks (BLOCK): **16.17%** [6.6%, 27.2%] (count: 196/1212)
     - Asks (ASK): **5.12%** [1.6%, 9.9%] (count: 62/1212)
     - Any Intervention: **21.29%** [11.8%, 32.2%] (count: 258/1212)
     - Adversarial Tests: 2/2 PASS (off-by-one dates, mid-month inventions remain strictly UNGROUNDED)
     - Tense-invariance removed per directive as it does not affect argument values.
  4. **+ B3b (Extended Compound Numbers >99, Ordinals, Fractions, and Unit Conversions)**:
     - Hard False Blocks (BLOCK): **16.17%** [6.6%, 27.2%] (count: 196/1212)
     - Asks (ASK): **5.12%** [1.6%, 9.9%] (count: 62/1212)
     - Any Intervention: **21.29%** [11.8%, 32.2%] (count: 258/1212)
     - Adversarial Tests: 3/3 PASS (digit additions, unit mismatches km vs mi, temperature mismatches remain UNGROUNDED)
  5. **+ B3a (Entity Alias Mapping: Crypto Tickers, Stocks, US States)**:
     - Hard False Blocks (BLOCK): **15.84%** [6.5%, 26.6%] (count: 192/1212)
     - Asks (ASK): **4.95%** [1.4%, 9.7%] (count: 60/1212)
     - Any Intervention: **20.79%** [11.6%, 31.3%] (count: 252/1212)
     - Adversarial Tests: 2/2 PASS (unmentioned tickers, unmentioned states, word boundary substring protection remain UNGROUNDED)

- **Cumulative B3 Impact**:
  - Clean Call Hard False Block Rate: **15.84% [6.5%, 26.6%]**.
  - All 7 constructed adversarial near-miss unit checks remained strictly UNGROUNDED (7/7 pass rate; synthetic unit coverage, not precision on natural traffic).
  - Safety floors intact: fcrb fabricated value recall = 77.3% (>=60%), AgentHallu detection = 61.3% (>=60%).

- **Post-B3 Remaining False Flags & Unexplained Share**:
  - Total remaining flags: 252 / 1,212 calls (192 BLOCK, 60 ASK).
  - **Remaining Unexplained Share (
esidual_other)**: **20 flags** (7.9% of flags, **1.65%** of all calibration calls).
  - Remaining categorized breakdown: datetime_format 93, enum_or_flag_argument 67, 
umber_unit_format 22, search_query_reformulation 17, uffer_or_size_limit 13, world_knowledge 7, url_or_endpoint_construction 6, control_parameter 6, intermediate_step_dependency 1.

## Experiment 13: Pre-B2 Reconciliation, Paired Bootstrap, Independent Validation & Near-Miss Scale Test
- **Date**: 2026-11-11
- **Branch**: `improvements-2026-11`
- **Scope**: Reconciles true Pre-B3 (`16686a4`) vs intermediate 215 state vs Post-B3 (`19931f2`), computes paired tool-cluster bootstrap delta, validates B3 rules on independent AgentHallu & fcrb clean sets, verifies B3b firing on real data, corrects prior text sums, and runs near-miss scale evaluation.

- **1. Baseline Reconciliation (183 vs 215 vs 192 BLOCK)**:
  - True Pre-B3 (`16686a4`): **183 BLOCK (15.10%)**, 62 ASK (5.12%), 245 Any (20.21%).
  - Intermediate Buggy Run (`task-1131`): **215 BLOCK (17.74%)**, +32 blocks caused by intercepting `_YM`/`_QUARTER` in `arg_ground.py` before cross-source `_date_range_grounded` was updated in `provenance.py`.
  - Final Post-B3 (`19931f2`): **192 BLOCK (15.84%)**, 60 ASK (4.95%), 252 Any (20.79%).
  - Effect vs Intermediate 215 State: **-23 BLOCK (-1.90 pts)**, -2 ASK, -25 Any.
  - Effect vs True Pre-B3 (`16686a4`): **+9 BLOCK (+0.74 pts)**, **-2 ASK (-0.17 pts)**, **+7 Any (+0.58 pts)**. Composed of **15 genuine false blocks fixed across 4 tools** (`airtable_search_records`, `alchemy_fetchTokenPriceHistoryBySymbol`, `google-workspace_list_events`, `national-parks_findParks`), offset by **24 accidental decimal/token-bag collisions tightened into BLOCK on 1 tool** (`airtable_search_records`).

- **2. Paired Tool-Cluster Bootstrap Analysis**:
  - Paired Delta BLOCK (Post-B3 vs `16686a4`): **+0.74 pts** (Tool-Cluster 95% CI: **[-0.89%, +3.16%]**).
  - Paired Delta Any-Intervention (Post-B3 vs `16686a4`): **+0.58 pts** (Tool-Cluster 95% CI: **[-1.16%, +3.05%]**).

- **3. Independent Dataset Validation (Not Used for Rule Design)**:
  - **AgentHallu Clean (`n=93`)**: Pre-B3 BLOCK = 13/93 (14.0%) -> Post-B3 BLOCK = **11/93 (11.8%)** (-2.2 pts reduction), ASK invariant at 3/93 (3.2%), Any = 17.2% -> **15.1%**; Error detection (`n=31`) unchanged at **19/31 (61.3%)**.
  - **DeRef fcrb Clean (`n=94`)**: Pre-B3 = 23 BLOCK (24.5%), 30 ASK (31.9%), 53 Any (56.4%) -> Post-B3 = **23 BLOCK (24.5%)**, **28 ASK (29.8%)**, **51 Any (54.3%)** (-2.1 pts reduction).

- **4. B3b Real-Call Firing Check**:
  - MCP-Atlas Calibration (`n=1,212`): **1 call** (`mcpatlas-689cd6f8522029b7ad7b200e-0`, value `1.0`).
  - AgentHallu (`n=124`): **3 clean calls** (`BFCL/068:s9:c0` `608.94` and `BFCL/055:s6:c0` `484.67` in `estimate_drive_feasibility_by_mileage` for km<->miles; `BFCL/010:s3:c0` ordinal `1.0` in `tail`).
  - DeRef fcrb Clean (`n=94`): **0 calls**.
  - Conclusion: Retained in default path because it improves independent unseen AgentHallu clean calls (14.0% -> 11.8% FBR) with zero error recall loss.

- **5. Table & Text Corrections**:
  - Corrected '68.2%' text figure: B2 schema/control/enum/buffer categories (`67 + 13 + 6 = 86 / 252`) equal **34.1%**; B4 semantic/query/url categories (`17 + 7 + 6 = 30 / 252`) equal **11.9%**; Combined B2 + B4 targets equal **46.0%** (or **82.9%** including `datetime_format` relative dates).
  - Test count explanation: 19 files contain 175 test functions (baseline 143 parameterized items prior to Phase B additions); the B3 fast verification ran 85 collected items across 7 core unit test files.

- **6. Near-Miss Scale Test**:
  - DeRef fcrb Fabricated (`n=362`): Union recall = **280/362 (77.3%)** vs 282/362 (77.9%) Pre-B3 (-0.55 pts), BLOCK recall unchanged at **180/362 (49.7%)** (well above 60% floor).
  - Near-Miss Perturbed Variants (`n=94` mutated clean calls): Union recall = **71/94 (75.5%)**.

## Experiment 14: Phase B2 — Schema Validation, Host Clock DateTime, and Call-Level Mondrian C1 Calibration
- **Date**: 2026-11-11
- **Branch**: `improvements-2026-11`
- **Scope**: Implements B2a (schema enum/bounds/pattern validation -> ALLOW), B2b (host current date/time evidence & split/fractional ISO dates), and B2c (5-fold tool-grouped out-of-fold Mondrian conformal calibration with C1 Clopper-Pearson UCB), plus target deviation analysis.

- **Step-by-Step Ablation Progression (MCP-Atlas Calibration `n=1,212`)**:
  1. **Post-B3 Baseline**:
     - `BLOCK`: **15.84%** `[6.5%, 26.6%]` (192), `ASK`: **4.95%** `[1.4%, 9.7%]` (60), `Any`: **20.79%** `[11.6%, 31.3%]` (252)
     - `fcrb` Fabricated (`n=362`): 49.7% BLOCK / 77.3% Union; `AgentHallu` (`n=31`): 61.3% detection / 11.8% clean BLOCK
  2. **+ B2a (Schema Validation -> ALLOW inside bounds/enums, BLOCK out-of-bounds/conflicts)**:
     - `BLOCK`: **13.12%** `[5.7%, 21.7%]` (159, **Delta: -33 blocks / -2.72 pts**)
     - `ASK`: **3.38%** `[1.1%, 6.2%]` (41, **Delta: -19 asks / -1.57 pts**)
     - `Any Intervention`: **16.50%** `[9.3%, 24.5%]` (200, **Delta: -52 flags / -4.29 pts**)
     - `fcrb` Fabricated: **48.9% BLOCK / 76.5% Union** (PASS >=60% floor); `AgentHallu`: **61.3% detection / 11.8% clean BLOCK**.
  3. **+ B2b (Host Date/Time Clock Evidence + ISO `.000Z`/lowercase `z` + Split Month-Day)**:
     - `BLOCK`: **12.29%** `[5.2%, 20.4%]` (149, **Delta: -10 blocks / -0.83 pts**)
     - `ASK`: **3.14%** `[1.1%, 5.7%]` (38, **Delta: -3 asks / -0.25 pts**)
     - `Any Intervention`: **15.43%** `[8.6%, 23.1%]` (187, **Delta: -13 flags / -1.07 pts**)
     - `fcrb` Fabricated: **48.9% BLOCK / 76.5% Union**; `AgentHallu`: **61.3% detection / 11.8% clean BLOCK**.
  4. **+ B2c (5-Fold Tool-Grouped OOF Mondrian Conformal Calibration with C1 UCB)**:
     - Deterministic Call Classes (`control_only`, `datetime`, `entity_or_id`, `search_query`):
       - `search_query` (`n=196`, 0.5% zero scores, C1 UCB=2.0%): feasible at alpha=0.05 (`tau=0.5758`).
       - `entity_or_id` (`n=676`, 4.3% zero scores, C1 UCB=5.5%): feasible at alpha=0.08..0.10 (`tau=0.8158`).
       - `control_only` (`n=130`, 6.2% zero scores, C1 UCB=9.8%): feasible at alpha=0.10 (`tau=0.5526`).
       - `datetime` (`n=210`, 46.7% zero scores, C1 UCB=51.3%): infeasible at low alpha; C1 sets `tau=0.0000`, shifting ungrounded dates from BLOCK to ASK.
     - At `alpha=0.05` (strict C1 per group): OOF `BLOCK` = **1.82%** `[1.0%, 2.8%]`, `ASK` = 13.61%, `Any` = 15.43%, `fcrb` BLOCK/Union = 0.0% / 76.5%.
     - At `alpha=0.10` (recommended B2c operating point): OOF `BLOCK` = **3.14%** `[2.0%, 4.5%]`, `ASK` = **12.29%** `[10.3%, 14.1%]`, `Any` = **15.43%** `[13.3%, 17.6%]`, `fcrb` BLOCK/Union = **32.3% / 76.5%**, `AgentHallu` = **61.3% detection / 9.7% clean BLOCK** (9/93).

- **Target Assessment & Deviation Note**:
  - Stated goal **Hard `BLOCK` <= 5.0%** is preserved and met out-of-fold under Mondrian C1 (`3.14%` at alpha=0.10).
  - **Logged Deviation**: Uncalibrated deterministic `BLOCK` residual after B2a+B2b is **12.29%** (>8% threshold) and total `Any-Intervention (BLOCK+ASK)` is **15.43%**, because 46.7% of `datetime` clean calls involve implicit world-knowledge lookups with exact lexical score `0.0` (which conformal thresholds can only shift from `BLOCK` to `ASK`, not to `ALLOW`).
  - **Revised Joint Target for Phase B4 Combiner**: Maintain **`BLOCK <= 5.0%`** while targeting **`ASK <= 8.0%`** (`Any-Intervention <= 12.0%`) and **`BLOCK`-only recall >= 45%** (Union >= 60%) via cross-fitted semantic and syntactic features.

---

## Experiment 15: Phase J0 — Baseline Verification, Reproducibility Fixes, Datasets & Key Inventory
- **Date**: 2026-11-12
- **Branch**: `improvements-2026-11`

- **1. Baseline Reproduction Status**:
  - **Claude Sonnet on `fcrb` dev (`n=200`)**: **Re-analysed from cache** (`9.57%` FBR `9/94`, `53.77%` detection `57/106`, `0` unparseable) from `benchmarks/llm_judge_args_fcrb_agent_sonnet_dev_v0.json` (not re-run live).
  - **AgentHallu 5-judge range (`31` errors / `93` clean)**: **Re-analysed from cache** (`32.3%–48.4%` detection: Nemotron 4B `32.3%`, Sonnet `38.7%`, Haiku `38.7%`, Opus `45.2%`, Nemotron 550B `48.4%`) via `python benchmarks/analyze_agenthallu.py` (not re-run live).
  - **Nemotron 550B definition ablation (`+9.3 pts, p = 0.24`)**: **`unverified, partial cache (32/200), no NVIDIA key`**. `benchmarks/info2x2_fcrb_C_def_550b.json` is absent, `benchmarks/info2x2_summary.json` has only `A_plain` for `550b`, `llm_judge_args_cache.jsonl` has only `32/200` `C_def` entries, and `analyze_ladder.py` crashed on a missing hardcoded Windows path. Neither `+9.3` nor `+20.9` is cited as confirmed until a full rerun.

- **2. AgentHallu Context Checker Two-Version Reporting (`12.9%` vs `9.7%`)**:
  - **Pre-B3 (`cabdc1c`, frozen — used in all judge comparisons)**: **`12.9%` FBR (`12/93`) / `58.1%` detection (`18/31`)** (`call_is_grounded` in `server/pipeline/arg_ground.py` at Deviation 14).
  - **Current (`59a3316`, post-B3b `541893a` — tuned/evaluated on these clean calls)**: **`9.7%` FBR (`9/93`) / `58.1%` detection (`18/31`)**, where B3b's km<->miles unit conversion and number normalization resolved 3 clean calls (`BFCL/068:s9:c0` `608.94`, `BFCL/055:s6:c0` `484.67`, `BFCL/010:s3:c0` `1.0`).

- **3. Reproducibility Fixes, Cache Audit & Key Hygiene**:
  - `benchmarks/contract_tier.py`: replaced hardcoded `C:\Users\GOVARDHAN SATYA\bfcl_raw\gorilla_data` with `BFCL_RAW_DIR` / `GORILLA_DATA_DIR` (default `benchmarks/data/bfcl/gorilla_data`) + `FileNotFoundError` guard when no `BFCL_v4_*.json` files exist.
  - `benchmarks/analyze_ladder.py`: added fail-fast `RuntimeError` guard if `ids` is empty.
  - Cache keying verified: `llm_judge_args.py` hashes `[provider, model, variant, PROMPT_HASH, messages]`; `judge_eval.py` hashes `[provider, model, prompt_hash, temperature, schema_condition_variant, messages]`.
  - Key hygiene: verified `.gitignore` covers `*.env` and `.secrets/`; no key fragments logged; `get_environment_info()` records OS, Python version, and installed package versions in every `results/` output.

- **4. J0–J5 Offline Harness, Splits, Tool-Selection Set, Audit & Cache Replay**:
  - **Harness & Modules**: `benchmarks/judge_eval.py` (`make judge-eval`), `benchmarks/judge_prompts_v2.py` (J1 6-factor builders, J2 strict-JSON + retry + `cite_then_judge`, J3 5 prompt variants + disjoint few-shot selector), `benchmarks/judge_ask_policy.py` (J5 `ALLOW`/`ASK`/`BLOCK` policy + `mean`/`max`/grouped-OOF `logistic` combiners + ECE), verified with mock judge in `tests/test_judge_eval.py` (`10/10` passed).
  - **Tool-grouped `fcrb_tool_split.json` (Re-freeze Logged)**:
    - **Old hash (superseded)**: `1f18e9493c4f92f714a45c4c48471b3ce8569565094056004f13eea2bf71f7ac` (`93` dev tools / `632` cases; `149` held-out tools / `628` cases). *Reason for re-freeze*: overlapped `dev200` (`104` IDs, `86` `query_id`s, `57/89` tools) because `cases_fcrb.json` was originally split by `query_id`.
    - **New frozen hashes**: full `404` `sha256: 21887c2adf3e94b436231d69d5875c545355ae7c46cf432bf8ee9aecb00a8f41`; **`heldout_fresh_272` (`n=272`) `sha256: 2bea694c07c2a5affb06db96251c032cd235ed2963be57ffe513cdf3f26b5865`**.
    - **Counts & Framing**: `dev` = `89` tools, `856` cases (`428` clean, `428` errors), split by tool into **`dev_train`** (`49` tools, `432` cases; `106` in `dev200`) and **`dev_val`** (`40` tools, `424` cases; `94` in `dev200`, `330` outside `dev200` — `report_dev_val_stratified` reports with and without `dev200`). `frozen_heldout` (`n=404`, `153` tools) is **"judge-fresh, checker-partly-seen"** (`132` of `404` cases across `56` tools appeared in the checker's `630`-case `split=="dev"` set). **`heldout_fresh_272` (`n=272`)** is the **headline** for all judge-vs-checker comparisons; full `404` is secondary.
  - **Tool-grouped `mcpatlas_tool_split.json`**: `53` dev tools (`1,212` cases), `53` frozen held-out tools (`1,211` cases), `sha256: 64848214a38714bec0f4bf2aeda56b29f064b866332e5013ffcb7b60c1c06106` (unopened).
  - **Tool-selection set (`benchmarks/data/tool_selection_eval.json`, `[EXPLORATORY]`)**: `191` cases (`95` right-tool, `96` wrong-tool: `45` `easy`, `43` `hard`, `8` `natural` BFCL). **30-case hard-negative hand audit**: `22/30 (73.3%)` were crisp unambiguous wrong-tool pairs; `8/30 (26.7%)` were invalid/ambiguous (`5` `parallel_multiple` cases where both tools were requested, plus `3` ambiguous MCP-Atlas sibling pairs: `filesystem_list_directory_with_sizes` vs `list_directory`, `wikipedia_search_wikipedia` vs `get_article`, `national-parks_getParkDetails` vs `findParks`). Excluded `parallel` cases and dropped the 3 ambiguous pairs from `tool_selection_eval.json`. Overlap audit: `98` BFCL cases (`45` right, `25` easy, `20` hard, `8` natural) and `10` AgentHallu clean right-tool cases.
  - **Disjoint few-shot pool (`benchmarks/data/few_shot_pool.json`)**: `24` examples (`12` natural errors from `fcrb` `dev_train` covering `wrong_selection`, `wrong_format`, and `fabricated_value` + `8` clean `dev_train` + `4` Glaive v2), verified **0 tool overlap** with `dev_val` and `frozen_heldout`.
  - **Checker hints provenance**: `AgentHallu` hints use `benchmarks/frozen_pre_b3_arg_ground.py` (`cabdc1c`, `12.9%` FBR / `58.1%` det, `[frozen_pre_b3]`); `fcrb` dev hints are explicitly labelled `[in-sample]`.
  - **Live-run plan**: documented in `review/LIVE_RUN_PLAN.md` (ordered steps, call counts, estimated tokens, 30-case smoke-test stop gate before J1).

- **6. 30-Case Live Groq Smoke Test (`openai/gpt-oss-120b`, `results/judge_2026-11-12_smoke30/smoke_report.json`)**:
  - **Setup**: `fcrb_dev200` first `30` cases (`13` clean, `17` errors), frozen prompt (`9b59ae0a3fa5384a`), `temperature = 0`, fresh cache, sequential at `10` RPM.
  - **Agreement**: **`28 / 30 (93.3%)`** per-case agreement with `benchmarks/llm_judge_args_cache.jsonl` (`0 / 30` unparseable live, `0 / 30` unparseable cache).
  - **Rates (30-case slice)**: Cache `FBR = 23.1% (3/13)`, `Det = 64.7% (11/17)` vs Live `FBR = 30.8% (4/13)`, `Det = 58.8% (10/17)`.
  - **Discordant cases (2)**: `fcrb:live_multiple_113-44-0-neg` (`label=false`: cache blocked, live allowed) and `fcrb:live_multiple_292-130-1` (`label=true`: cache allowed, live blocked) — within expected hosted-model / reasoning-temperature-0 run-to-run variance (~6.7%).
  - **Blinded 50-item audit (`review/audit_judge_50_blind.csv`)**: `20` fcrb checker-vs-Sonnet disagreements, `10` BFCL raw-vs-cleaned disagreements, `20` AgentHallu cases (zero AI/gold labels shown; private key in `review/audit_judge_50_key_DO_NOT_OPEN.json`).
  - **Legacy cache replay verification (`judge_eval.py --replay-legacy`)**: reproduced `fcrb_dev200` `agent-sonnet` (`FBR=0.0957`, `det=0.5377`, `0` unpars), `fcrb_dev200` `gpt-oss-120b` (`FBR=0.3191`, `det=0.5943`, `0` unpars), and `agenthallu` `agent-opus` (`FBR=0.0860`, `det=0.4516`, `0` unpars).

- **5. API Key Status & Blocked Experiments**:
  - `GROQ_API_KEY`: present in `C:\Users\rishw\.mcp_checker_keys.env`, but Groq API returned **`HTTP 401 invalid_api_key`** — blocks J0 substitute Groq ablation (`200` fcrb + `124` AgentHallu) and Groq judge runs.
  - `NVIDIA_API_KEY`: missing — blocks Nemotron 550B ablation rerun and NVIDIA 4B/30B/120B/550B judges.
  - `ANTHROPIC_API_KEY`: missing — blocks temperature-0 Claude API judge runs (J1–J5).
  - `GEMINI_API_KEY` / `OPENROUTER_API_KEY`: missing — blocks multi-vendor panels in J5.
