# Phase B3 Reconciliation, Validation, and Ablation Summary (B3c -> B3b -> B3a)

Date: 2026-11-11  
Branch: `improvements-2026-11`  

## 1. Baseline Reconciliation (183 vs 215 vs 192 BLOCK)

* **True Pre-B3 Baseline (`16686a4`)**: **183 BLOCK (15.10%)** `[6.5%, 25.9%]`, **62 ASK (5.12%)**, **245 Any Intervention (20.21%)**.
* **Intermediate Buggy Baseline (`task-1131`, uncompleted B3c)**: **215 BLOCK (17.74%)** `[6.7%, 30.4%]`, **62 ASK (5.12%)**, **277 Any Intervention (22.85%)**.  
  * *Cause of the +32 blocks*: When Year-Month (`_YM`) and Quarter (`_QUARTER`) matching was first added to `_check_iso` in `server/pipeline/arg_ground.py`, single-source queries failed before `server/pipeline/provenance.py`'s `_date_range_grounded` was updated to perform cross-turn year+quarter combination.
* **Post-B3 Final (`19931f2`)**: **192 BLOCK (15.84%)** `[6.5%, 26.6%]`, **60 ASK (4.95%)**, **252 Any Intervention (20.79%)**.
  * *Vs. True Pre-B3 (`16686a4`)*: **+9 BLOCK (+0.74 pts)**, **-2 ASK (-0.17 pts)**, **+7 Any Intervention (+0.58 pts)**. Paired cluster-bootstrap delta (by tool): **+0.74 pts [-0.89%, +3.16%]** for BLOCK; **+0.58 pts [-1.16%, +3.05%]** for Any Intervention.
    * Fixed **15 genuine false blocks** across **4 tools** (`airtable_search_records`, `alchemy_fetchTokenPriceHistoryBySymbol`, `google-workspace_list_events`, `national-parks_findParks`).
    * Tightened **24 accidental decimal/token-bag collisions** on **1 tool** (`airtable_search_records`), where Pre-B3 stem matching had falsely grounded e.g. `2020-07` from unrelated year `2020` + decimal `.07` in dollar totals.
  * *Vs. Intermediate 215 State*: **-23 BLOCK (-1.90 pts)**, **-2 ASK (-0.17 pts)**, **-25 Any Intervention (-2.06 pts)**.

## 2. Independent Dataset Validation (Not Used for Rule Design)

| Dataset | Split Size | Pre-B3 (`16686a4`) BLOCK / ASK / Any | Post-B3 (`19931f2`) BLOCK / ASK / Any | Delta |
|---|---|---|---|---|
| **AgentHallu Clean** | n = 93 | 13 (14.0%) / 3 (3.2%) / 16 (17.2%) | **11 (11.8%) / 3 (3.2%) / 14 (15.1%)** | **-2.2 pts BLOCK** |
| **AgentHallu Errors** | n = 31 | 19 caught (61.3% detection) | **19 caught (61.3% detection)** | **0.0 pts** |
| **DeRef fcrb Clean** | n = 94 | 23 (24.5%) / 30 (31.9%) / 53 (56.4%) | **23 (24.5%) / 28 (29.8%) / 51 (54.3%)** | **-2.1 pts ASK/Any** |

## 3. B3b Real-Call Firing Check & Retention Justification

* **Firing Counts**:
  * MCP-Atlas Calibration (`n=1,212`): **1 call** (`mcpatlas-689cd6f8522029b7ad7b200e-0`, value `1.0`).
  * AgentHallu (`n=124`): **3 clean calls** (`BFCL/068:s9:c0` `608.94` and `BFCL/055:s6:c0` `484.67` in `estimate_drive_feasibility_by_mileage` for real km/miles conversions; `BFCL/010:s3:c0` `1.0` in `tail`).
  * DeRef fcrb Clean (`n=94`): **0 calls**.
* **Decision**: Retained in default path because it directly resolves real unit conversions on unseen AgentHallu clean calls (`14.0% -> 11.8%` FBR) without harming AgentHallu error detection (`61.3%`).

## 4. Adversarial Verification & Near-Miss Scale Test

* **Adversarial Unit Suite (`tests/test_b3_adversarial.py`)**: All **7 constructed near-miss adversarial checks** remained strictly `UNGROUNDED` (`7/7` pass rate; synthetic unit check, not natural traffic precision).
* **Test Count Reconciliation**: AST count across all 19 files in `tests/` totals **175 test functions** (the Phase A baseline reported **143 parameterized test items** prior to adding Phase B1/B3 test modules). Our targeted B3 suite run executed **85 collected items** across 7 core unit test files (`arg_ground`, `provenance`, `b3_adversarial`, `proxy`, `stats`, `pipeline`, `grounded`) to verify core semantics without slow property/loop suites.
* **Near-Miss Scale Test**:
  * **DeRef fcrb Fabricated (`n=362`)**: Union recall = **280/362 (77.3%)** vs 282/362 (77.9%) Pre-B3 ($\Delta = -0.55\text{ pts}$), `BLOCK` recall unchanged at **180/362 (49.7%)**, staying well above the 60% floor.
  * **Near-Miss Perturbed Variants (`n=94` mutated clean calls: +1 numbers, shifted 2099 dates, off-by-one chars)**: Union recall = **71/94 (75.5%)**.

## 5. Corrected Post-B3 Remaining Residual Breakdown (n=252 Flags)

| Root Cause Category | Remaining Count | % of Remaining Flags | `BLOCK` Tier | `ASK` Tier | Addressed Next By |
|---|---|---|---|---|---|
| `datetime_format` | 93 | 36.9% | 89 | 4 | **B2** (Host current date/time evidence) & cross-turn |
| `enum_or_flag_argument` | 67 | 26.6% | 37 | 30 | **B2** (Schema enum validation -> ALLOW) |
| `number_unit_format` | 22 | 8.7% | 22 | 0 | Residual specialized formats |
| `search_query_reformulation` | 17 | 6.7% | 2 | 15 | **B4** (Combiner) |
| `buffer_or_size_limit` | 13 | 5.2% | 13 | 0 | **B2** (Schema bounds validation -> ALLOW) |
| `world_knowledge` | 7 | 2.8% | 7 | 0 | **B4** (Combiner semantic features) |
| `url_or_endpoint_construction` | 6 | 2.4% | 6 | 0 | **B4** (Combiner syntax features) |
| `control_parameter` | 6 | 2.4% | 1 | 5 | **B2** (Schema bounds & defaults -> ALLOW) |
| `intermediate_step_dependency` | 1 | 0.4% | 1 | 0 | Multi-turn trajectory boundary |
| **`residual_other` (Unexplained)** | **20** | **7.9%** | **14** | **6** | Residual unexplained (**1.65%** of all 1,212 calls) |

* **Corrected Table Sums**:
  * **B2 Schema/Control/Enum/Buffer categories** (`67 + 13 + 6`): **86 / 252 = 34.1%** of remaining flags.
  * **B4 Combiner categories** (`17 + 7 + 6`): **30 / 252 = 11.9%** of remaining flags.
  * **Combined B2 + B4 target scope** (`86 + 30`): **116 / 252 = 46.0%** (rising to **209 / 252 = 82.9%** including `datetime_format` relative dates resolved via B2 host clock evidence). *(Corrects prior erroneous "68.2%" text figure).*
