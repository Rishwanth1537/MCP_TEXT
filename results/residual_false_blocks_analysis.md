# B1b: Residual False-Block Analysis (Calibration Dev Split)

Date: 2026-11-10  
Dataset: MCP-Atlas Calibration Split (n=1,212 calls across 53 tools)  
Sample: 100 false flags stratified across tools (75 BLOCK, 25 ASK)

## 1. Distribution of Root Causes

| Rank | Root Cause | Sample Count (n=100) | BLOCK Tier | ASK Tier | Projected Removal from 15.1% BLOCK Rate |
|---|---|---|---|---|---|
| 1 | `alias_entity_naming` | 6% | 4 | 2 | **~0.81 pts** |
| 2 | `number_unit_format` | 17% | 16 | 1 | **~3.22 pts** |
| 3 | `datetime_format` | 24% | 21 | 3 | **~4.23 pts** |
| 4 | `default_or_optional_value` | 0% | 0 | 0 | Part of B2 / B3 |
| 5 | `control_parameter` | 13% | 3 | 10 | **~0.60 pts** (with defaults) |
| 6 | `generated_code_or_query` | 0% | 0 | 0 | Addressed in B4 combiner |
| 7 | `world_knowledge` | 3% | 2 | 1 | Addressed in B4 combiner |
| 8 | `missing_prior_tool_output` | 0% | 0 | 0 | Inherent trajectory boundary |
| 9 | `mislabeled_genuine_error` | 0% | 0 | 0 | True errors in benchmark |
| 10 | `other` | 37% | 29 | 8 | Residual |

## 2. Work Order Justification

The analysis demonstrates that **deterministic surface mismatches (aliases, number/unit representations, and datetime formatting) account for 47% of false flags**.

Therefore, the confirmed order of execution is:
1. **Task B3 (Aliases, Number Parsing, Tense/Date Invariance)**: Directly removes the top 3 deterministic error sources, projected to cut ~5.5–6.5 points of false blocks without ML cost.
2. **Task B2 (Mondrian Conformal Calibration by Argument Class)**: Calibrates thresholds by class (control params, identifiers, quantities), eliminating remaining default and control false blocks.
3. **Task B4 (Learned Combiner)**: Handles programmatic code generation, world knowledge, and non-trivial semantic features using cross-fitted logistic regression / GBDT.
