# Phase B Reporting Fixes & Cluster Rigor

Date: 2026-11-10  
Branch: `improvements-2026-11`

## 1. MCP-Atlas Cluster-Bootstrap Rates & Distribution

* **Total Calibration Traffic**: n = 1212 clean calls across 53 distinct tools.
* **Largest Tool Share (Calibration)**: `airtable_search_records` = 97/1212 (8.0%).
* **Largest Tool Share (Held-out)**: `cli-mcp-server_run_command` = 95/1211 (7.8%).

### Calibration Rates with Cluster-Bootstrap CIs

| Decision Tier | Point Estimate | Primary: Cluster Bootstrap (by Tool) | Secondary: Cluster Bootstrap (by Task) | Secondary: Exact Wilson Interval |
|---|---|---|---|---|
| **Hard False Block (`BLOCK`)** | **15.1%** | **[6.5%, 25.9%]** | [12.7%, 17.5%] | [13.2%, 17.2%] |
| **Confirmation Rate (`ASK`)** | **5.1%** | **[1.6%, 9.9%]** | [3.8%, 6.5%] | [4.0%, 6.5%] |

### 5-Fold Tool-Grouped Out-of-Fold Cross-Fitting
To prevent in-sample overfitting on the calibration set:
* **Out-of-Fold BLOCK Rate**: **15.1%** [6.5%, 25.9%] (n=1212).

### Top 15 Tools False-Block Breakdown (Calibration Split)

| Tool Name | Total Calls | Hard Blocks (`BLOCK`) | Asks (`ASK`) | Passes (`ALLOW`) |
|---|---|---|---|---|
| `airtable_search_records` | 97 | 32 (33.0%) | 0 (0.0%) | 65 |
| `filesystem_list_directory` | 79 | 3 (3.8%) | 0 (0.0%) | 76 |
| `oxylabs_google_search_scraper` | 77 | 1 (1.3%) | 13 (16.9%) | 63 |
| `airtable_list_tables` | 62 | 0 (0.0%) | 0 (0.0%) | 62 |
| `alchemy_fetchTokenPriceHistoryBySymbol` | 60 | 53 (88.3%) | 0 (0.0%) | 7 |
| `filesystem_list_allowed_directories` | 59 | 0 (0.0%) | 0 (0.0%) | 59 |
| `mcp-code-executor_execute_code` | 53 | 1 (1.9%) | 0 (0.0%) | 52 |
| `mongodb_collection-schema` | 53 | 1 (1.9%) | 0 (0.0%) | 52 |
| `national-parks_findParks` | 44 | 2 (4.5%) | 0 (0.0%) | 42 |
| `airtable_list_records` | 42 | 16 (38.1%) | 0 (0.0%) | 26 |
| `met-museum_get-museum-object` | 42 | 0 (0.0%) | 0 (0.0%) | 42 |
| `wikipedia_search_wikipedia` | 39 | 0 (0.0%) | 2 (5.1%) | 37 |
| `wikipedia_get_article` | 38 | 0 (0.0%) | 0 (0.0%) | 38 |
| `arxiv_search_papers` | 36 | 5 (13.9%) | 1 (2.8%) | 30 |
| `osm-mcp-server_geocode_address` | 33 | 5 (15.2%) | 0 (0.0%) | 28 |

---

## 2. Tier Recall Breakdown: BLOCK-only vs ASK-only vs Union

| Evaluation Set | Subset / Target | Total N | BLOCK-only | ASK-only | Union (`BLOCK + ASK`) |
|---|---|---|---|---|---|
| **DeRef fcrb Fabricated** | Absent value recall | 362 | 49.7% [44.6%, 54.9%] (k=180) | 28.2% [23.8%, 33.0%] (k=102) | **77.9% [73.4%, 81.9%] (k=282)** |
| **AgentHallu Errors** | Detection recall | 31 | 48.4% [32.0, 65.2] (k=15) | 12.9% [5.1, 28.9] (k=4) | **61.3% [43.8%, 76.3%] (k=19) [EXPLORATORY]** |
| **AgentHallu Clean Calls** | Clean call intervention | 93 | 14.0% [8.3%, 22.5%] (k=13) | 3.2% [1.1%, 9.1%] (k=3) | **17.2% [10.9%, 26.1%] (k=16)** |

---

## 3. Redone A4 Matched-FPR Comparison on `wrong_selection`

* **Dataset Size & Subsets**: DeRef fcrb dev sample (N = 200 calls):
  * Clean control calls: n = 94
  * Total error calls: n = 106 (`fabricated_value`: 49, `wrong_selection`: 43, `wrong_format`: 5, `other`: 9).
* **Matched False-Block Operating Point**:
  * Checker Clean FBR: **10.6%** [5.9, 18.5] (10/94)
  * Claude Sonnet Clean FBR: **9.6%** [5.1, 17.2] (9/94)
* **Catch Rate on `wrong_selection` (n = 43)**:
  * Deterministic Checker: **25.6%** [14.9, 40.2] (k = 11/43)
  * Claude Sonnet: **41.9%** [28.4, 56.7] (k = 18/43)
  * Exact McNemar Test: 12 discordant cases favoring Sonnet vs 5 favoring Checker ($p = 0.1435$, not statistically significant at n=43).

---

## 4. A6 Headline & 50-Item Human Spot-Check Protocol

* **Headline Metric**: Full-data raw consensus AUROC = **0.889** [0.825, 0.941] (200 cases) / **0.930** [0.858, 0.979] (all 19,443 instances) / **0.843** [0.756, 0.904] (dev 56 cases).
* **Labelled Sensitivity Analysis**: Cleaned high-confidence subset AUROC = **0.947** [0.915, 0.974] (dev 56 cases, n=448, n_wrong=24).
* **50-Item Human Spot-Check Protocol**:
  1. Draw 50 items stratified from `review/audit2/` (25 where AI consensus agreed with gold, 25 where AI consensus disagreed with gold).
  2. Single blinded human adjudicator reviews the original query, tool schema, gold value, and agent output without seeing AI consensus votes.
  3. Adjudication choices: (A) Genuine Hallucination, (B) Harmless Format/Style Variant, (C) Valid Alternative Value, (D) Underspecified Context / Ambiguous.
  4. Measure Cohen's $\kappa$ between human adjudication and AI consensus to validate the sensitivity subset.
