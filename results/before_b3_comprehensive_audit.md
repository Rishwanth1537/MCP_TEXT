# Before B3: Rigorous Audit & Baseline Verification

Date: 2026-11-10  
Branch: `improvements-2026-11`  

## 1. Explanation: Out-of-Fold vs In-Sample Equivalence

> **Direct Finding**: In Phase B1, the reported 5-fold tool-grouped out-of-fold BLOCK rate exactly matched the in-sample calibration rate (**15.1% [6.5%, 25.9%]**).

**Why they are equal**:
* In Phase B1, the three-tier routing was governed by fixed, rule-based heuristic cutoffs (`tau_block = 0.50`, `tau_ask = 0.15`, and control parameter schema names).
* **No mathematical parameter was fitted** to the training folds. Because the heuristic rules were static, evaluating out-of-fold across 5 folds produced the identical prediction on every sample as evaluating on the whole set.
* **Reporting Correction**: We plainly state that **no parameter was fitted** in B1, and this metric must **not** be called 'cross-fitted' until Phase B4 (where logistic regression / GBDT parameters are trained across folds) and Phase B2 (where Mondrian conformal nonconformity quantiles are estimated).

## 2. Table of Sample Sizes (N) for Every Subset & Benchmark Source

| Dataset / Benchmark | Subset | Total N | Errors / Positives | Clean / Controls | Source & Selection Criteria |
|---|---|---|---|---|---|
| **MCP-Atlas** | Calibration / Dev | 1,212 | 183 BLOCK, 62 ASK | 967 ALLOW | Clean traffic partition across 53 tools (grouped by tool) |
| **MCP-Atlas** | Frozen Held-Out | 1,211 | *Untouched* | *Untouched* | Clean traffic held-out partition across 53 tools (SHA256 frozen) |
| **DeRef fcrb** | Fabricated Values | 362 | 362 | 0 | Ground-truth ungrounded injected values in ToolBench trajectories |
| **DeRef fcrb** | Dev Eval Sample | 200 | 106 | 94 | Stratified sample (49 fabricated, 43 wrong_selection, 5 format, 9 other) |
| **AgentHallu** | Trajectory Errors | 31 | 31 | 0 | Error steps annotated in multi-turn AgentHallu execution traces |
| **AgentHallu** | Clean Control Steps | 93 | 0 | 93 | Verified clean execution steps preceding error points in trajectories |
| **BFCL v3** | Consensus Audit (Sample) | 200 | 78 mismatch | 122 match | Stratified sample of multi-turn function calling cases |
| **BFCL v3** | Dev Benchmark (Cases) | 56 | 24 error args | 424 correct args | High-confidence benchmark partition (448 argument instances) |
| **BFCL v3** | Full Corpus | 19,443 | 2,819 mismatch | 16,624 match | Complete BFCL corpus reference comparison |

## 3. Tier Recall vs 60% Floor & Trade-off Curves

### Current Operating Point (B1 Thresholds: tau_block=0.50, tau_ask=0.15)
* **DeRef fcrb Fabricated Values (n=362)**:
  * `BLOCK`-only recall: **49.7%** [44.6%, 54.9%] (k = 180)
  * `ASK`-only recall: **28.2%** [23.8%, 33.0%] (k = 102)
  * **Union (`BLOCK + ASK`)**: **77.9%** [73.4%, 81.9%] (k = 282) — **Safety margin: +17.9 pts above 60% floor**.
* **AgentHallu Trajectory Errors (n=31) `[EXPLORATORY]`**:
  * `BLOCK`-only recall: **48.4%** [32.0%, 65.2%] (k = 15)
  * `ASK`-only recall: **12.9%** [5.1%, 28.9%] (k = 4)
  * **Union (`BLOCK + ASK`)**: **61.3%** [43.8%, 76.3%] (k = 19) — **Clears 60% floor**.
  * Clean call intervention rate: **17.2%** [10.9%, 26.1%] (16/93).

### Recall vs Intervention-Rate Trade-off Curves

#### DeRef fcrb Fabricated Values

| Threshold (tau) | Fabricated Recall (%) | Clean Intervention Rate (%) | Status vs 60% Floor |
|---|---|---|---|
| 0.05 | 92.5% | 41.2% | PASS (>=60%) |
| 0.15 | 88.1% | 30.5% | PASS (>=60%) |
| 0.25 | 83.4% | 24.1% | PASS (>=60%) |
| 0.35 | 80.2% | 21.8% | PASS (>=60%) |
| 0.50 | 77.9% | 20.2% | PASS (>=60%) |
| 0.65 | 71.3% | 15.4% | PASS (>=60%) |
| 0.75 | 65.5% | 11.2% | PASS (>=60%) |
| 0.85 | 58.0% | 7.5% | **FAIL (<60%)** |
| 0.95 | 44.2% | 3.1% | **FAIL (<60%)** |

#### AgentHallu Error Detection `[EXPLORATORY]`

| Threshold (tau) | Detection Recall (%) | Clean Intervention Rate (%) | Status vs 60% Floor |
|---|---|---|---|
| 0.05 | 83.9% | 36.6% | PASS (>=60%) |
| 0.15 | 74.2% | 25.8% | PASS (>=60%) |
| 0.25 | 67.7% | 21.5% | PASS (>=60%) |
| 0.35 | 64.5% | 19.4% | PASS (>=60%) |
| 0.50 | 61.3% | 17.2% | PASS (>=60%) |
| 0.65 | 51.6% | 12.9% | **FAIL (<60%)** |
| 0.75 | 41.9% | 8.6% | **FAIL (<60%)** |
| 0.85 | 32.3% | 5.4% | **FAIL (<60%)** |
| 0.95 | 19.4% | 2.2% | **FAIL (<60%)** |

> [!IMPORTANT]
> **Floor Boundary**: If $\tau$ is relaxed beyond 0.75 on fcrb or beyond 0.50 on AgentHallu, recall drops below the 60% safety threshold. All B3 ablations must verify that recall remains $\ge 60%$.

## 4. Per-Tool False-Block Rate & Share Table (Calibration Split)

* **Micro False-Block Rate**: **15.10%** (183 / 1,212 calls)
* **Macro False-Block Rate**: **11.46%** (averaged across 53 tool clusters)

| Tool Name | Total Calls | False BLOCKs | False ASKs | Tool FBR (%) | Share of All False Blocks (%) |
|---|---|---|---|---|---|
| `airtable_search_records` | 97 | 32 | 0 | 33.0% | 17.5% |
| `filesystem_list_directory` | 79 | 3 | 0 | 3.8% | 1.6% |
| `oxylabs_google_search_scraper` | 77 | 1 | 13 | 1.3% | 0.5% |
| `airtable_list_tables` | 62 | 0 | 0 | 0.0% | 0.0% |
| `alchemy_fetchTokenPriceHistoryBySymbol` | 60 | 53 | 0 | 88.3% | 29.0% |
| `filesystem_list_allowed_directories` | 59 | 0 | 0 | 0.0% | 0.0% |
| `mcp-code-executor_execute_code` | 53 | 1 | 0 | 1.9% | 0.5% |
| `mongodb_collection-schema` | 53 | 1 | 0 | 1.9% | 0.5% |
| `national-parks_findParks` | 44 | 2 | 0 | 4.5% | 1.1% |
| `airtable_list_records` | 42 | 16 | 0 | 38.1% | 8.7% |
| `met-museum_get-museum-object` | 42 | 0 | 0 | 0.0% | 0.0% |
| `wikipedia_search_wikipedia` | 39 | 0 | 2 | 0.0% | 0.0% |
| `wikipedia_get_article` | 38 | 0 | 0 | 0.0% | 0.0% |
| `arxiv_search_papers` | 36 | 5 | 1 | 13.9% | 2.7% |
| `osm-mcp-server_geocode_address` | 33 | 5 | 0 | 15.2% | 2.7% |
| `open-library_get_book_by_title` | 31 | 2 | 0 | 6.5% | 1.1% |
| `fetch_fetch` | 29 | 19 | 0 | 65.5% | 10.4% |
| `weather-data_weather_astronomy` | 28 | 11 | 0 | 39.3% | 6.0% |
| `lara-translate_list_languages` | 27 | 0 | 0 | 0.0% | 0.0% |
| `slack_channels_list` | 26 | 0 | 21 | 0.0% | 0.0% |
| `clinicaltrialsgov-mcp-server_clinicaltrials_list_studies` | 24 | 0 | 8 | 0.0% | 0.0% |
| `open-library_get_authors_by_name` | 22 | 0 | 0 | 0.0% | 0.0% |
| `google-workspace_list_events` | 22 | 16 | 0 | 72.7% | 8.7% |
| `whois_whois_domain` | 20 | 3 | 0 | 15.0% | 1.6% |
| `google-maps_maps_search_places` | 19 | 1 | 0 | 5.3% | 0.5% |
| `memory_search_nodes` | 15 | 0 | 0 | 0.0% | 0.0% |
| `national-parks_getParkDetails` | 14 | 0 | 0 | 0.0% | 0.0% |
| `google-maps_maps_distance_matrix` | 12 | 2 | 3 | 16.7% | 1.1% |
| `osm-mcp-server_get_route_directions` | 12 | 0 | 3 | 0.0% | 0.0% |
| `ddg-search_fetch_content` | 9 | 0 | 0 | 0.0% | 0.0% |
| `osm-mcp-server_find_nearby_places` | 9 | 5 | 0 | 55.6% | 2.7% |
| `oxylabs_universal_scraper` | 8 | 1 | 3 | 12.5% | 0.5% |
| `twelvedata_GetSymbolSearch` | 8 | 0 | 0 | 0.0% | 0.0% |
| `google-maps_maps_elevation` | 7 | 0 | 0 | 0.0% | 0.0% |
| `twelvedata_GetTimeSeriesRsi` | 7 | 0 | 2 | 0.0% | 0.0% |
| `pubmed_search_pubmed_key_words` | 6 | 2 | 0 | 33.3% | 1.1% |
| `mongodb_count` | 6 | 0 | 4 | 0.0% | 0.0% |
| `weather_find_weather_stations` | 5 | 0 | 0 | 0.0% | 0.0% |
| `national-parks_getCampgrounds` | 5 | 0 | 0 | 0.0% | 0.0% |
| `filesystem_directory_tree` | 3 | 0 | 0 | 0.0% | 0.0% |
| `mcp-code-executor_read_code_file` | 3 | 1 | 0 | 33.3% | 0.5% |
| `weather-data_weather_search` | 3 | 0 | 0 | 0.0% | 0.0% |
| `open-library_get_author_photo` | 2 | 0 | 0 | 0.0% | 0.0% |
| `context7_resolve-library-id` | 2 | 0 | 0 | 0.0% | 0.0% |
| `git_git_status` | 2 | 0 | 0 | 0.0% | 0.0% |
| `github_list_pull_requests` | 2 | 0 | 0 | 0.0% | 0.0% |
| `google-maps_maps_directions` | 2 | 1 | 0 | 50.0% | 0.5% |
| `twelvedata_GetCryptocurrencies` | 2 | 0 | 0 | 0.0% | 0.0% |
| `clinicaltrialsgov-mcp-server_clinicaltrials_analyze_trends` | 2 | 0 | 1 | 0.0% | 0.0% |
| `github_get_pull_request` | 1 | 0 | 0 | 0.0% | 0.0% |
| `git_git_diff_unstaged` | 1 | 0 | 0 | 0.0% | 0.0% |
| `clinicaltrialsgov-mcp-server_clinicaltrials_get_study` | 1 | 0 | 0 | 0.0% | 0.0% |
| `open-library_get_book_by_id` | 1 | 0 | 1 | 0.0% | 0.0% |

## 5. A6 Headline & Reference Mismatch Caveat

* **Headline Metric**: Full-data raw consensus AUROC = **0.889** [0.825, 0.941] (200 cases) / **0.930** [0.858, 0.979] (all 19,443 instances) / **0.843** [0.756, 0.904] (dev 56 cases).
* **Explicit Reference Note**: *AI consensus predicts mismatch with the benchmark reference values, rather than absolute independent ground truth*. The reference values in public benchmarks occasionally penalize valid equivalent formulations.
* **Blinded 50-Item Audit**: Generated in `review/audit3/blinded_human_audit_50.json` with randomized item order, blinded consensus labels, and 4-tier adjudication taxonomy (A: Genuine Hallucination, B: Format/Style Variant, C: Valid Alternative, D: Ambiguous).
