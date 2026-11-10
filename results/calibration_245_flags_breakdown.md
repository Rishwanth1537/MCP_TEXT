# Breakdown of All 245 Calibration False Flags (183 BLOCK, 62 ASK)

Date: 2026-11-10  
Dataset: MCP-Atlas Calibration Split (53 tools, 1,212 calls)  
Total Flags Analyzed: **245 calls** (100% of calibration false flags, not a sample)

## 1. Full Taxonomy of All 245 False Flags

| Category | Sub-Category Description | Total Flags | % of All Flags | BLOCK Tier | ASK Tier |
|---|---|---|---|---|---|
| `datetime_format` |  | 80 | 32.7% | 76 | 4 |
| `enum_or_flag_argument` |  | 67 | 27.3% | 37 | 30 |
| `number_unit_format` |  | 22 | 9.0% | 22 | 0 |
| `residual_other` |  | 21 | 8.6% | 15 | 6 |
| `search_query_reformulation` |  | 16 | 6.5% | 2 | 14 |
| `buffer_or_size_limit` |  | 13 | 5.3% | 13 | 0 |
| `world_knowledge` |  | 7 | 2.9% | 7 | 0 |
| `control_parameter` |  | 7 | 2.9% | 1 | 6 |
| `url_or_endpoint_construction` | Constructed URL, API endpoint, or repository path | 6 | 2.4% | 6 | 0 |
| `alias_entity_naming` |  | 5 | 2.0% | 3 | 2 |
| `intermediate_step_dependency` |  | 1 | 0.4% | 1 | 0 |

## 2. Breakdown of the Former 'Other' Bucket (37%)

In the initial 100-sample audit, 37% was labeled as 'other'. Decomposing all 245 flags reveals:

1. **`enum_or_flag_argument`**: Standard tool enums and flags (e.g. `channel_types: 'public_channel, private_channel'`, `interval: '1d'`). Accounts for the bulk of Slack and Alchemy tool flags.
2. **`url_or_endpoint_construction`**: API endpoint URLs constructed by the agent (e.g. `fetch` calls to GitHub API `/repos/owner/name`).
3. **`buffer_or_size_limit`**: Safe default buffer ceilings (e.g. `max_length: 999999` in scraper/fetcher tools).
4. **`search_query_reformulation`**: Keyword extraction and search query condensation in search tools (e.g. Google scraper, ArXiv, Wikipedia).
5. **`intermediate_step_dependency`**: Multi-turn tool execution where IDs or record keys came from prior tool outputs in the trajectory.

## 3. Spot-Check of 20 Calls for Mislabeled Clean Calls

We manually inspected 20 calls (10 BLOCK, 10 ASK) across diverse tools (`osm-mcp-server`, `fetch`, `slack`, `weather`, `airtable`, `alchemy`):

* **Genuine Mislabeled Hallucinations in Benchmark**: **0 / 20 (0.0%)**.
* **Finding**: All 20 spot-checked calls were legitimate, valid tool executions where the agent correctly interpreted user intent or tool API requirements. The checker raised false blocks due to strict string lexical matching failure rather than benchmark labeling corruption.

