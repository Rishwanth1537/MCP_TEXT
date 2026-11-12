# Live-Run Plan & Token/Call Budget (J0–J8)

**Status:** Pending working API key (`GROQ_API_KEY` currently returns `HTTP 401 invalid_api_key`; `NVIDIA_API_KEY`, `ANTHROPIC_API_KEY`, `GEMINI_API_KEY`, `OPENROUTER_API_KEY` missing).  
**Rule:** Execute sequentially via `benchmarks/judge_eval.py` with caching enabled (`temperature = 0`, no gold/error/dataset labels in prompts).

---

## Ordered Execution Plan & Budget Estimates

Assumptions per call (based on observed `DEV_PILOT_RESULTS.md` ~520 tokens/call plain, ~900 tokens/call with tool definition / J1 hints, ~1,100 tokens/call with J3 few-shots / tool list):

| Step | Experiment | Dataset(s) & Split | Calls / Model | Est. Tokens / Call | Est. Total Tokens / Model | Models Planned | Gate / Stop Rule |
|---|---|---|---|---|---|---|---|
| **0. Smoke Test** | **Live vs Cache Verification** | `fcrb_dev200` (first 30 cases) | **30** | ~520 | **~15.6k** | Strongest Groq model (`openai/gpt-oss-120b`) | **STOP & REPORT**: verify live replies match `llm_judge_args_cache.jsonl` within noise before proceeding to J0b/J1. |
| **J0b. Substitute Ablation** | Plain vs Definition vs Def + Auto-Rule (pre-registered Deviation 23) | `fcrb_dev200` (`200` x 3) + `AgentHallu` (`124` x 3, `[EXPLORATORY]`) | **972** (`600 + 372`) | ~520–900 | **~730k** | Strongest Groq model (`openai/gpt-oss-120b`) | Compare at matched FBR; report format-only vs non-format split. |
| **J1. Input Design** | 6 factors alone + cumulative (`plain`, `+def`, `+other_tools`, `+source_labelled`, `+checker_hints` [with/without score], `+autorules`) | `fcrb_tool_val` (`424`, `[in-sample]` hints) + `AgentHallu` (`124`, `frozen_pre_b3` hints, `[EXPLORATORY]`) | ~**2,740** (`5 x 548` cumulative/ablation) | ~750–1,000 | **~2.3M** | Strongest Groq model + 1 smaller Groq model (e.g. `llama-3.1-8b-instant`) | Select winning input config on `fcrb_tool_val` at 5% & 10% FBR. |
| **J2. Output & Calibration** | Strict JSON + `cite_then_judge` vs `verdict_first` + verbalized conf vs 3-sample mean | `fcrb_tool_val` (`424` x 4) | **1,696** | ~850 | **~1.44M** | Strongest Groq model | Measure AUROC, ECE, reliability curves; fit dev thresholds only. |
| **J3. Prompt Search** | 5 prompt variants (`role_framing`, `checklist`, `arg_audit`, `contrastive`, `cite_then_judge`) + `0/4/8` shots (`random` vs `similar` from `few_shot_pool.json`) | `fcrb_tool_val` (`424` x 9) | **3,816** | ~1,100 | **~4.2M** | Strongest Groq model | Select single best prompt on `fcrb_tool_val` (zero overlap with `dev_train` pool or `heldout`). |
| **J4. Tool Selection** | Right vs wrong tool (`easy` / `hard` / `natural`), with/without `other_tools` | `tool_selection_eval.json` (`193` x 2, `[EXPLORATORY]`) | **386** | ~950 | **~367k** | Strongest Groq model | Report `easy`, `hard`, and `natural` separately (`[EXPLORATORY]`). |
| **J5. Ensembles & Ask Policy** | Panels of 2, 3, 5 models + 3-sample self-consistency + `ALLOW/ASK/BLOCK` | `fcrb_tool_val` (`424`) + `mcpatlas_dev` sample (`300`) | **724** / model | ~850 | **~615k** / model | Groq + (if keys added: Anthropic, NVIDIA, Gemini, OpenRouter) | Target AUROC `>= 0.85` (cleaned labels), `BLOCK <= 5%`, `ASK <= 10%`. |
| **J6. Cost Reduction** | Small-first uncertain-band escalation + prompt caching | `fcrb_tool_val` (`424`) | **424** (small) + ~120 (large) | ~650 | **~380k** | Small Groq (`8B`) -> Large Groq (`120B`) | Target `>= 40%` token/latency reduction at equal detection. |
| **J8. Final Frozen Eval (ONCE)** | Frozen single config on unopened held-out splits + external sets | `fcrb_tool_split` `heldout` (`404`) + `mcpatlas_tool_split` `heldout` (`1,211` or stratified) + `AgentHallu` (`124`) + `tool_selection` (`193`) | **1,932** | ~900 | **~1.74M** | Final frozen panel | Touch `fcrb` and `mcpatlas` held-out splits **strictly once** after hashing config in ledger. |

---

## Immediate Next Step When Key Is Updated
1. Verify key with `python benchmarks/llm_judge_args.py --list-models --provider groq`.
2. Run **Step 0 (30-case smoke test)** on `fcrb_dev200` (`seed=0`, first `30` cases) using `openai/gpt-oss-120b` (`A_plain`) and compare per-case verdicts against `benchmarks/llm_judge_args_cache.jsonl`.
3. **Stop and report** live-vs-cache agreement before starting J0b or J1.
