"""
J1 Prompt Builders, J2 Strict-JSON Schema & Retry Validator, and J3 Prompt Variants + Few-Shot Selector.

Ground Rules enforced:
- Zero gold labels, error-type labels, or dataset names in any prompt.
- Few-shot pool comes strictly from benchmarks/data/few_shot_pool.json (disjoint from eval/heldout).
"""
from __future__ import annotations

import hashlib
import json
import random
import re
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

ROOT = Path(__file__).resolve().parents[1]

VALID_VERDICTS = {"correct", "incorrect", "uncertain"}

J2_SCHEMA_INSTRUCTION = (
    'Respond ONLY with valid JSON matching this exact schema: '
    '{"verdict": "correct|incorrect|uncertain", "failing_arguments": ["arg_name", ...], '
    '"error_type": "none|fabricated_value|wrong_selection|wrong_format|swapped_slot|stale_value|wrong_tool|other", '
    '"evidence_quote": "short quote from context or none", "confidence": 0-100}.'
)

J2_CITE_THEN_JUDGE_INSTRUCTION = (
    'First list where each argument value came from in the context (or "none"), then decide the verdict. '
    'Respond ONLY with valid JSON matching this exact schema: '
    '{"argument_citations": {"arg_name": "exact source quote or none"}, '
    '"verdict": "correct|incorrect|uncertain", "failing_arguments": ["arg_name", ...], '
    '"error_type": "none|fabricated_value|wrong_selection|wrong_format|swapped_slot|stale_value|wrong_tool|other", '
    '"evidence_quote": "short quote from context or none", "confidence": 0-100}.'
)

J3_VARIANTS: Dict[str, str] = {
    "role_framing": (
        "You are a strict security-auditor verifying AI agent tool calls before execution. "
        "Check whether the chosen tool and every argument value are justified by the user request, "
        "verified prior tool outputs, and tool definition (if provided)."
    ),
    "checklist": (
        "Verify the proposed tool call step by step:\n"
        "1. Is the chosen tool the right tool for the user's goal?\n"
        "2. Is each argument value supported by user text or verified tool outputs (not unverified assistant text)?\n"
        "3. Do all values satisfy parameter types, enums, and format conventions in the tool definition?\n"
        "4. Check for swapped slots (e.g., origin/destination), stale values from superseded turns, or wrong selections."
    ),
    "arg_audit": (
        "Audit each argument of the proposed tool call independently against the context and tool definition. "
        "A value present in the context can still be wrong if it refers to the wrong entity, slot, or turn."
    ),
    "contrastive": (
        "Play devil's advocate before approving this tool call: ask yourself 'How could this tool choice or any "
        "argument value be wrong?' Check for fabricated numbers/dates, wrong-entity selection, swapped slots, "
        "stale turn values, and format mismatches."
    ),
    "cite_then_judge": (
        "You are an evidence-grounded tool-call verifier. Never guess support: trace each argument value to an "
        "explicit user or tool-output span before issuing a verdict."
    ),
}


def extract_auto_format_rules(tool_def_text: str) -> List[str]:
    if not tool_def_text:
        return []
    rules: List[str] = []
    if re.search(r"City,\s*State|city\s+and\s+state|e\.g\.?\s*[A-Z][a-z]+,\s*[A-Z]{2}", tool_def_text):
        rules.append("Format convention: location parameters requesting 'City, State' must include city and state separated by a comma.")
    if re.search(r"YYYY-MM-DD", tool_def_text):
        rules.append("Format convention: date parameters specifying YYYY-MM-DD must use ISO-8601 YYYY-MM-DD format.")
    if re.search(r"HH:MM", tool_def_text):
        rules.append("Format convention: time parameters specifying HH:MM must use 24-hour HH:MM format.")
    if re.search(r'"enum"\s*:', tool_def_text):
        rules.append("Schema convention: parameters with an 'enum' list must use one of the exact allowed enum values.")
    return rules


def format_source_labelled_context(case: Dict[str, Any]) -> str:
    """Label user, tool outputs, and unverified agent text separately when structured fields exist."""
    if any(k in case for k in ("user", "tool_results", "agent")):
        parts = []
        if case.get("user"):
            parts.append(f"[USER]\n{case['user']}")
        if case.get("tool_results"):
            tr = case["tool_results"]
            if isinstance(tr, list):
                tr_txt = "\n".join(f"[TOOL_OUTPUT #{i+1}] {x}" for i, x in enumerate(tr))
            else:
                tr_txt = f"[TOOL_OUTPUT]\n{tr}"
            parts.append(tr_txt)
        if case.get("agent"):
            parts.append(f"[AGENT_UNVERIFIED - NOT EVIDENCE]\n{case['agent']}")
        return "\n\n".join(parts)
    # Fallback: tag lines in flat premise
    raw = case.get("premise", "")
    lines = []
    for ln in raw.splitlines():
        low = ln.lower()
        if low.startswith("[tool") or low.startswith("tool output"):
            lines.append(f"[TOOL_OUTPUT] {ln}")
        elif low.startswith("assistant:") or low.startswith("agent:"):
            lines.append(f"[AGENT_UNVERIFIED] {ln}")
        elif low.startswith("user"):
            lines.append(f"[USER] {ln}")
        else:
            lines.append(ln)
    return "\n".join(lines)


def build_checker_hints(
    call: Dict[str, Any],
    evidence_text: str,
    include_score: bool = False,
    checker_scope: str = "in_sample",
) -> str:
    """Run deterministic arg_ground checker as structured evidence hints for the judge.
    - checker_scope='frozen_pre_b3': uses frozen cabdc1c checker (required for AgentHallu so hints are not tuned on clean calls).
    - checker_scope='in_sample': uses current checker and labels hints [in-sample] (for fcrb dev).
    """
    if checker_scope == "frozen_pre_b3":
        from benchmarks.frozen_pre_b3_arg_ground import call_score, check_arguments  # noqa: PLC0415
        tag = "[frozen_pre_b3]"
    else:
        from server.pipeline.arg_ground import call_score, check_arguments  # noqa: PLC0415
        tag = "[in-sample]"

    args = call.get("arguments") or {}
    if not isinstance(args, dict) or not args:
        return f"Deterministic checker hints {tag}: (no arguments to check)"
    verdicts = check_arguments(args, evidence_text)
    lines = [f"Deterministic checker hints per argument {tag}:"]
    for v in verdicts:
        status = v.verdict
        lines.append(f'- arg "{v.argument}" = {json.dumps(v.value, ensure_ascii=False)}: {status}')
    if include_score:
        sc = call_score(args, evidence_text)
        lines.append(f"Deterministic checker call_score {tag}: {sc:.3f}")
    return "\n".join(lines)


def select_few_shots(
    case: Dict[str, Any],
    pool: List[Dict[str, Any]],
    k: int = 4,
    mode: str = "random",
    seed: int = 0,
) -> List[Dict[str, Any]]:
    if k <= 0 or not pool:
        return []
    if mode == "random":
        rng = random.Random(seed)
        chosen = list(pool)
        rng.shuffle(chosen)
        return chosen[:k]
    if mode == "similar":
        q_tokens = set(re.findall(r"[a-z0-9_]+", (case.get("premise") or "").lower()))

        def sim(ex: Dict[str, Any]) -> float:
            e_tokens = set(re.findall(r"[a-z0-9_]+", (ex.get("premise") or "").lower()))
            union = q_tokens | e_tokens
            return len(q_tokens & e_tokens) / len(union) if union else 0.0

        ranked = sorted(pool, key=sim, reverse=True)
        return ranked[:k]
    raise ValueError(f"Unknown few-shot mode: {mode}")


def build_j1_j3_messages(
    case: Dict[str, Any],
    *,
    include_definition: bool = False,
    include_other_tools: bool = False,
    source_labelled: bool = False,
    include_checker_hints: bool = False,
    include_checker_score: bool = False,
    checker_scope: str = "in_sample",
    include_autorules: bool = False,
    variant: str = "role_framing",
    j2_order: str = "verdict_first",
    few_shots: Optional[List[Dict[str, Any]]] = None,
) -> Tuple[List[Dict[str, str]], str]:
    """Unified J1+J2+J3 message builder with zero label leakage."""
    base_sys = J3_VARIANTS.get(variant, J3_VARIANTS["role_framing"])
    schema_instr = J2_CITE_THEN_JUDGE_INSTRUCTION if j2_order == "cite_then_judge" else J2_SCHEMA_INSTRUCTION
    system_prompt = f"{base_sys}\n\n{schema_instr}"

    if source_labelled:
        ctx = format_source_labelled_context(case)
    else:
        ctx = case.get("premise") or case.get("user_request") or case.get("context") or ""

    if include_definition and case.get("premise_with_schema"):
        if source_labelled and "Tool definition" in case["premise_with_schema"]:
            def_block = "Tool definition" + case["premise_with_schema"].split("Tool definition", 1)[1]
            ctx = f"{ctx}\n\n{def_block}"
        elif not source_labelled:
            ctx = case["premise_with_schema"]

    sections = [f"Context:\n{ctx}"]

    if include_other_tools and case.get("available_tools"):
        tools_txt = json.dumps(case["available_tools"], ensure_ascii=False)
        sections.append(f"Other available tools:\n{tools_txt}")

    call = case.get("calls") or case.get("call") or {"name": case.get("chosen_tool", ""), "arguments": case.get("arguments", {})}
    if isinstance(call, list):
        call_obj = call[0] if call else {}
    else:
        call_obj = call

    if include_checker_hints:
        sections.append(build_checker_hints(call_obj, ctx, include_score=include_checker_score, checker_scope=checker_scope))

    if include_autorules:
        def_part = ctx.split("Tool definition", 1)[1] if "Tool definition" in ctx else json.dumps(case.get("available_tools", ""))
        rules = extract_auto_format_rules(def_part)
        if rules:
            sections.append("Extracted format rules:\n- " + "\n- ".join(rules))

    if few_shots:
        shot_lines = []
        for idx, ex in enumerate(few_shots, 1):
            ans = {
                "verdict": "correct" if ex["supported"] else "incorrect",
                "failing_arguments": ex.get("unsupported_arguments", []),
                "error_type": "none" if ex["supported"] else "fabricated_value",
                "evidence_quote": "none",
                "confidence": 95,
            }
            shot_lines.append(
                f"Example {idx}:\nContext: {ex['premise'][:400]}\n"
                f"Proposed tool call: {json.dumps(ex['call'], ensure_ascii=False)}\n"
                f"Answer: {json.dumps(ans)}"
            )
        sections.insert(0, "Reference Examples:\n" + "\n\n".join(shot_lines))

    sections.append(f"Proposed tool call:\n{json.dumps(call, ensure_ascii=False)}")
    msgs = [
        {"role": "system", "content": system_prompt},
        {"role": "user", "content": "\n\n".join(sections)},
    ]
    phash = hashlib.sha256(system_prompt.encode("utf-8")).hexdigest()[:16]
    return msgs, phash


def validate_j2_verdict(text: str, require_citations: bool = False) -> Tuple[Optional[Dict[str, Any]], Optional[str]]:
    """Validate J2 strict JSON verdict. Returns (parsed_dict, error_reason)."""
    if not text or not text.strip():
        return None, "Empty response."
    m = re.search(r"\{.*\}", text, re.S)
    if not m:
        return None, "No JSON object found."
    try:
        obj = json.loads(m.group(0))
    except Exception as exc:
        return None, f"Invalid JSON: {exc}"
    if not isinstance(obj, dict):
        return None, "Top-level JSON must be an object."

    verdict = obj.get("verdict")
    if not isinstance(verdict, str) or verdict.strip().lower() not in VALID_VERDICTS:
        return None, f"Invalid or missing 'verdict' (must be one of {sorted(VALID_VERDICTS)})."
    verdict = verdict.strip().lower()

    failing = obj.get("failing_arguments")
    if not isinstance(failing, list):
        return None, "Missing or non-list 'failing_arguments'."

    conf = obj.get("confidence")
    if not isinstance(conf, (int, float)) or isinstance(conf, bool) or not (0 <= float(conf) <= 100):
        return None, "Missing or out-of-range 'confidence' (0-100)."

    if require_citations and not isinstance(obj.get("argument_citations"), dict):
        return None, "Missing 'argument_citations' object required in cite-then-judge mode."

    conf_f = float(conf)
    if verdict == "correct":
        p_err = (100.0 - conf_f) / 100.0
    elif verdict == "incorrect":
        p_err = conf_f / 100.0
    else:
        p_err = 0.5

    return {
        "verdict": verdict,
        "supported": verdict == "correct",
        "failing_arguments": [str(x) for x in failing],
        "error_type": str(obj.get("error_type", "other")),
        "evidence_quote": str(obj.get("evidence_quote", "")),
        "argument_citations": obj.get("argument_citations"),
        "confidence": conf_f,
        "error_score": float(p_err),
    }, None
