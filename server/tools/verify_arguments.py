"""
verify_argument_grounding: argument-level grounding of a planned tool call against its context (deterministic, no model).

Three-tier decision (Phase B1):
  ALLOW: call_score >= tau_ask (default 0.88)
  ASK:   tau_block <= call_score < tau_ask OR absent value belongs to low-confidence / control class
  BLOCK: call_score < tau_block (default 0.50) on high-confidence checkable argument

Legacy binary mode is supported when decision_mode="binary".
"""
from __future__ import annotations

from typing import Any, List

from .. import session as session_module
from ..models.schemas import (
    ArgumentGroundingRequest,
    ArgumentGroundingResponse,
    ArgumentVerdictModel,
    BlockReason,
    Decision,
    HallucinationType,
    MismatchedSpan,
)
from ..pipeline.arg_ground import check_arguments

LOW_CONFIDENCE_PARAM_SUBSTRINGS = (
    "limit", "offset", "page", "page_size", "sort", "order", "direction",
    "filter", "query", "search", "keyword", "keywords", "prompt",
    "cursor", "next_page", "mode", "format", "view", "fields",
    "params", "options", "config", "type", "category", "tag",
)


def is_low_confidence_arg(arg_name: str) -> bool:
    arg_clean = arg_name.strip().lower()
    return any(sub in arg_clean for sub in LOW_CONFIDENCE_PARAM_SUBSTRINGS)


def run_argument_grounding(req: ArgumentGroundingRequest) -> ArgumentGroundingResponse:
    verdicts = check_arguments(req.arguments, req.context)
    optional = set((req.schema_hint or {}).get("optional", [])) if req.policy == "required_only" else set()
    effective = [v for v in verdicts if v.verdict != "UNCHECKABLE" and v.argument not in optional]
    call_score = min((v.score for v in effective), default=1.0)
    
    # Thresholds
    tau_block = req.tau_block if req.tau_block is not None else 0.50
    tau_ask = req.tau_ask if req.tau_ask is not None else req.tau
    
    tau_threshold = tau_ask if req.decision_mode == "three_tier" else req.tau

    ungrounded_effective = [v for v in effective if v.score < tau_threshold]
    
    if req.decision_mode == "binary":
        if call_score >= req.tau:
            decision = Decision.ALLOW
            block_reason = BlockReason.NONE
        else:
            decision = Decision.BLOCK
            block_reason = BlockReason.HIGH_CONFIDENCE_ABSENT
    else:
        # Three-tier mode (Phase B1)
        if call_score >= tau_ask:
            decision = Decision.ALLOW
            block_reason = BlockReason.NONE
        elif call_score < tau_block:
            # Check if any ungrounded argument is a high-confidence entity vs low-confidence control param
            high_conf_missing = [v for v in ungrounded_effective if not is_low_confidence_arg(v.argument)]
            if high_conf_missing:
                decision = Decision.BLOCK
                block_reason = BlockReason.HIGH_CONFIDENCE_ABSENT
            else:
                decision = Decision.ASK
                block_reason = BlockReason.LOW_CONFIDENCE_CLASS
        else:
            decision = Decision.ASK
            block_reason = BlockReason.MEDIUM_CONFIDENCE_UNGROUNDED

    is_grounded = (decision == Decision.ALLOW)

    mismatched: List[MismatchedSpan] = []
    if not is_grounded:
        mismatched = [
            MismatchedSpan(
                span=f"{v.argument}={v.value!r}",
                issue=v.reason or "value not supported by the context",
                hallucination_type=HallucinationType.CONTENT,
            )
            for v in ungrounded_effective
        ]

    if req.session_id:
        session_module.tracker.record(req.session_id, call_score)

    if decision == Decision.ALLOW:
        feedback = "All checkable argument values are supported by the context."
    elif decision == Decision.ASK:
        feedback = (
            f"Human confirmation recommended ({block_reason.value}): argument values require user confirmation before proceeding:\n"
            + "\n".join(f"- {m.span}: {m.issue}" for m in mismatched)
        )
    else:
        feedback = (
            f"Tool call blocked ({block_reason.value}): high-confidence ungrounded arguments detected:\n"
            + "\n".join(f"- {m.span}: {m.issue}" for m in mismatched)
        )

    return ArgumentGroundingResponse(
        is_grounded=is_grounded,
        decision=decision,
        block_reason=block_reason,
        call_score=round(call_score, 4),
        tau=req.tau,
        tau_block=tau_block,
        tau_ask=tau_ask,
        per_argument=[
            ArgumentVerdictModel(
                argument=v.argument,
                value=v.value,
                verdict=v.verdict,
                score=round(v.score, 4),
                reason=v.reason,
            )
            for v in verdicts
        ],
        mismatched_spans=mismatched,
        feedback=feedback,
    )
