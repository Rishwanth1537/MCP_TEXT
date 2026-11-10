"""
Unit tests for the InterceptingToolProxy middleware.

All ML model calls are mocked — these tests run without downloading models.
"""
from __future__ import annotations
import pytest
from unittest.mock import patch, MagicMock

from server.proxy.interceptor import (
    InterceptingToolProxy,
    RiskLevel,
    BlockReason,
    guarded,
)
from server.models.schemas import HallucinationType, MismatchedSpan


# ── Test helpers ───────────────────────────────────────────────────────────────

def _make_tool(return_value="executed"):
    """Return a simple stub function that records calls."""
    calls = []
    def _tool(**kwargs):
        calls.append(kwargs)
        return return_value
    _tool.calls = calls
    return _tool


_SIMPLE_SCHEMA = {
    "required": ["amount", "to_account"],
    "properties": {
        "amount":     {"type": "number"},
        "to_account": {"type": "string"},
    },
}

_ALIGNED_RESULT = MagicMock(
    is_aligned=True, mismatches=[], schema_drift_score=1.0
)
_SCHEMA_FAIL_RESULT = MagicMock(
    is_aligned=False,
    mismatches=[
        MismatchedSpan(
            span="to_account",
            issue="Required field 'to_account' is missing.",
            hallucination_type=HallucinationType.SCHEMA,
        )
    ],
    schema_drift_score=0.5,
)
_LOGIC_PASS_RESULT = MagicMock(
    is_consistent=True,
    mismatched_spans=[],
    factual_consistency_score=0.92,
    ged_normalised=0.08,
    feedback="Reasoning chain is structurally consistent.",
)
_LOGIC_FAIL_RESULT = MagicMock(
    is_consistent=False,
    mismatched_spans=[
        MismatchedSpan(
            span="500",
            issue="Entailment score 0.01 — span not supported by premise.",
            hallucination_type=HallucinationType.CONTENT,
        )
    ],
    factual_consistency_score=0.12,
    ged_normalised=0.88,
    feedback="Reasoning chain inconsistent.",
)


# ── LOW risk: always passes through ───────────────────────────────────────────

def test_low_risk_bypasses_all_verification():
    tool = _make_tool("read_ok")
    proxy = InterceptingToolProxy(tool_fn=tool, risk_level=RiskLevel.LOW)
    result = proxy.call(agent_args={"query": "SELECT * FROM logs"})
    assert result.blocked is False
    assert result.tool_output == "read_ok"
    assert len(tool.calls) == 1


# ── MEDIUM risk: schema only ──────────────────────────────────────────────────

def test_medium_risk_passes_on_valid_schema():
    tool = _make_tool()
    proxy = InterceptingToolProxy(
        tool_fn=tool,
        tool_schema=_SIMPLE_SCHEMA,
        risk_level=RiskLevel.MEDIUM,
    )
    with patch("server.tools.schema.run_schema_alignment", return_value=_ALIGNED_RESULT):
        result = proxy.call(agent_args={"amount": 50.0, "to_account": "ACC-001"})
    assert result.blocked is False
    assert len(tool.calls) == 1


def test_medium_risk_blocks_on_schema_failure():
    tool = _make_tool()
    proxy = InterceptingToolProxy(
        tool_fn=tool,
        tool_schema=_SIMPLE_SCHEMA,
        risk_level=RiskLevel.MEDIUM,
    )
    with patch("server.tools.schema.run_schema_alignment", return_value=_SCHEMA_FAIL_RESULT):
        result = proxy.call(agent_args={"amount": 50.0})  # missing to_account
    assert result.blocked is True
    assert result.block_reason == BlockReason.SCHEMA_MISMATCH
    assert len(result.schema_mismatches) == 1
    assert len(tool.calls) == 0  # tool must NOT have been called


# ── HIGH risk: schema + logic ─────────────────────────────────────────────────

def test_high_risk_executes_when_both_pass():
    tool = _make_tool("transfer_ok")
    proxy = InterceptingToolProxy(
        tool_fn=tool,
        tool_schema=_SIMPLE_SCHEMA,
        risk_level=RiskLevel.HIGH,
        logic_mode="hybrid",
    )
    with patch("server.tools.schema.run_schema_alignment", return_value=_ALIGNED_RESULT), \
         patch("server.tools.verify_logic.run_verification", return_value=_LOGIC_PASS_RESULT):
        result = proxy.call(
            agent_args={"amount": 50.0, "to_account": "ACC-001"},
            reasoning_context="Transfer $50 to ACC-001.",
            source_context="Approved: transfer $50 to ACC-001.",
        )
    assert result.blocked is False
    assert result.tool_output == "transfer_ok"
    assert len(tool.calls) == 1


def test_high_risk_blocks_on_schema_failure_before_logic():
    """Schema failure short-circuits before logic check is even attempted."""
    tool = _make_tool()
    proxy = InterceptingToolProxy(
        tool_fn=tool,
        tool_schema=_SIMPLE_SCHEMA,
        risk_level=RiskLevel.HIGH,
        logic_mode="hybrid",
    )
    with patch("server.tools.schema.run_schema_alignment", return_value=_SCHEMA_FAIL_RESULT) as mock_schema, \
         patch("server.tools.verify_logic.run_verification") as mock_logic:
        result = proxy.call(
            agent_args={"amount": 50.0},
            reasoning_context="Transfer $50.",
            source_context="Approved: $50.",
        )
    assert result.blocked is True
    assert result.block_reason == BlockReason.SCHEMA_MISMATCH
    mock_logic.assert_not_called()  # logic check never ran
    assert len(tool.calls) == 0


def test_high_risk_blocks_on_logic_failure():
    """Schema passes but logic is inconsistent → blocked."""
    tool = _make_tool()
    proxy = InterceptingToolProxy(
        tool_fn=tool,
        tool_schema=_SIMPLE_SCHEMA,
        risk_level=RiskLevel.HIGH,
        logic_mode="hybrid",
    )
    with patch("server.tools.schema.run_schema_alignment", return_value=_ALIGNED_RESULT), \
         patch("server.tools.verify_logic.run_verification", return_value=_LOGIC_FAIL_RESULT):
        result = proxy.call(
            agent_args={"amount": 500.0, "to_account": "ACC-001"},
            reasoning_context="Transfer $500 to ACC-001.",  # hallucinated amount
            source_context="Approved: transfer $50 to ACC-001.",
        )
    assert result.blocked is True
    assert result.block_reason == BlockReason.LOGIC_INCONSISTENT
    assert len(result.logic_mismatches) == 1
    assert result.logic_mismatches[0].span == "500"
    assert len(tool.calls) == 0


def test_high_risk_blocks_when_no_evidence():
    """A HIGH-risk call without any evidence is blocked instead of running on the schema check alone."""
    tool = _make_tool("ok")
    proxy = InterceptingToolProxy(tool_fn=tool, tool_schema=_SIMPLE_SCHEMA, risk_level=RiskLevel.HIGH)
    with patch("server.tools.schema.run_schema_alignment", return_value=_ALIGNED_RESULT):
        result = proxy.call(agent_args={"amount": 50.0, "to_account": "ACC-001"})
    assert result.blocked is True
    assert result.block_reason == BlockReason.NO_EVIDENCE
    assert len(tool.calls) == 0


def test_high_risk_without_evidence_runs_only_if_explicitly_allowed():
    """require_evidence=False restores the old behaviour (schema check only)."""
    tool = _make_tool("ok")
    proxy = InterceptingToolProxy(tool_fn=tool, tool_schema=_SIMPLE_SCHEMA, risk_level=RiskLevel.HIGH, require_evidence=False)
    with patch("server.tools.schema.run_schema_alignment", return_value=_ALIGNED_RESULT):
        result = proxy.call(agent_args={"amount": 50.0, "to_account": "ACC-001"})
    assert result.blocked is False
    assert len(tool.calls) == 1


def test_default_mode_checks_the_actual_arguments():
    """Default mode is argument grounding: a wrong value is blocked, the right one passes, no model needed."""
    tool = _make_tool("ok")
    proxy = InterceptingToolProxy(tool_fn=tool, tool_schema=_SIMPLE_SCHEMA, risk_level=RiskLevel.HIGH)
    assert proxy.logic_mode == "arguments"
    ctx = "User: please send 50 dollars to ACC-001."
    bad = proxy.call({"amount": 5000.0, "to_account": "ACC-001"}, source_context=ctx)
    good = proxy.call({"amount": 50.0, "to_account": "ACC-001"}, source_context=ctx)
    assert bad.blocked is True and good.blocked is False
    assert tool.calls == [{"amount": 50.0, "to_account": "ACC-001"}]


@pytest.mark.parametrize("mode,target", [("hybrid", "server.tools.verify_logic.run_verification"),
                                         ("grounded", "server.tools.verify_grounded.run_grounded_verification")])
def test_arguments_are_checked_even_when_the_reasoning_text_passes(mode, target):
    """Reasoning says 50 and passes the text check, but the call carries 5000: it must still be blocked."""
    tool = _make_tool("ok")
    proxy = InterceptingToolProxy(tool_fn=tool, tool_schema=_SIMPLE_SCHEMA, risk_level=RiskLevel.HIGH, logic_mode=mode)
    with patch(target, return_value=_LOGIC_PASS_RESULT):
        result = proxy.call({"amount": 5000.0, "to_account": "ACC-001"},
                            reasoning_context="Send 50 dollars to ACC-001.",
                            source_context="User: please send 50 dollars to ACC-001.")
    assert result.blocked is True
    assert any("amount" in m.span for m in result.logic_mismatches)
    assert len(tool.calls) == 0


def test_evidence_provider_overrides_agent_supplied_context():
    """With an evidence_provider the agent cannot ground a value in a context it wrote itself."""
    tool = _make_tool("ok")
    proxy = InterceptingToolProxy(tool_fn=tool, tool_schema=_SIMPLE_SCHEMA, risk_level=RiskLevel.HIGH,
                                  evidence_provider=lambda: "User: please send 50 dollars to ACC-001.")
    forged = proxy.call({"amount": 5000.0, "to_account": "ACC-001"}, source_context="Approved: send 5000 dollars to ACC-001.")
    honest = proxy.call({"amount": 50.0, "to_account": "ACC-001"})
    assert forged.blocked is True and honest.blocked is False


def test_unknown_logic_mode_is_rejected():
    with pytest.raises(ValueError):
        InterceptingToolProxy(tool_fn=_make_tool(), logic_mode="typo")


# ── Custom GED threshold ───────────────────────────────────────────────────────

def test_custom_ged_threshold_passed_to_logic_check():
    """The proxy's ged_threshold is forwarded to run_verification."""
    tool = _make_tool()
    proxy = InterceptingToolProxy(
        tool_fn=tool,
        tool_schema=_SIMPLE_SCHEMA,
        risk_level=RiskLevel.HIGH,
        ged_threshold=0.20,
        logic_mode="hybrid",
    )
    captured_kwargs = {}

    def _capture(**kwargs):
        captured_kwargs.update(kwargs)
        return _LOGIC_PASS_RESULT

    with patch("server.tools.schema.run_schema_alignment", return_value=_ALIGNED_RESULT), \
         patch("server.tools.verify_logic.run_verification", side_effect=_capture):
        proxy.call(
            agent_args={"amount": 50.0, "to_account": "ACC-001"},
            reasoning_context="Transfer $50.",
            source_context="Approved: $50.",
        )
    assert captured_kwargs.get("threshold") == 0.20


# ── @guarded decorator ─────────────────────────────────────────────────────────

def test_guarded_decorator_attaches_proxy():
    @guarded(tool_schema=_SIMPLE_SCHEMA, risk_level=RiskLevel.HIGH)
    def my_tool(amount: float, to_account: str) -> str:
        return "done"

    # Direct call still works (bypasses proxy)
    assert my_tool(amount=50.0, to_account="X") == "done"
    # Proxy attribute exists
    assert hasattr(my_tool, "proxy")
    assert isinstance(my_tool.proxy, InterceptingToolProxy)


def test_guarded_decorator_proxy_blocks():
    @guarded(tool_schema=_SIMPLE_SCHEMA, risk_level=RiskLevel.HIGH)
    def my_tool(amount: float, to_account: str) -> str:
        return "done"

    with patch("server.tools.schema.run_schema_alignment", return_value=_SCHEMA_FAIL_RESULT):
        result = my_tool.proxy.call(agent_args={"amount": 50.0})  # missing to_account

    assert result.blocked is True


# ── No tool_schema: logic-only check ─────────────────────────────────────────

def test_no_schema_skips_to_logic_check():
    """When tool_schema is empty, schema check is skipped; logic check runs."""
    tool = _make_tool("ok")
    proxy = InterceptingToolProxy(
        tool_fn=tool,
        tool_schema={},  # no schema
        risk_level=RiskLevel.HIGH,
        logic_mode="hybrid",
    )
    with patch("server.tools.verify_logic.run_verification", return_value=_LOGIC_PASS_RESULT):
        result = proxy.call(
            agent_args={"x": 1},
            reasoning_context="Do something.",
            source_context="Context: x is 1.",
        )
    assert result.blocked is False
    assert len(tool.calls) == 1


# ── Three-Tier Decision: ALLOW | ASK | BLOCK (Task B1) ──────────────────────

from server.models.schemas import Decision


def test_three_tier_allow_executes_tool():
    tool = _make_tool('executed_allow')
    proxy = InterceptingToolProxy(
        tool_fn=tool,
        tool_schema={'properties': {'account': {'type': 'string'}}},
        risk_level=RiskLevel.HIGH,
        decision_mode='three_tier',
    )
    result = proxy.call(
        agent_args={'account': 'ACC-123'},
        source_context='Approved transfer to account ACC-123.',
    )
    assert result.decision == Decision.ALLOW
    assert result.blocked is False
    assert result.requires_confirmation is False
    assert result.tool_output == 'executed_allow'
    assert len(tool.calls) == 1


def test_three_tier_block_halts_execution():
    tool = _make_tool()
    proxy = InterceptingToolProxy(
        tool_fn=tool,
        tool_schema={'properties': {'account': {'type': 'string'}}},
        risk_level=RiskLevel.HIGH,
        decision_mode='three_tier',
    )
    # High-confidence identifier completely missing from context
    result = proxy.call(
        agent_args={'account': 'FABRICATED-999'},
        source_context='Context mentions ACC-123 only.',
    )
    assert result.decision == Decision.BLOCK
    assert result.blocked is True
    assert result.requires_confirmation is False
    assert result.block_reason == BlockReason.HIGH_CONFIDENCE_ABSENT
    assert len(tool.calls) == 0


def test_three_tier_ask_suspends_execution_for_confirmation():
    tool = _make_tool()
    proxy = InterceptingToolProxy(
        tool_fn=tool,
        tool_schema={'properties': {'limit': {'type': 'integer'}}},
        risk_level=RiskLevel.HIGH,
        decision_mode='three_tier',
    )
    # Control param ungrounded -> should trigger ASK, not hard BLOCK
    result = proxy.call(
        agent_args={'limit': 50},
        source_context='User asked: list all recent transactions.',
    )
    assert result.decision == Decision.ASK
    assert result.blocked is False
    assert result.requires_confirmation is True
    assert result.block_reason == BlockReason.LOW_CONFIDENCE_CLASS
    assert len(tool.calls) == 0  # tool not called without confirmation


def test_binary_compatibility_mode_blocks_on_ask():
    tool = _make_tool()
    proxy = InterceptingToolProxy(
        tool_fn=tool,
        tool_schema={'properties': {'limit': {'type': 'integer'}}},
        risk_level=RiskLevel.HIGH,
        decision_mode='binary',
    )
    result = proxy.call(
        agent_args={'limit': 50},
        source_context='User asked: list all recent transactions.',
    )
    assert result.decision == Decision.BLOCK
    assert result.blocked is True
    assert result.requires_confirmation is False
    assert len(tool.calls) == 0
