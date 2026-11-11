"""Unit tests for Phase B2: Schema Validation, Host DateTime Evidence, and Mondrian C1 Calibration."""
import pytest
from server.models.schemas import ArgumentGroundingRequest, BlockReason, Decision
from server.tools.verify_arguments import run_argument_grounding
from server.pipeline.provenance import check_call
from server.pipeline.arg_ground import call_argument_class, mondrian_conformal_threshold_ucb


def test_b2a_enum_and_buffer_inside_schema_allow():
    # Enum valid inside tool registry (no user conflict)
    req = ArgumentGroundingRequest(
        arguments={"interval": "1d"},
        context="Check Ethereum prices last week.",
        tool_name="alchemy_fetchTokenPriceHistoryBySymbol",
        decision_mode="three_tier",
    )
    res = run_argument_grounding(req)
    assert res.decision == Decision.ALLOW

    # CSV Enum valid for Slack
    req_slack = ArgumentGroundingRequest(
        arguments={"channel_types": "public_channel, private_channel"},
        context="List the movie suggestions channel.",
        tool_name="slack_channels_list",
        decision_mode="three_tier",
    )
    res_slack = run_argument_grounding(req_slack)
    assert res_slack.decision == Decision.ALLOW

    # Buffer ceiling valid inside fetch_fetch
    req_fetch = ArgumentGroundingRequest(
        arguments={"max_length": 999999},
        context="Fetch the documentation page.",
        tool_name="fetch_fetch",
        decision_mode="three_tier",
    )
    res_fetch = run_argument_grounding(req_fetch)
    assert res_fetch.decision == Decision.ALLOW


def test_b2a_control_param_with_explicit_schema_bounds_allows():
    req = ArgumentGroundingRequest(
        arguments={"limit": 25},
        context="List users.",
        schema_hint={"properties": {"limit": {"type": "integer", "minimum": 1, "maximum": 100}}},
        decision_mode="three_tier",
    )
    res = run_argument_grounding(req)
    assert res.decision == Decision.ALLOW


def test_b2a_schema_violations_and_enum_conflicts_block():
    # Out-of-bounds negative buffer
    req_neg = ArgumentGroundingRequest(
        arguments={"max_length": -5},
        context="Fetch page.",
        tool_name="fetch_fetch",
        decision_mode="three_tier",
    )
    res_neg = run_argument_grounding(req_neg)
    assert res_neg.decision == Decision.BLOCK

    # Invalid enum value
    req_bad_enum = ArgumentGroundingRequest(
        arguments={"interval": "invalid_step"},
        context="Check Ethereum price.",
        tool_name="alchemy_fetchTokenPriceHistoryBySymbol",
        decision_mode="three_tier",
    )
    res_bad_enum = run_argument_grounding(req_bad_enum)
    assert res_bad_enum.decision == Decision.BLOCK

    # Contradicts explicit user enum preference
    req_conflict = ArgumentGroundingRequest(
        arguments={"mode": "driving"},
        context="Get walking directions to the museum.",
        tool_name="google-maps_maps_distance_matrix",
        decision_mode="three_tier",
    )
    res_conflict = run_argument_grounding(req_conflict)
    assert res_conflict.decision == Decision.BLOCK
    assert not check_call({"mode": "walking"}, [("user", "Get walking directions")], "P3", tool_name="google-maps_maps_distance_matrix").blocked
    assert check_call({"mode": "driving"}, [("user", "Get walking directions")], "P3", tool_name="google-maps_maps_distance_matrix").blocked


def test_b2b_host_datetime_grounding_and_adversarial():
    host = ("host", "Current date and time: 2025-07-15T12:00:00Z")

    # Relative today / yesterday / tomorrow resolved via host clock
    assert not check_call({"dt": "2025-07-15"}, [host, ("user", "Check events for today.")], "P3").blocked
    assert not check_call({"dt": "2025-07-14"}, [host, ("user", "Check events from yesterday.")], "P3").blocked
    assert not check_call({"dt": "2025-07-16"}, [host, ("user", "Schedule a call for tomorrow.")], "P3").blocked

    # Adversarial off-by-one against host clock must remain BLOCKED
    assert check_call({"dt": "2025-07-16"}, [host, ("user", "Check events for today.")], "P3").blocked
    assert check_call({"dt": "2025-07-12"}, [host, ("user", "Check events from yesterday.")], "P3").blocked

    # Fractional seconds .000Z & lowercase z ISO timestamps
    assert not check_call({"startTime": "2025-07-15T00:00:00.000Z"}, [("user", "Events on July 15, 2025.")], "P3").blocked
    assert not check_call({"startTime": "2025-07-15t00:00:00z"}, [("user", "Events on July 15, 2025.")], "P3").blocked

    # Split month+year and month+day ("this July 2025 ... after July 15th")
    assert not check_call(
        {"timeMin": "2025-07-15T00:00:00Z"},
        [("user", "My visit sometime this July 2025 (after July 15th)")],
        "P3",
    ).blocked
    # Adversarial off-by-one day must still block
    assert check_call(
        {"timeMin": "2025-07-16T00:00:00Z"},
        [("user", "My visit sometime this July 2025 (after July 15th)")],
        "P3",
    ).blocked


def test_b2c_mondrian_conformal_ucb():
    assert call_argument_class({"limit": 10, "mode": "walking"}) == "control_only"
    assert call_argument_class({"query": "python tutorials"}) == "search_query"
    assert call_argument_class({"timeMin": "2025-07-15T00:00:00Z"}) == "datetime"
    assert call_argument_class({"address": "123 Main St"}) == "entity_or_id"

    # Synthetic calibration scores: 100 calls in control_only (all >= 0.90), 50 in datetime (10 score 0.0)
    scores = [0.95] * 100 + [0.0] * 10 + [0.90] * 40
    groups = ["control_only"] * 100 + ["datetime"] * 50
    cal = mondrian_conformal_threshold_ucb(scores, groups, alpha=0.05, delta=0.10)

    # control_only has 0 zero scores out of 100 -> UCB(0/100) <= 0.05 -> feasible at tau=0.95
    assert cal["control_only"]["feasible_at_alpha"] is True
    assert cal["control_only"]["tau"] > 0.0
    assert cal["control_only"]["ucb_at_tau"] <= 0.05

    # datetime has 20% zero scores -> exceeds alpha=0.05 -> tau falls back to 0.0 (feasible_at_alpha=False)
    assert cal["datetime"]["feasible_at_alpha"] is False
    assert cal["datetime"]["tau"] == 0.0
