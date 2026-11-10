"""Unit tests for provenance-aware grounding (server/pipeline/provenance.py, Deviation 21)."""
import pytest

from server.pipeline.provenance import (
    check_call,
    evidence_from_agentdojo_premise,
    evidence_from_agenthallu_premise,
)

USER = ("user", "How many issues did the assaultcube repository get in November 2024?")
TOOL = ("tool", '{"owner": "assaultcube", "repo": "AC", "id": 15102551, "created_at": "2013-12-04"}')


def leaf(v, path):
    return next(lv for lv in v.leaves if lv.path == path)


def test_tool_output_grounds_identifier_with_provenance():
    v = check_call({"repo": "AC", "repo_id": 15102551}, [USER, TOOL], "P1")
    assert not v.blocked
    assert leaf(v, "repo").source == "tool" and leaf(v, "repo_id").source == "tool"


def test_user_source_wins_over_tool():
    v = check_call({"owner": "assaultcube"}, [USER, TOOL], "P1")
    assert leaf(v, "owner").source == "user"


def test_agent_text_is_not_evidence_by_default():
    ev = [USER, ("agent", "I will use the repository xyz-invented")]
    assert check_call({"repo": "xyz-invented"}, ev, "P1").blocked
    assert not check_call({"repo": "xyz-invented"}, ev, "P1", include_agent=True).blocked


def test_invented_value_still_blocks():
    assert check_call({"repo": "totally-different-name"}, [USER, TOOL], "P3").blocked


@pytest.mark.parametrize("value,ok", [("2024-11-01", True), ("2024-11-30", True), ("2024-11-15", False),
                                      ("2024-11-30T23:59:59Z", True), ("2024-12-01", False)])
def test_date_range_expansion(value, ok):
    assert check_call({"since": value}, [USER], "P2").blocked is (not ok)


def test_date_range_not_in_p1():
    assert check_call({"since": "2024-11-01"}, [USER], "P1").blocked


def test_year_bounds():
    ev = [("user", "papers published in 2023")]
    assert not check_call({"from": "2023-01-01", "to": "2023-12-31"}, ev, "P2").blocked
    assert check_call({"from": "2023-06-01"}, ev, "P2").blocked


def test_nested_dicts_checked_in_p2_not_p1():
    args = {"params": {"symbol": "ZZZQ", "interval": "1day"}}
    assert not check_call(args, [USER], "P1").blocked          # dict is UNCHECKABLE in P1 (as in the old checker)
    assert check_call(args, [USER], "P2").blocked              # leaf-by-leaf in P2


def test_control_parameters_only_in_p3():
    assert check_call({"max_results": 50}, [USER], "P2").blocked
    assert not check_call({"max_results": 50}, [USER], "P3").blocked
    assert check_call({"max_results": "fifty-ish"}, [USER], "P3").blocked     # not an integer: normal check


def test_program_arguments_uncheckable_in_p3():
    v = check_call({"code": "import math\nprint(2024-2013)"}, [USER], "P3")
    assert not v.blocked and leaf(v, "code").reason == "generated program"


def test_free_text_query_rule():
    assert not check_call({"query": "assaultcube repository issues"}, [USER], "P3").blocked
    assert check_call({"query": "weather forecast paris"}, [USER], "P3").blocked


def test_json_string_values_are_walked_in_p3():
    v = check_call({"filter": '{"a": 1}', "payload": '{"repo": "nonexistent-zzz"}'}, [USER, TOOL], "P3")
    assert v.blocked and any(lv.path == "payload.repo" for lv in v.leaves)


def test_long_tool_output_exact_match_beyond_fuzzy_window():
    big = "x " * 120_000 + " needle-identifier-42 " + "y " * 10
    early = "needle-identifier-42 " + "x " * 120_000
    assert not check_call({"id": "needle-identifier-42"}, [("tool", early)], "P1").blocked
    assert not check_call({"id": "needle-identifier-42"}, [("tool", big)], "P1").blocked


def test_agenthallu_premise_split():
    p = "User task:\nAdd 38 liters\n[tool call] liter_to_gallon {\"liter\": 38}\n[tool result] {\"gallon\": 10.04}\n[assistant] ok"
    ev = evidence_from_agenthallu_premise(p)
    assert ev[0] == ("user", "Add 38 liters")
    assert [s for s, _ in ev[1:]] == ["agent", "tool", "agent"]


def test_agentdojo_premise_split():
    p = "[system] be helpful\n[user] pay bill\n[assistant] calling\n[tool] IBAN DE123"
    assert [s for s, _ in evidence_from_agentdojo_premise(p)] == ["user", "user", "agent", "tool"]
