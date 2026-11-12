"""Unit tests for J0-J5 judge harness, prompt builders, J2 retry/validation, J3 variants, and J5 ask policy."""
from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import pytest

from benchmarks.judge_ask_policy import apply_ask_policy, combine_panel_scores, evaluate_ask_policy, expected_calibration_error
from benchmarks.judge_eval import (
    build_judge_messages,
    evaluate_judge,
    extract_auto_format_rules,
    get_environment_info,
    judge_cache_key,
    parse_judge_output,
)
from benchmarks.judge_prompts_v2 import (
    J3_VARIANTS,
    build_j1_j3_messages,
    select_few_shots,
    validate_j2_verdict,
)


def test_auto_format_rules_extracted_from_schema():
    schema_txt = 'Tool definition:\n{"name": "get_weather", "parameters": {"properties": {"location": {"description": "City, State e.g. Seattle, WA"}, "date": {"description": "YYYY-MM-DD"}, "unit": {"enum": ["C", "F"]}}}}'
    rules = extract_auto_format_rules(schema_txt)
    assert len(rules) == 3


def test_cache_key_sensitive_to_prompt_model_and_temp():
    msgs = [{"role": "user", "content": "hi"}]
    k1 = judge_cache_key("groq", "m1", "hash1", 0.0, "A_plain", msgs)
    k2 = judge_cache_key("groq", "m2", "hash1", 0.0, "A_plain", msgs)
    k3 = judge_cache_key("groq", "m1", "hash2", 0.0, "A_plain", msgs)
    k4 = judge_cache_key("groq", "m1", "hash1", 0.7, "A_plain", msgs)
    assert len({k1, k2, k3, k4}) == 4


def test_prompt_hygiene_no_gold_or_dataset_names():
    case = {
        "id": "fcrb:123",
        "dataset": "fcrb",
        "label": False,
        "mutation": "wrong_selection",
        "premise": "User: book a flight to Paris.",
        "call": {"name": "book_flight", "arguments": {"city": "London"}},
    }
    msgs, _ = build_judge_messages(case, condition="A_plain")
    joined = json.dumps(msgs).lower()
    assert "fcrb" not in joined
    assert "wrong_selection" not in joined
    assert "label" not in joined


def test_evaluate_judge_fails_on_empty_cases():
    with pytest.raises(ValueError, match="0 cases"):
        evaluate_judge([], provider="groq", model="test", dry_run=True)


def test_parse_judge_output_confidence_scoring():
    res = parse_judge_output('{"supported": false, "unsupported_arguments": ["city"], "confidence": 85}')
    assert res["parsed"] is True
    assert res["supported"] is False
    assert res["error_score"] == pytest.approx(0.85)


def test_j1_all_six_factors_and_j3_variants():
    assert len(J3_VARIANTS) >= 5
    case = {
        "id": "c1",
        "user": "Book weather in Austin, TX for 2026-11-12",
        "tool_results": ["Austin, TX forecast available"],
        "agent": "I think London is nice.",
        "premise": "Book weather in Austin, TX for 2026-11-12",
        "premise_with_schema": 'Book weather in Austin, TX for 2026-11-12\n\nTool definition:\n{"name":"get_weather","parameters":{"properties":{"loc":{"description":"City, State"}}}}',
        "call": {"name": "get_weather", "arguments": {"loc": "Austin, TX"}},
        "available_tools": [{"name": "get_weather"}, {"name": "get_news"}],
    }
    pool = json.loads(Path("benchmarks/data/few_shot_pool.json").read_text(encoding="utf-8"))["examples"]
    shots = select_few_shots(case, pool, k=4, mode="similar")
    assert len(shots) == 4
    msgs, _ = build_j1_j3_messages(
        case,
        include_definition=True,
        include_other_tools=True,
        source_labelled=True,
        include_checker_hints=True,
        include_checker_score=True,
        include_autorules=True,
        variant="checklist",
        j2_order="cite_then_judge",
        few_shots=shots,
    )
    txt = msgs[1]["content"]
    assert "[AGENT_UNVERIFIED - NOT EVIDENCE]" in txt
    assert "Other available tools:" in txt
    assert "Deterministic checker hints per argument [in-sample]:" in txt
    assert "Deterministic checker call_score [in-sample]:" in txt
    assert "Extracted format rules:" in txt
    assert "Reference Examples:" in txt


def test_j2_validation_and_mock_judge_retry(tmp_path, monkeypatch):
    monkeypatch.setenv("GROQ_API_KEY", "test-only-dummy")
    calls = {"n": 0}

    def mock_chat(provider, key, model, msgs, max_tokens=280, temperature=0.0):
        calls["n"] += 1
        if calls["n"] == 1:
            return {"choices": [{"message": {"content": "not valid json"}}], "usage": {"prompt_tokens": 10, "completion_tokens": 5}}, 0.01
        return {
            "choices": [{
                "message": {
                    "content": json.dumps({
                        "argument_citations": {"city": "User said Paris"},
                        "verdict": "correct",
                        "failing_arguments": [],
                        "error_type": "none",
                        "evidence_quote": "Paris",
                        "confidence": 92,
                    })
                }
            }],
            "usage": {"prompt_tokens": 20, "completion_tokens": 15},
        }, 0.02

    cases = [
        {"id": "t1", "premise": "User: Paris", "call": {"name": "go", "arguments": {"city": "Paris"}}, "label": True},
        {"id": "t2", "premise": "User: Paris", "call": {"name": "go", "arguments": {"city": "Mars"}}, "label": False},
    ]
    res = evaluate_judge(
        cases,
        provider="groq",
        model="mock-model",
        schema_mode="j2",
        j2_order="cite_then_judge",
        cache_path=tmp_path / "cache.jsonl",
        chat_fn=mock_chat,
        rpm=0,
    )
    assert calls["n"] == 3  # 1 retry on t1 + 1 call on t2
    assert res["n"] == 2
    assert res["n_unparseable"] == 0
    assert res["rows"][0]["confidence"] == 92.0


def test_j2_validator_rejects_bad_schema():
    parsed, err = validate_j2_verdict('{"verdict": "maybe", "failing_arguments": [], "confidence": 50}')
    assert parsed is None and "verdict" in err
    parsed2, err2 = validate_j2_verdict('{"verdict": "correct", "failing_arguments": [], "confidence": 90}', require_citations=True)
    assert parsed2 is None and "argument_citations" in err2


def test_j5_ask_policy_and_panel_combiners():
    scores = np.array([
        [0.05, 0.10, 0.08],
        [0.45, 0.55, 0.50],
        [0.92, 0.88, 0.95],
        [0.10, 0.12, 0.15],
        [0.85, 0.90, 0.80],
        [0.40, 0.60, 0.52],
    ])
    y_err = np.array([False, False, True, False, True, True])
    groups = ["t1", "t2", "t3", "t1", "t2", "t3"]

    assert list(apply_ask_policy(combine_panel_scores(scores, "mean"), 0.2, 0.8)) == ["ALLOW", "ASK", "BLOCK", "ALLOW", "BLOCK", "ASK"]
    oof = combine_panel_scores(scores, "logistic", y_true_err=y_err, groups=groups, n_splits=3)
    assert oof.shape == (6,)
    out = evaluate_ask_policy(scores, y_err, tau_allow=0.2, tau_block=0.8, method="mean", judge_names=["J1", "J2", "J3"])
    assert out["false_block_rate"] == pytest.approx(0.0)
    assert out["ask_rate_correct"] == pytest.approx(1.0 / 3.0)
    assert out["catch_rate_union"] == pytest.approx(1.0)
    assert 0.0 <= expected_calibration_error(y_err, scores.mean(axis=1)) <= 1.0


def test_environment_info_present():
    env = get_environment_info()
    assert "os" in env and "python" in env and "packages" in env


def test_checker_hints_scope_and_tool_selection_breakdown():
    from benchmarks.judge_eval import report_tool_selection_breakdown
    from benchmarks.judge_prompts_v2 import build_checker_hints

    ah_txt = build_checker_hints({"name": "estimate_drive_feasibility_by_mileage", "arguments": {"dist": 608.94}}, "distance 980 km", checker_scope="frozen_pre_b3")
    fcrb_txt = build_checker_hints({"name": "get_weather", "arguments": {"city": "Paris"}}, "Paris", checker_scope="in_sample")
    assert "[frozen_pre_b3]" in ah_txt
    assert "[in-sample]" in fcrb_txt

    cases = [
        {"id": "1", "label": True, "difficulty": "right"},
        {"id": "2", "label": False, "difficulty": "easy"},
        {"id": "3", "label": False, "difficulty": "hard"},
        {"id": "4", "label": False, "difficulty": "natural"},
    ]
    rows = [
        {"id": "1", "label": True, "parsed": True, "supported": True},
        {"id": "2", "label": False, "parsed": True, "supported": False},
        {"id": "3", "label": False, "parsed": True, "supported": True},
        {"id": "4", "label": False, "parsed": True, "supported": False},
    ]
    rep = report_tool_selection_breakdown(rows, cases)
    assert rep["status"] == "EXPLORATORY"
    assert rep["subsets"]["easy"]["catch_rate"] == pytest.approx(1.0)
    assert rep["subsets"]["hard"]["catch_rate"] == pytest.approx(0.0)
    assert rep["subsets"]["natural"]["catch_rate"] == pytest.approx(1.0)
