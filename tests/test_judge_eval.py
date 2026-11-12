"""Unit tests for the J0 judge_eval harness and dataset hygiene."""
from __future__ import annotations

import json
from pathlib import Path

import pytest

from benchmarks.judge_eval import (
    build_judge_messages,
    evaluate_judge,
    extract_auto_format_rules,
    judge_cache_key,
    parse_judge_output,
)


def test_auto_format_rules_extracted_from_schema():
    schema_txt = 'Tool definition:\n{"name": "get_weather", "parameters": {"properties": {"location": {"description": "City, State e.g. Seattle, WA"}, "date": {"description": "YYYY-MM-DD"}, "unit": {"enum": ["C", "F"]}}}}'
    rules = extract_auto_format_rules(schema_txt)
    assert len(rules) == 3
    assert any("City, State" in r for r in rules)
    assert any("YYYY-MM-DD" in r for r in rules)
    assert any("enum" in r for r in rules)


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
