"""
Near-miss adversarial regression tests for Phase B3.
Every rule added in B3 must verify that valid reformulations are GROUNDED,
while near-miss adversarial corruptions (off-by-one dates, digit changes,
unit mismatches, wrong tickers) remain strictly UNGROUNDED.
"""

import pytest
from server.pipeline.arg_ground import check_arguments, check_value


# -----------------------------------------------------------------------------
# B3c: DateTime Normalization & Adversarial Tests
# -----------------------------------------------------------------------------
def test_b3c_valid_dates():
    # Full ISO
    res = check_value("date", "2025-07-15", "I have an appointment on July 15, 2025.")
    assert res.verdict == "GROUNDED"

    # Full ISO with midnight timestamp
    res = check_value("timeMin", "2025-07-15T00:00:00Z", "I have an appointment on July 15, 2025.")
    assert res.verdict == "GROUNDED"

    # Year-Month
    res = check_value("value", "2023-07", "I need car sales for July 2023.")
    assert res.verdict == "GROUNDED"

    # Quarter
    res = check_value("value", "2020-Q1", "Review report for the first quarter of 2020.")
    assert res.verdict == "GROUNDED"

    # Month boundaries when period requested
    res_start = check_value("dt", "2020-04-01", "Weather data for April 2020.")
    assert res_start.verdict == "GROUNDED"
    res_end = check_value("dt", "2020-04-30", "Weather data for April 2020.")
    assert res_end.verdict == "GROUNDED"


def test_b3c_near_miss_adversarial_dates():
    # 1. Off-by-one day corruption
    res = check_value("date", "2025-07-16", "I have an appointment on July 15, 2025.")
    assert res.verdict == "UNGROUNDED", "Off-by-one day must be UNGROUNDED"

    # 2. Wrong month corruption
    res = check_value("date", "2025-08-15", "I have an appointment on July 15, 2025.")
    assert res.verdict == "UNGROUNDED", "Wrong month must be UNGROUNDED"

    # 3. Wrong year corruption
    res = check_value("date", "2024-07-15", "I have an appointment on July 15, 2025.")
    assert res.verdict == "UNGROUNDED", "Wrong year must be UNGROUNDED"

    # 4. Off-by-one month in Year-Month format
    res = check_value("value", "2023-08", "I need car sales for July 2023.")
    assert res.verdict == "UNGROUNDED", "Off-by-one month must be UNGROUNDED"

    # 5. Wrong quarter corruption
    res = check_value("value", "2020-Q2", "Review report for the first quarter of 2020.")
    assert res.verdict == "UNGROUNDED", "Wrong quarter must be UNGROUNDED"

    # 6. Mid-month arbitrary day invention when only month requested
    res = check_value("dt", "2020-04-15", "Weather data for April 2020.")
    assert res.verdict == "UNGROUNDED", "Arbitrary mid-month date must be UNGROUNDED"
