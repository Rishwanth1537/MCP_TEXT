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


# -----------------------------------------------------------------------------
# B3b: Extended Number & Unit Parsing & Adversarial Tests
# -----------------------------------------------------------------------------
def test_b3b_spelled_numbers():
    # Compound numbers > 99
    res = check_value("amount", 125.0, "Transfer one hundred twenty-five dollars.")
    assert res.verdict == "GROUNDED"
    res2 = check_value("count", 2500, "Found two thousand five hundred results.")
    assert res2.verdict == "GROUNDED"

    # Adversarial: off-by-one or digit injection
    bad1 = check_value("amount", 126.0, "Transfer one hundred twenty-five dollars.")
    assert bad1.verdict == "UNGROUNDED", "Off-by-one number must be UNGROUNDED"
    bad2 = check_value("amount", 1250.0, "Transfer one hundred twenty-five dollars.")
    assert bad2.verdict == "UNGROUNDED", "Injected digit must be UNGROUNDED"


def test_b3b_ordinals_and_fractions():
    # Ordinals
    res_ord = check_value("index", 3, "Select the third option.")
    assert res_ord.verdict == "GROUNDED"
    bad_ord = check_value("index", 4, "Select the third option.")
    assert bad_ord.verdict == "UNGROUNDED", "Off-by-one ordinal must be UNGROUNDED"

    # Fractions
    res_frac = check_value("ratio", 0.5, "Cut the budget in half.")
    assert res_frac.verdict == "GROUNDED"
    bad_frac = check_value("ratio", 0.75, "Cut the budget in half.")
    assert bad_frac.verdict == "UNGROUNDED", "Wrong fraction must be UNGROUNDED"


def test_b3b_unit_conversions():
    # Distance: miles -> km
    res_km = check_value("dist", 16.09, "The race distance is 10 miles.")
    assert res_km.verdict == "GROUNDED"
    bad_km = check_value("dist", 25.0, "The race distance is 10 miles.")
    assert bad_km.verdict == "UNGROUNDED", "Invalid conversion must be UNGROUNDED"

    # Temperature: C -> F
    res_temp = check_value("temp", 32.0, "Water freezing point is 0 degrees Celsius.")
    assert res_temp.verdict == "GROUNDED"
    bad_temp = check_value("temp", 45.0, "Water freezing point is 0 degrees Celsius.")
    assert bad_temp.verdict == "UNGROUNDED", "Invalid temperature must be UNGROUNDED"

    # Currency symbol
    res_curr = check_value("price", 150.0, "Total price is $150.")
    assert res_curr.verdict == "GROUNDED"
    bad_curr = check_value("price", 160.0, "Total price is $150.")
    assert bad_curr.verdict == "UNGROUNDED", "Wrong currency amount must be UNGROUNDED"
