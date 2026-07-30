"""Benford's Law conformity tests: a genuinely Benford-distributed sample
should score low MAD / close conformity, while a deliberately skewed
sample (all leading digit 9) should be flagged as nonconformant.
"""
import numpy as np
import pandas as pd

from credit_risk.benford import benford_test


def test_benford_conforming_distribution_scores_low_mad():
    # log-uniform values genuinely follow Benford's Law by construction.
    rng = np.random.default_rng(42)
    values = 10 ** rng.uniform(0, 6, size=20000)
    series = pd.Series(values)

    result = benford_test(series, "synthetic_conforming")

    assert result["mad"] < 0.012
    assert result["conformity"] in ("Close conformity", "Acceptable conformity")


def test_benford_nonconforming_distribution_is_flagged():
    # Every value forced to start with digit 9: maximally non-Benford.
    rng = np.random.default_rng(42)
    values = rng.uniform(9000, 9999, size=5000)
    series = pd.Series(values)

    result = benford_test(series, "synthetic_nonconforming")

    assert result["mad"] > 0.015
    assert result["conformity"] == "Nonconformity"
    assert result["observed_proportions"][9] > 0.9


def test_first_digit_extraction():
    from credit_risk.benford import first_digits

    series = pd.Series([12345, 0.0456, 987, 5.0, -3, 0])
    digits = first_digits(series)
    # Negative and zero values are dropped before digit extraction.
    assert len(digits) == 4
    assert set(digits.values) == {1, 4, 9, 5}
