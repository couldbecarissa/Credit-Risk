"""predict_score() regression test against the PD -> score worked table
in credit_risk_model_guide.md, Section 7 ("Score Examples at Different PDO").
"""
import numpy as np
import pytest

from credit_risk.model import predict_score, risk_tier


def _logit(pd_value: float) -> float:
    return np.log(pd_value / (1 - pd_value))


# (probability_of_default, expected_score) pairs, straight from the guide.
GUIDE_TABLE = [
    (0.0196, 600),
    (0.0099, 620),
    (0.0376, 580),
    (0.0741, 560),
]


@pytest.mark.parametrize("pd_value,expected_score", GUIDE_TABLE)
def test_predict_score_matches_guide_worked_table(pd_value, expected_score):
    score = predict_score(_logit(pd_value))
    assert score == pytest.approx(expected_score, abs=1.0)


def test_score_decreases_as_pd_increases():
    low_risk_score = predict_score(_logit(0.01))
    high_risk_score = predict_score(_logit(0.30))
    assert low_risk_score > high_risk_score


def test_risk_tier_boundaries():
    assert risk_tier(0.01) == "Low risk"
    assert risk_tier(0.10) == "Medium risk"
    assert risk_tier(0.20) == "High risk"
    assert risk_tier(0.50) == "Very high risk"
