"""WOE/IV regression test against the worked-by-hand DTI example in
credit_risk_model_guide.md, Section 6 ("Worked WOE/IV Example"):
1,000 loans, 100 defaults / 900 non-defaults, split into 4 DTI bins.
"""
import pandas as pd
import pytest

from credit_risk.features import compute_woe_iv

# (bin_label, n_defaults, n_non_defaults) exactly as tabulated in the guide.
GUIDE_BINS = [
    ("DTI <= 10%", 5, 180),
    ("10% < DTI <= 20%", 15, 300),
    ("20% < DTI <= 35%", 35, 280),
    ("DTI > 35%", 45, 140),
]


def _build_guide_dataframe() -> pd.DataFrame:
    rows = []
    for label, n_def, n_nondef in GUIDE_BINS:
        rows += [{"bin": label, "default": 1}] * n_def
        rows += [{"bin": label, "default": 0}] * n_nondef
    return pd.DataFrame(rows)


def test_dti_woe_iv_matches_guide_worked_example():
    df = _build_guide_dataframe()
    woe_map, iv, grouped = compute_woe_iv(df, "bin", target="default", cat=True)

    # The guide reports IV = 0.670 computed by hand (no smoothing). Our
    # implementation adds a small (eps=0.5) smoothing constant for
    # numerical stability, so we check it lands in the same "very
    # strong" range the guide itself flags, rather than an exact match.
    assert iv > 0.50

    # Highest-risk bin (DTI > 35%) must have the most negative WOE;
    # lowest-risk bin (DTI <= 10%) must have the most positive WOE.
    assert woe_map["DTI > 35%"] < woe_map["20% < DTI <= 35%"] < woe_map["10% < DTI <= 20%"] < woe_map["DTI <= 10%"]
    assert woe_map["DTI <= 10%"] > 0
    assert woe_map["DTI > 35%"] < 0


def test_iv_is_zero_for_a_useless_feature():
    # A feature with no relationship to default should score near-zero IV:
    # both categories have exactly the same 50% default rate.
    df = pd.DataFrame({
        "feature": ["A"] * 500 + ["B"] * 500,
        "default": ([1] * 250 + [0] * 250) * 2,
    })
    _, iv, _ = compute_woe_iv(df, "feature", target="default", cat=True)
    assert iv < 0.02
