"""Benford's Law first-digit conformity test, used here as a data-integrity
/ anomaly screen on reported financial fields (loan_amnt, annual_inc,
revol_bal), complementing the outlier findings already documented in
EDA_Report.md (e.g. dti values up to 999, annual_inc up to $11M).

Same statistical technique used in the author's own published Medium
article on Benford's Law for fraud detection, applied here to a lending
dataset rather than general financial statements.
"""
import numpy as np
import pandas as pd
from scipy.stats import chisquare
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

DIGITS = np.arange(1, 10)
EXPECTED_PROPORTIONS = np.log10(1 + 1 / DIGITS)

# Nigrini's standard MAD conformity thresholds for the first-digit test.
MAD_THRESHOLDS = [
    (0.006, "Close conformity"),
    (0.012, "Acceptable conformity"),
    (0.015, "Marginally acceptable conformity"),
]


def first_digits(series: pd.Series) -> pd.Series:
    """Extract the leading (first significant) digit of each positive value."""
    values = series.dropna()
    values = values[values > 0]
    return values.apply(lambda x: int(str(x).lstrip("0.").replace(".", "")[0]))


def mad_conformity(mad: float) -> str:
    for threshold, label in MAD_THRESHOLDS:
        if mad < threshold:
            return label
    return "Nonconformity"


def benford_test(series: pd.Series, feature_name: str = "") -> dict:
    """Run the first-digit Benford's Law test on a numeric series.

    Returns observed vs. expected digit proportions, a chi-square
    goodness-of-fit statistic and p-value, the Mean Absolute Deviation
    (MAD) and its Nigrini conformity label.
    """
    digits = first_digits(series)
    n = len(digits)
    if n == 0:
        raise ValueError(f"No positive values available to test for '{feature_name}'")

    observed_counts = digits.value_counts().reindex(DIGITS, fill_value=0).sort_index()
    observed_proportions = (observed_counts / n).values

    expected_counts = EXPECTED_PROPORTIONS * n

    chi2_stat, p_value = chisquare(f_obs=observed_counts.values, f_exp=expected_counts)

    mad = float(np.mean(np.abs(observed_proportions - EXPECTED_PROPORTIONS)))
    conformity = mad_conformity(mad)

    return {
        "feature": feature_name,
        "n": n,
        "observed_proportions": dict(zip(DIGITS.tolist(), observed_proportions.tolist())),
        "expected_proportions": dict(zip(DIGITS.tolist(), EXPECTED_PROPORTIONS.tolist())),
        "chi2_stat": float(chi2_stat),
        "p_value": float(p_value),
        "mad": mad,
        "conformity": conformity,
    }


def plot_benford(result: dict, out_path: str):
    """Bar chart of observed vs. expected first-digit distribution."""
    observed = [result["observed_proportions"][d] for d in DIGITS]
    expected = [result["expected_proportions"][d] for d in DIGITS]

    fig, ax = plt.subplots(figsize=(9, 6))
    width = 0.35
    ax.bar(DIGITS - width / 2, observed, width, label="Observed", color="steelblue")
    ax.bar(DIGITS + width / 2, expected, width, label="Expected (Benford)", color="gray", alpha=0.7)
    ax.set_xticks(DIGITS)
    ax.set_xlabel("Leading Digit")
    ax.set_ylabel("Proportion")
    ax.set_title(
        f"Benford's Law First-Digit Test: {result['feature']}\n"
        f"MAD={result['mad']:.4f} ({result['conformity']}), "
        f"chi2 p-value={result['p_value']:.4f}"
    )
    ax.legend()
    plt.tight_layout()
    plt.savefig(out_path, dpi=150)
    plt.close(fig)


def run_benford_screen(df: pd.DataFrame, features: list, out_dir: str = ".") -> pd.DataFrame:
    """Run the Benford's Law screen across multiple features and return a
    summary DataFrame. Saves one plot per feature to out_dir.
    """
    rows = []
    for feat in features:
        result = benford_test(df[feat], feature_name=feat)
        plot_benford(result, f"{out_dir}/benford_{feat}.png")
        rows.append({
            "feature": feat,
            "n": result["n"],
            "mad": round(result["mad"], 5),
            "conformity": result["conformity"],
            "chi2_p_value": round(result["p_value"], 5),
        })
    return pd.DataFrame(rows)
