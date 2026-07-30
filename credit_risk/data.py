"""Data loading and cleaning, ported from credit_risk_model_guide.md
(Section 8, `load_and_prepare`), adapted for the bundled demo sample.
"""
import numpy as np
import pandas as pd

MODEL_FEATURES = [
    "loan_amnt", "int_rate", "annual_inc", "dti",
    "fico_range_low", "revol_util", "revol_bal",
    "open_acc", "delinq_2yrs", "inq_last_6mths",
    "pub_rec", "emp_length", "term", "home_ownership",
    "purpose", "default",
]

CONTINUOUS_FEATURES = [
    "loan_amnt", "int_rate", "annual_inc", "dti",
    "fico_range_low", "revol_util", "revol_bal",
    "open_acc", "delinq_2yrs", "inq_last_6mths",
    "pub_rec", "emp_length", "term",
]

CATEGORICAL_FEATURES = ["home_ownership", "purpose"]

_EMP_MAP = {
    "< 1 year": 0, "1 year": 1, "2 years": 2, "3 years": 3,
    "4 years": 4, "5 years": 5, "6 years": 6, "7 years": 7,
    "8 years": 8, "9 years": 9, "10+ years": 10,
}


def load_and_prepare(path: str) -> pd.DataFrame:
    """Load a Lending Club CSV, filter to resolved loans, create the
    binary target, clean feature columns, and select the modeling set.
    """
    df = pd.read_csv(path, low_memory=False)

    df = df[df["loan_status"].isin(["Fully Paid", "Charged Off"])].copy()
    df["default"] = (df["loan_status"] == "Charged Off").astype(int)

    df = df[MODEL_FEATURES].copy()

    df["emp_length"] = df["emp_length"].map(_EMP_MAP)

    df["term"] = df["term"].astype(str).str.strip().str.replace(" months", "", regex=False)
    df["term"] = pd.to_numeric(df["term"], errors="coerce")

    if df["int_rate"].dtype == object:
        df["int_rate"] = df["int_rate"].astype(str).str.replace("%", "", regex=False)
        df["int_rate"] = pd.to_numeric(df["int_rate"], errors="coerce")

    for col in ["annual_inc", "revol_bal", "loan_amnt"]:
        cap = df[col].quantile(0.999)
        df[col] = df[col].clip(upper=cap)

    num_cols = df.select_dtypes(include=[np.number]).columns
    df[num_cols] = df[num_cols].fillna(df[num_cols].median())

    cat_cols = df.select_dtypes(include="object").columns
    for col in cat_cols:
        df[col] = df[col].fillna(df[col].mode()[0])

    return df
