"""Model training and scorecard scaling, ported from
credit_risk_model_guide.md (Sections 7-8's train_logistic_model,
build_scorecard, predict_score).
"""
import numpy as np
import pandas as pd
from sklearn.linear_model import LogisticRegression


def train_logistic_model(X_train, y_train) -> LogisticRegression:
    """Train a logistic regression model with balanced class weights to
    handle the minority (defaulter) class.
    """
    model = LogisticRegression(
        penalty="l2",
        C=1.0,
        class_weight="balanced",
        max_iter=1000,
        solver="lbfgs",
        random_state=42,
    )
    model.fit(X_train, y_train)
    return model


def build_scorecard(model: LogisticRegression, woe_feature_names: list,
                     base_score: int = 600,
                     base_odds: int = 50,
                     pdo: int = 20) -> pd.DataFrame:
    """Convert logistic regression coefficients into a points-based
    reference table (points contribution per variable at WOE=0).
    """
    factor = pdo / np.log(2)
    offset = base_score - factor * np.log(base_odds)

    n = len(woe_feature_names)
    intercept = model.intercept_[0]
    coefs = model.coef_[0]

    records = []
    for i, feat in enumerate(woe_feature_names):
        beta = coefs[i]
        points = -(beta * 0 + intercept / n) * factor + offset / n
        records.append({
            "Feature": feat,
            "Coefficient": round(beta, 4),
            "Points at WOE=0": round(points, 1),
        })

    return pd.DataFrame(records)


def predict_score(log_odds: float,
                   base_score: int = 600,
                   base_odds: int = 50,
                   pdo: int = 20) -> float:
    """Convert log-odds to a scaled credit score (higher score = lower risk)."""
    factor = pdo / np.log(2)
    offset = base_score - factor * np.log(base_odds)
    return offset + factor * (-log_odds)


def risk_tier(pd_value: float) -> str:
    """Map a predicted probability of default to a simple lending-decision
    tier, mirroring the guide's Section 1 worked examples.
    """
    if pd_value < 0.05:
        return "Low risk"
    elif pd_value < 0.15:
        return "Medium risk"
    elif pd_value < 0.30:
        return "High risk"
    return "Very high risk"
