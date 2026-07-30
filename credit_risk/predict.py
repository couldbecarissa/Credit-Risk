"""Score a single loan application using a trained artifact (model +
WOE maps), without retraining. Used by service.py.
"""
import pandas as pd

from credit_risk.model import predict_score, risk_tier


def _woe_for_value(feature_map: dict, raw_value) -> float:
    """Look up the WOE value for a single raw feature value.

    Continuous features were binned with pd.qcut, so the map keys are
    pandas Interval objects; categorical features map raw categories
    directly. Unseen bins/categories fall back to 0 (neutral, matching
    the fillna(0) behaviour used at training time).
    """
    mapping = feature_map["map"]
    if feature_map["type"] == "continuous":
        for interval, woe in mapping.items():
            try:
                if raw_value in interval:
                    return float(woe)
            except TypeError:
                continue
        return 0.0
    else:
        return float(mapping.get(raw_value, 0.0))


def score_application(raw_features: dict, artifact: dict) -> dict:
    """Compute PD, a scaled credit score, and a risk tier for one
    application, using the persisted model artifact.
    """
    model = artifact["model"]
    woe_maps = artifact["woe_maps"]
    feature_order = artifact["feature_order"]

    woe_vector = []
    for col in feature_order:
        base_feature = col[:-4] if col.endswith("_WOE") else col
        feature_map = woe_maps.get(base_feature)
        if feature_map is None:
            woe_vector.append(0.0)
            continue
        raw_value = raw_features.get(base_feature)
        woe_vector.append(_woe_for_value(feature_map, raw_value) if raw_value is not None else 0.0)

    X = pd.DataFrame([woe_vector], columns=feature_order)
    pd_value = float(model.predict_proba(X)[:, 1][0])
    log_odds = model.decision_function(X)[0]
    score = predict_score(log_odds)

    return {
        "probability_of_default": round(pd_value, 4),
        "credit_score": round(score, 1),
        "risk_tier": risk_tier(pd_value),
    }
