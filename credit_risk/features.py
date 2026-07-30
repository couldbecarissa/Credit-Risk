"""WOE/IV feature engineering and multicollinearity checks, ported from
credit_risk_model_guide.md (Sections 6 and 8's compute_woe_iv,
screen_features_by_iv, apply_woe_transform, compute_vif).
"""
import numpy as np
import pandas as pd
from sklearn.linear_model import LinearRegression


def compute_woe_iv(df: pd.DataFrame, feature: str,
                    target: str = "default",
                    bins: int = 10,
                    cat: bool = False) -> tuple:
    """Compute Weight of Evidence (WOE) and Information Value (IV) for a
    single feature. Returns (woe_map, iv, grouped_stats_df).
    """
    total_events = df[target].sum()
    total_non_events = len(df) - total_events

    if cat:
        grouped = df.groupby(feature)[target].agg(["sum", "count"])
    else:
        df = df.copy()
        df["__bin__"] = pd.qcut(df[feature], q=bins, duplicates="drop")
        grouped = df.groupby("__bin__", observed=False)[target].agg(["sum", "count"])

    grouped.columns = ["events", "total"]
    grouped["non_events"] = grouped["total"] - grouped["events"]

    eps = 0.5
    grouped["events"] = grouped["events"] + eps
    grouped["non_events"] = grouped["non_events"] + eps

    grouped["dist_events"] = grouped["events"] / total_events
    grouped["dist_non_events"] = grouped["non_events"] / total_non_events

    grouped["woe"] = np.log(grouped["dist_non_events"] / grouped["dist_events"])
    grouped["iv"] = (grouped["dist_non_events"] - grouped["dist_events"]) * grouped["woe"]

    iv = grouped["iv"].sum()
    woe_map = grouped["woe"].to_dict()

    return woe_map, iv, grouped


def screen_features_by_iv(df: pd.DataFrame,
                           cont_features: list,
                           cat_features: list,
                           target: str = "default",
                           min_iv: float = 0.02) -> pd.DataFrame:
    """Compute IV for all features and return a ranked DataFrame. Features
    with IV < min_iv are flagged as not selected.
    """
    results = []
    for feat in cont_features:
        try:
            _, iv, _ = compute_woe_iv(df, feat, target, bins=10, cat=False)
            results.append({"feature": feat, "iv": iv, "type": "continuous"})
        except Exception:
            pass

    for feat in cat_features:
        try:
            _, iv, _ = compute_woe_iv(df, feat, target, bins=10, cat=True)
            results.append({"feature": feat, "iv": iv, "type": "categorical"})
        except Exception:
            pass

    iv_df = pd.DataFrame(results).sort_values("iv", ascending=False)

    def categorize_iv(iv):
        if iv < 0.02:
            return "Useless"
        elif iv < 0.10:
            return "Weak"
        elif iv < 0.30:
            return "Medium"
        elif iv < 0.50:
            return "Strong"
        else:
            return "Very Strong (check leakage)"

    iv_df["strength"] = iv_df["iv"].apply(categorize_iv)
    iv_df["selected"] = iv_df["iv"] >= min_iv
    return iv_df


def apply_woe_transform(df: pd.DataFrame,
                         cont_features: list,
                         cat_features: list,
                         target: str = "default") -> tuple:
    """Apply WOE transformation to all features. Returns (df_woe, woe_maps)."""
    df_woe = df[[target]].copy()
    woe_maps = {}

    for feat in cont_features:
        try:
            woe_map, _, _ = compute_woe_iv(df, feat, target, bins=10, cat=False)
            bins_series = pd.qcut(df[feat], q=10, duplicates="drop", retbins=False)
            df_woe[feat + "_WOE"] = bins_series.map(woe_map).astype(float)
            woe_maps[feat] = {"type": "continuous", "map": woe_map}
        except Exception:
            pass

    for feat in cat_features:
        try:
            woe_map, _, _ = compute_woe_iv(df, feat, target, bins=10, cat=True)
            df_woe[feat + "_WOE"] = df[feat].map(woe_map).astype(float)
            woe_maps[feat] = {"type": "categorical", "map": woe_map}
        except Exception:
            pass

    df_woe.dropna(axis=1, how="all", inplace=True)
    df_woe.fillna(0, inplace=True)

    return df_woe, woe_maps


def compute_vif(X: pd.DataFrame) -> pd.DataFrame:
    """Compute Variance Inflation Factor for all features. VIF > 10 signals
    severe multicollinearity.
    """
    vif_data = []
    cols = X.columns.tolist()

    for col in cols:
        others = [c for c in cols if c != col]
        model = LinearRegression().fit(X[others], X[col])
        r2 = model.score(X[others], X[col])
        vif = 1 / (1 - r2) if r2 < 1.0 else np.inf
        vif_data.append({"feature": col, "VIF": round(vif, 2)})

    return pd.DataFrame(vif_data).sort_values("VIF", ascending=False)
