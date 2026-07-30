"""Train the credit risk PD model end-to-end and persist an artifact that
service.py loads at startup (train/serve separation: the service never
retrains per request).

Usage:
    python train.py                              # trains on the bundled sample
    python train.py --data path/to/full_dataset.csv
"""
import argparse
import json
import pickle
from pathlib import Path

import numpy as np
from sklearn.model_selection import train_test_split

from credit_risk.data import load_and_prepare, CONTINUOUS_FEATURES, CATEGORICAL_FEATURES
from credit_risk.features import screen_features_by_iv, apply_woe_transform, compute_vif
from credit_risk.model import train_logistic_model, build_scorecard
from credit_risk.evaluate import evaluate_model, plot_roc_curve, plot_ks_chart, compute_psi
from credit_risk.expected_loss import compute_expected_loss
from credit_risk.benford import run_benford_screen

BENFORD_FEATURES = ["loan_amnt", "annual_inc", "revol_bal"]


def main(data_path: str, out_dir: str = "reports", model_dir: str = "models"):
    Path(out_dir).mkdir(exist_ok=True)
    Path(model_dir).mkdir(exist_ok=True)

    print(f"Loading and preparing data from {data_path} ...")
    df = load_and_prepare(data_path)
    print(f"  Records: {len(df):,} | Default rate: {df['default'].mean():.2%}")

    print("\n-- Benford's Law data-integrity screen --")
    benford_summary = run_benford_screen(df, BENFORD_FEATURES, out_dir=out_dir)
    print(benford_summary.to_string(index=False))

    print("\n-- IV feature screening --")
    iv_df = screen_features_by_iv(df, CONTINUOUS_FEATURES, CATEGORICAL_FEATURES)
    print(iv_df.to_string(index=False))

    selected_cont = iv_df[(iv_df["type"] == "continuous") & (iv_df["selected"])]["feature"].tolist()
    selected_cat = iv_df[(iv_df["type"] == "categorical") & (iv_df["selected"])]["feature"].tolist()

    print("\n-- WOE transformation --")
    df_woe, woe_maps = apply_woe_transform(df, selected_cont, selected_cat)
    print(f"  WOE features created: {df_woe.shape[1] - 1}")

    X = df_woe.drop(columns="default")
    y = df_woe["default"]

    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.20, random_state=42, stratify=y
    )
    print(f"\n  Train size: {len(X_train):,} | Test size: {len(X_test):,}")

    print("\n-- VIF multicollinearity check --")
    vif_df = compute_vif(X_train)
    print(vif_df.to_string(index=False))
    high_vif = vif_df[vif_df["VIF"] > 10]["feature"].tolist()
    if high_vif:
        print(f"  Dropping high-VIF features: {high_vif}")
        X_train = X_train.drop(columns=high_vif)
        X_test = X_test.drop(columns=high_vif)

    print("\n-- Training logistic regression --")
    model = train_logistic_model(X_train, y_train)

    results = evaluate_model(model, X_test, y_test, "Test")
    print(f"\n  AUC:      {results['auc']:.4f}")
    print(f"  KS:       {results['ks']:.4f}")
    print(f"  Gini:     {results['gini']:.4f}")
    print(f"  Accuracy: {results['accuracy']:.4f}")

    plot_roc_curve(y_test, results["y_pred_prob"], results["auc"], f"{out_dir}/roc_curve.png")
    plot_ks_chart(y_test, results["y_pred_prob"], f"{out_dir}/ks_chart.png")

    scorecard = build_scorecard(model, X_train.columns.tolist())

    pd_values = model.predict_proba(X_test)[:, 1]
    ead_values = df.loc[y_test.index, "loan_amnt"].values
    el_df = compute_expected_loss(pd_values, lgd=0.75, ead_values=ead_values)
    print(f"\n  Portfolio Expected Loss / EAD: {el_df['EL'].sum() / el_df['EAD'].sum():.3%}")

    psi_val = compute_psi(model.predict_proba(X_train)[:, 1], model.predict_proba(X_test)[:, 1])
    print(f"  PSI (train vs test): {psi_val:.4f}")

    artifact = {
        "model": model,
        "woe_maps": woe_maps,
        "feature_order": X_train.columns.tolist(),
        "selected_cont": selected_cont,
        "selected_cat": selected_cat,
    }
    with open(f"{model_dir}/model.pkl", "wb") as f:
        pickle.dump(artifact, f)

    run_report = {
        "n_records": len(df),
        "default_rate": float(df["default"].mean()),
        "auc": results["auc"],
        "ks": results["ks"],
        "gini": results["gini"],
        "accuracy": results["accuracy"],
        "psi_train_vs_test": psi_val,
        "expected_loss_rate": float(el_df["EL"].sum() / el_df["EAD"].sum()),
        "benford_summary": benford_summary.to_dict(orient="records"),
    }
    with open(f"{out_dir}/run_report.json", "w") as f:
        json.dump(run_report, f, indent=2)

    print(f"\nSaved model artifact to {model_dir}/model.pkl")
    print(f"Saved plots and run report to {out_dir}/")
    print("\n-- Pipeline complete --")
    return artifact, run_report


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Train the credit risk PD model.")
    parser.add_argument("--data", default="data/sample_loans.csv",
                         help="Path to a Lending Club CSV (defaults to the bundled demo sample).")
    parser.add_argument("--out-dir", default="reports")
    parser.add_argument("--model-dir", default="models")
    args = parser.parse_args()
    main(args.data, args.out_dir, args.model_dir)
