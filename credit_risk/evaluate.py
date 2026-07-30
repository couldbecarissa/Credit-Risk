"""Model evaluation metrics and plots, ported from
credit_risk_model_guide.md (Sections 5, 8, and 14's compute_ks_statistic,
compute_gini, evaluate_model, plot_roc_curve, plot_ks_chart, compute_psi).
"""
import numpy as np
import matplotlib
matplotlib.use("Agg")  # headless-safe for service/container use
import matplotlib.pyplot as plt
from sklearn.metrics import roc_auc_score, roc_curve, classification_report, accuracy_score


def compute_ks_statistic(y_true: np.ndarray, y_scores: np.ndarray) -> tuple:
    """KS = maximum separation between the cumulative distributions of
    defaulters and non-defaulters, across score thresholds.
    Returns (ks_stat, score_threshold_at_ks).
    """
    order = np.argsort(-y_scores)
    y_true_s = y_true[order]
    y_scores_s = y_scores[order]

    n_events = y_true.sum()
    n_non_events = len(y_true) - n_events

    cum_events = np.cumsum(y_true_s) / n_events
    cum_non_events = np.cumsum(1 - y_true_s) / n_non_events

    ks_values = np.abs(cum_events - cum_non_events)
    idx = np.argmax(ks_values)

    return ks_values[idx], y_scores_s[idx]


def compute_gini(auc: float) -> float:
    """Gini coefficient: Gini = 2 x AUC - 1."""
    return 2 * auc - 1


def evaluate_model(model, X_test, y_test, dataset_name: str = "Test") -> dict:
    """Comprehensive model evaluation: AUC, KS, Gini, accuracy, and a
    classification report at the default 0.5 threshold.
    """
    y_pred_prob = model.predict_proba(X_test)[:, 1]
    y_pred = model.predict(X_test)

    auc = roc_auc_score(y_test, y_pred_prob)
    ks, ks_t = compute_ks_statistic(y_test.values, y_pred_prob)
    gini = compute_gini(auc)
    accuracy = accuracy_score(y_test, y_pred)
    report = classification_report(
        y_test, y_pred, target_names=["Non-Default", "Default"], output_dict=True
    )

    return {
        "dataset": dataset_name,
        "auc": auc,
        "ks": ks,
        "ks_threshold": ks_t,
        "gini": gini,
        "accuracy": accuracy,
        "classification_report": report,
        "y_pred_prob": y_pred_prob,
    }


def plot_roc_curve(y_test, y_pred_prob, auc, out_path: str, title: str = "ROC Curve"):
    fpr, tpr, _ = roc_curve(y_test, y_pred_prob)

    fig, ax = plt.subplots(figsize=(8, 7))
    ax.plot(fpr, tpr, color="steelblue", lw=2, label=f"Model (AUC = {auc:.4f})")
    ax.plot([0, 1], [0, 1], "k--", lw=1.5, label="Random Classifier (AUC = 0.50)")
    ax.fill_between(fpr, tpr, alpha=0.15, color="steelblue")
    ax.set_xlabel("False Positive Rate (1 - Specificity)")
    ax.set_ylabel("True Positive Rate (Sensitivity / Recall)")
    ax.set_title(title)
    ax.legend(loc="lower right")
    plt.tight_layout()
    plt.savefig(out_path, dpi=150)
    plt.close(fig)


def plot_ks_chart(y_test, y_pred_prob, out_path: str, title: str = "KS Chart"):
    order = np.argsort(-y_pred_prob)
    y_true_s = y_test.values[order]
    n = len(y_true_s)
    n_events = y_true_s.sum()
    n_non_events = n - n_events

    cum_events = np.cumsum(y_true_s) / n_events
    cum_non_events = np.cumsum(1 - y_true_s) / n_non_events
    ks_values = np.abs(cum_events - cum_non_events)
    ks_idx = np.argmax(ks_values)
    pct_pop = np.arange(1, n + 1) / n * 100

    fig, ax = plt.subplots(figsize=(10, 6))
    ax.plot(pct_pop, cum_events * 100, color="crimson", lw=2, label="Cumulative Defaulters (%)")
    ax.plot(pct_pop, cum_non_events * 100, color="steelblue", lw=2, label="Cumulative Non-Defaulters (%)")
    ax.axvline(x=pct_pop[ks_idx], color="gray", linestyle="--", lw=1.5)
    ax.set_xlabel("% of Population (sorted by descending risk score)")
    ax.set_ylabel("Cumulative %")
    ax.set_title(title)
    ax.legend()
    plt.tight_layout()
    plt.savefig(out_path, dpi=150)
    plt.close(fig)


def compute_psi(expected: np.ndarray, actual: np.ndarray, bins: int = 10) -> float:
    """Population Stability Index between a development (expected) and
    current (actual) score distribution.
    PSI < 0.10 stable, 0.10-0.25 monitor, > 0.25 significant drift.
    """
    _, bin_edges = np.histogram(expected, bins=bins)
    bin_edges[0] = -np.inf
    bin_edges[-1] = np.inf

    exp_counts, _ = np.histogram(expected, bins=bin_edges)
    act_counts, _ = np.histogram(actual, bins=bin_edges)

    exp_pct = exp_counts / len(expected)
    act_pct = act_counts / len(actual)

    eps = 1e-4
    exp_pct = np.where(exp_pct == 0, eps, exp_pct)
    act_pct = np.where(act_pct == 0, eps, act_pct)

    return float(np.sum((act_pct - exp_pct) * np.log(act_pct / exp_pct)))
