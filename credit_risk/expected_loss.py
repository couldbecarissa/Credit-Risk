"""Expected Loss calculation, ported from credit_risk_model_guide.md
(Section 2 / Section 8's compute_expected_loss): EL = PD x LGD x EAD.
"""
import numpy as np
import pandas as pd


def compute_expected_loss(pd_values: np.ndarray,
                           lgd: float,
                           ead_values: np.ndarray) -> pd.DataFrame:
    """Compute Expected Loss for each loan in a portfolio.
    EL = PD x LGD x EAD.
    """
    el = pd_values * lgd * ead_values
    return pd.DataFrame({
        "PD": pd_values,
        "LGD": lgd,
        "EAD": ead_values,
        "EL": el,
    })
