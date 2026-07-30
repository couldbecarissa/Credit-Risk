# Credit Risk Modeling: Estimating Probability of Default
### From Zero to Scorecard — A Quantitative Finance Tutorial

---

> **How to use this guide:** Read sequentially. Every section builds on the last.
> Concepts are introduced with real-world intuition first, then mathematics, then code.
> Numerical examples are worked by hand so you can verify your own model outputs.
> By the end, you will be able to build, evaluate, and interpret a production-grade credit risk model.

---

## Table of Contents

1. [What is Credit Risk? — The Real-World Intuition](#1-what-is-credit-risk--the-real-world-intuition)
2. [The Three Pillars: PD, LGD, and EAD](#2-the-three-pillars-pd-lgd-and-ead)
3. [The Basel Framework — Why It Matters](#3-the-basel-framework--why-it-matters)
4. [What Does "Default" Actually Mean?](#4-what-does-default-actually-mean)
5. [Understanding the Dataset](#5-understanding-the-dataset)
6. [Feature Engineering — WOE and Information Value](#6-feature-engineering--woe-and-information-value)
7. [Logistic Regression — The Industry Standard](#7-logistic-regression--the-industry-standard)
8. [The Full Python Implementation](#8-the-full-python-implementation)
9. [Line-by-Line Code Explanation](#9-line-by-line-code-explanation)
10. [Feature Analysis — Identifying Key Risk Drivers](#10-feature-analysis--identifying-key-risk-drivers)
11. [Model Performance — ROC, KS, and Gini](#11-model-performance--roc-ks-and-gini)
12. [Numerical Examples — Worked by Hand](#12-numerical-examples--worked-by-hand)
13. [Business and Risk Interpretation](#13-business-and-risk-interpretation)
14. [Model Validation — PSI, OOT, and Vintage Analysis](#14-model-validation--psi-oot-and-vintage-analysis)
15. [Exercises and Further Exploration](#15-exercises-and-further-exploration)
16. [References and Further Reading](#16-references-and-further-reading)

---

## 1. What is Credit Risk? — The Real-World Intuition

Before we write a single line of Python, we need to understand the business problem we are solving.

### The Lending Analogy

Imagine you are a bank. A customer walks in and asks to borrow **$20,000** to consolidate their credit card debt. Your job is to decide: *should you lend this person money?*

You cannot know for certain whether they will repay. But you are not flying blind either. You have information: their income, their past payment history, how much other debt they already carry, how long they have been employed. All of this is evidence — evidence that helps you estimate the **probability** that this particular borrower will or will not repay.

This is exactly what a **credit risk model** does. It ingests all the available evidence about a borrower and outputs a single number: the **Probability of Default (PD)** — the chance that this person will fail to repay within the next 12 months.

### Two Scenarios

**Scenario A — Good Borrower:**
Maria earns $85,000/year, has a FICO score of 740, has never missed a payment, and carries a debt-to-income ratio of 18%. Your model outputs PD = 1.2%. This means there is a 1.2% chance she defaults. The bank approves the loan.

**Scenario B — Risky Borrower:**
James earns $32,000/year, has a FICO score of 560, has two 30-day delinquencies in the past year, and carries a debt-to-income ratio of 52%. Your model outputs PD = 23.5%. The bank declines the loan or offers it at a significantly higher interest rate to compensate for the risk.

The credit risk model does not make the final decision — it provides a rigorous, data-driven estimate that *informs* the decision. This is the distinction between intuition-based lending (which led to the 2008 financial crisis) and model-based lending (the modern approach).

### Why Does This Matter at Scale?

A single bank may hold **millions** of loans in its portfolio. It is physically impossible for a human analyst to assess each borrower individually. A credit risk model allows banks to:

- Make consistent, objective lending decisions at scale
- Price loans correctly (higher risk = higher interest rate)
- Quantify how much money they expect to lose (provisioning)
- Satisfy regulators that they hold enough capital to absorb losses

---

## 2. The Three Pillars: PD, LGD, and EAD

Credit risk is not a single number — it is a framework built on three fundamental components. Think of them as the three dimensions of the same problem.

### Probability of Default (PD)

**Definition:** The likelihood that a borrower will fail to make required payments within a defined time horizon — typically **one year**.

- PD is expressed as a number between 0 and 1 (or equivalently, 0% to 100%)
- A PD of 0.03 means there is a 3% chance the borrower defaults in the next 12 months
- PD is estimated at the **individual borrower level** using historical data and statistical models
- Under Basel III, the regulatory minimum floor for PD is 0.03% — even the safest borrowers carry some nonzero risk

> **This guide focuses on estimating PD.** It is the most model-intensive component, requiring the richest data and the most sophisticated statistical machinery.

### Loss Given Default (LGD)

**Definition:** The fraction of the outstanding loan amount that the bank will lose *if* the borrower defaults.

```
LGD = 1 - Recovery Rate
```

Even when a borrower defaults, the bank does not necessarily lose everything. It can:
- Seize and sell collateral (e.g., repossess a car, foreclose on a house)
- Pursue legal recovery through a debt collection process
- Sell the delinquent loan to a debt collector at a discount

**Typical LGD Ranges:**

| Loan Type | Typical LGD | Why |
|---|---|---|
| Secured mortgage (low LTV) | 10–25% | Collateral covers most of the loan |
| Secured auto loan | 30–50% | Car depreciates; recovery is partial |
| Unsecured personal loan | 60–85% | No collateral; collection is expensive |
| Credit card | 70–90% | No collateral; high workout costs |
| Senior secured corporate | 25–45% | Assets available; legal priority |

**Example:**
A bank has a $100,000 unsecured personal loan. The borrower defaults. After spending $5,000 on debt collection, the bank recovers $25,000 from the borrower's assets. LGD = 1 - ($25,000 - $5,000) / $100,000 = 1 - 0.20 = **80%**.

### Exposure at Default (EAD)

**Definition:** The total outstanding amount the bank is exposed to at the *moment* the borrower defaults.

For a simple term loan (e.g., a mortgage or car loan), EAD is just the remaining balance on the loan at the time of default. For revolving credit products like credit cards, it is trickier — borrowers tend to draw down their credit lines *more* as they approach default, so EAD must account for this:

```
EAD = Current Outstanding Balance + Credit Conversion Factor (CCF) × Undrawn Commitment
```

**Example:**
A credit card has a $15,000 limit. The borrower currently has $6,000 outstanding. Historical data shows that borrowers use an additional 60% of their undrawn balance before defaulting (CCF = 0.60).

```
EAD = $6,000 + 0.60 × ($15,000 - $6,000)
    = $6,000 + 0.60 × $9,000
    = $6,000 + $5,400
    = $11,400
```

### The Expected Loss Formula — Bringing It All Together

The three pillars combine into the single most important formula in credit risk:

```
Expected Loss (EL) = PD × LGD × EAD
```

This is the **average amount of money** the bank expects to lose on a given loan over the next year. It is the conceptual foundation for:
- **Loan pricing:** the credit spread on a loan should at minimum cover EL
- **Provisioning:** the bank must set aside EL as a provision (accounting reserve) under IFRS 9
- **Capital allocation:** the bank holds capital against losses *above* EL (unexpected losses)

**Full Worked Example:**

| Loan | PD | LGD | EAD | Expected Loss |
|---|---|---|---|---|
| Maria's personal loan | 1.2% | 75% | $20,000 | 0.012 × 0.75 × $20,000 = **$180** |
| James's personal loan | 23.5% | 75% | $20,000 | 0.235 × 0.75 × $20,000 = **$3,525** |

Maria's loan has an EL of $180/year. James's loan has an EL of $3,525/year — nearly **20 times** more. No wonder the bank declines James or charges him a much higher rate.

---

## 3. The Basel Framework — Why It Matters

If you work in finance, you will inevitably hear about "Basel." Understanding what it is and why it matters puts the credit risk model in its proper regulatory context.

### The Big Picture

The **Basel Accords** are international banking regulations developed by the Basel Committee on Banking Supervision (BCBS), housed at the Bank for International Settlements (BIS) in Basel, Switzerland. Their core purpose is to ensure that banks hold enough capital to survive large credit losses without requiring taxpayer bailouts.

The central idea: the riskier a bank's loan portfolio, the more capital it must hold as a buffer.

### Three Generations of Basel

**Basel I (1988):** Crude and simple. All corporate loans received a 100% risk weight regardless of borrower quality. A AAA-rated Microsoft loan and a CCC-rated junk bond loan both required the same capital. No incentive to accurately measure risk.

**Basel II (2004):** Introduced risk-sensitive capital requirements. Banks could choose between three approaches of increasing sophistication:

| Approach | Who Estimates PD? | Who Estimates LGD? | Who Estimates EAD? |
|---|---|---|---|
| Standardized Approach (SA) | Regulator (external ratings) | Regulator | Regulator |
| Foundation IRB (F-IRB) | **Bank** | Regulator | Regulator |
| Advanced IRB (A-IRB) | **Bank** | **Bank** | **Bank** |

*IRB = Internal Ratings-Based.* Under IRB, banks build their own statistical models — like the one we are building in this guide — to estimate PD. This is where the science of credit risk modeling truly matters.

**Basel III / IV (2010–2028):** Tightened capital requirements, introduced leverage ratios, added liquidity requirements (LCR, NSFR), and introduced output floors to prevent IRB banks from gaming the system by estimating unrealistically low PD/LGD values.

### The Capital Requirement Formula

Under Advanced IRB, the risk-weighted assets (RWA) for a corporate loan depend on PD, LGD, EAD, and maturity through the **Vasicek ASRF formula** (Asymptotic Single Risk Factor model):

```
K = LGD × N[ (N⁻¹(PD) + √R × N⁻¹(0.999)) / √(1-R) ] − PD × LGD

RWA = K × 12.5 × EAD

Minimum Capital = 8% × RWA
```

Where:
- `N(·)` = standard normal CDF
- `N⁻¹(·)` = inverse standard normal CDF
- `R` = asset correlation (a function of PD; higher for lower-PD, investment-grade borrowers)
- `0.999` = the 99.9th percentile — capital must absorb a 1-in-1,000-year loss event

> **Key insight:** The model targets the **99.9% confidence level**. The bank's capital is designed to survive losses that occur with only a 0.1% annual probability. This is why accurate PD estimation is not just an academic exercise — it directly determines how much capital a bank must lock up.

---

## 4. What Does "Default" Actually Mean?

This seems like a simple question, but the regulatory definition is precise and important. The label we use to train our model depends entirely on how we define the target variable.

### The Basel / EBA Definition (Article 178, CRR)

A default is triggered when **at least one** of the following conditions is met:

**Condition 1 — Quantitative (Objective):**
The borrower is more than **90 days past due** (DPD) on any material credit obligation to the bank. In other words, they have not made a required payment for 3+ months.

**Condition 2 — Qualitative ("Unlikely to Pay"):**
The bank concludes it is unlikely that the borrower will repay in full, regardless of the DPD status. Indicators include:
- The loan is placed on non-accrual status (interest stops being recognized as income)
- A credit-related write-down or specific provision has been taken
- The bank sells the loan at a material economic loss
- The borrower files for bankruptcy
- Distressed restructuring: debt forgiveness or rescheduling granted due to financial distress

The 90-day rule is the **backstop** — even if the bank believes the borrower will eventually pay, it must classify them as defaulted once 90 DPD is breached.

### What This Means for Our Target Variable

In the **Lending Club** dataset, we define our binary target variable as:

```
loan_status = "Charged Off"  →  Default  (Y = 1)
loan_status = "Fully Paid"   →  Non-Default  (Y = 0)
```

"Charged Off" means Lending Club has given up on collecting the loan (written off as a loss) — equivalent to the "unlikely to pay" criterion. Loans that are still current or only mildly late are excluded from the training data to avoid ambiguous labels.

Under **IFRS 9** (the accounting standard), the same concept drives the three-stage ECL model:
- **Stage 1:** No significant increase in credit risk → 12-month ECL
- **Stage 2:** Significant Increase in Credit Risk (SICR) → Lifetime ECL
- **Stage 3:** Credit-impaired (defaulted) → Lifetime ECL

Our PD model directly feeds into the Stage 1 and Stage 2 provisioning calculations.

---

## 5. Understanding the Dataset

### The Lending Club Dataset

We will build our model using the **Lending Club** loan dataset, one of the most widely studied public credit datasets. Lending Club was a peer-to-peer lending platform that published detailed borrower and loan data.

**Dataset Characteristics:**
- ~2.9 million records (2007–2020 originations)
- 141 features (we will use a curated subset)
- Target: Binary — Fully Paid (0) vs. Charged Off (1)
- Typical default rate: ~18–22% of completed loans
- Available at: [https://www.kaggle.com/datasets/wordsforthewise/lending-club](https://www.kaggle.com/datasets/wordsforthewise/lending-club)

> **Note on class imbalance:** The ~20% default rate in Lending Club is higher than the typical 2–5% seen in prime bank portfolios. This is because Lending Club served near-prime and subprime borrowers. In a prime bank portfolio, you would see more severe imbalance. We will address this in the model.

### Key Features — Our Risk Drivers

Here are the features we will use, grouped by category:

**Borrower Financial Profile:**

| Feature | Description | Risk Direction |
|---|---|---|
| `annual_inc` | Annual income (USD) | Higher income → lower risk |
| `dti` | Debt-to-Income ratio (%) | Higher DTI → higher risk |
| `revol_util` | Revolving credit utilization (%) | Higher utilization → higher risk |
| `emp_length` | Employment length (years) | Longer tenure → lower risk |

**Credit History:**

| Feature | Description | Risk Direction |
|---|---|---|
| `fico_range_low` | FICO score (lower bound of range) | Higher FICO → lower risk |
| `delinq_2yrs` | Number of 30+ DPD delinquencies in past 2 years | More delinquencies → higher risk |
| `inq_last_6mths` | Number of credit inquiries in last 6 months | More inquiries → higher risk |
| `pub_rec` | Number of public records (bankruptcies) | More records → higher risk |
| `open_acc` | Number of open credit accounts | Moderate open accounts → lower risk |
| `revol_bal` | Total revolving balance outstanding | Higher balance → higher risk |
| `mths_since_last_delinq` | Months since last delinquency | More recent delinquency → higher risk |

**Loan Characteristics:**

| Feature | Description | Risk Direction |
|---|---|---|
| `loan_amnt` | Requested loan amount (USD) | Higher amount → slightly higher risk |
| `int_rate` | Interest rate (%) | Higher rate → richer proxy for risk |
| `term` | Loan term (36 or 60 months) | 60-month → higher risk |
| `purpose` | Loan purpose (debt_consolidation, medical, etc.) | Varies by purpose |
| `home_ownership` | RENT / OWN / MORTGAGE | MORTGAGE typically lower risk |

### Why `int_rate` Is a Special Case

Interest rate is the lender's own assessment of risk — it already encodes significant information about default probability. Including it in a PD model can be powerful but risks **circularity** (the rate was set partly based on a prior risk assessment). In practice, many modelers include it for pure predictive performance; others exclude it for interpretability or fairness reasons. We will include it with a note.

### Exploratory Data Analysis (EDA) Snapshot

Before modeling, always understand your data:

```python
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns

df = pd.read_csv('lending_club.csv', low_memory=False)

# Filter to completed loans only
df = df[df['loan_status'].isin(['Fully Paid', 'Charged Off'])].copy()

# Create binary target
df['default'] = (df['loan_status'] == 'Charged Off').astype(int)

print(f"Total records: {len(df):,}")
print(f"Default rate: {df['default'].mean():.2%}")
print(f"\nClass distribution:")
print(df['default'].value_counts())
```

**Typical Output:**
```
Total records: 1,345,887
Default rate: 21.43%

Class distribution:
0    1,057,222  (Fully Paid)
1      288,665  (Charged Off)
```

---

## 6. Feature Engineering — WOE and Information Value

This section covers the most important — and most misunderstood — step in credit scorecard development: **Weight of Evidence (WOE) transformation** and **Information Value (IV)** screening.

### Why Do We Need Special Feature Engineering?

Logistic regression assumes a **linear relationship** between each predictor and the log-odds of default. But in credit risk, these relationships are almost never linear:

- A FICO score of 600 vs. 700 is a huge difference in risk
- A FICO score of 800 vs. 850 is almost no difference
- A DTI of 10% vs. 30% is moderate risk increase
- A DTI of 30% vs. 50% is a much more severe increase

Simply feeding raw FICO scores into logistic regression will miss these non-linearities. WOE transformation solves this by converting each feature's bins into values that **directly measure their relationship with default risk**.

### Step 1 — Binning

Before computing WOE, we divide each continuous feature into bins (intervals). For a categorical feature, each category is already a bin.

**Methods for binning:**
- **Equal-frequency binning:** Each bin contains approximately the same number of observations (e.g., deciles). Simple and robust.
- **Optimal (supervised) binning:** Maximizes the IV while maintaining monotonicity of WOE across bins. Implemented by the `optbinning` Python package. This is the industry standard.

**Monotonicity Constraint:** WOE values should be monotonically increasing or decreasing across bins. This enforces business logic — if your model says "more debt is better," something is wrong. Monotonicity also satisfies regulatory interpretability requirements.

### Step 2 — Computing Weight of Evidence (WOE)

For each bin `i` of a feature:

```
WOE_i = ln( P(Non-Default in bin i) / P(Default in bin i) )
       = ln( (Non-Events_i / Total Non-Events) / (Events_i / Total Events) )
```

Where:
- **Events** = defaults (Y = 1)
- **Non-Events** = non-defaults (Y = 0)
- The ratio inside the log is: "how many non-defaulters are in this bin relative to defaulters?"

**Interpretation:**
- **WOE > 0:** This bin has proportionally more non-defaulters → **lower risk bin**
- **WOE < 0:** This bin has proportionally more defaulters → **higher risk bin**
- **WOE = 0:** This bin mirrors the overall default rate exactly — no discriminating power
- Larger absolute WOE values indicate stronger discrimination

### Step 3 — Computing Information Value (IV)

IV aggregates the discriminating power of a feature across **all** its bins:

```
IV = Σᵢ [ (Non-Events_i% − Events_i%) × WOE_i ]
```

where the percentages are computed relative to their respective totals. The IV is always non-negative.

**IV Interpretation (Industry-Standard Thresholds):**

| IV Range | Predictive Power | Action |
|---|---|---|
| < 0.02 | Useless | Remove from model |
| 0.02 – 0.10 | Weak | Use with caution |
| 0.10 – 0.30 | Medium | Good candidate |
| 0.30 – 0.50 | Strong | Strong predictor |
| > 0.50 | Suspicious | Check for data leakage |

> **Data Leakage Warning:** If a feature has IV > 0.50, be suspicious. It may contain future information that would not be available at the time of loan application. For example, `total_pymnt` (total amount paid) is known *after* loan outcome — including it would be leakage.

### Worked WOE/IV Example

Suppose we have 1,000 loans: **100 defaults** (events) and **900 non-defaults** (non-events). We bin the DTI feature into four bins:

| Bin | # Defaults | # Non-Defaults | Events% | Non-Events% | WOE | (NE% - E%) × WOE |
|---|---|---|---|---|---|---|
| DTI ≤ 10% | 5 | 180 | 5/100 = 5.0% | 180/900 = 20.0% | ln(20.0/5.0) = **1.386** | (0.20-0.05)×1.386 = **0.208** |
| 10% < DTI ≤ 20% | 15 | 300 | 15.0% | 33.3% | ln(33.3/15.0) = **0.799** | (0.333-0.15)×0.799 = **0.146** |
| 20% < DTI ≤ 35% | 35 | 280 | 35.0% | 31.1% | ln(31.1/35.0) = **−0.118** | (0.311-0.35)×(−0.118) = **0.005** |
| DTI > 35% | 45 | 140 | 45.0% | 15.6% | ln(15.6/45.0) = **−1.058** | (0.156-0.45)×(−1.058) = **0.311** |
| **Total** | **100** | **900** | **100%** | **100%** | | **IV = 0.670** |

**Reading the results:**
- DTI ≤ 10%: WOE = +1.386 → Very low-risk bin (lots of non-defaulters here)
- DTI > 35%: WOE = −1.058 → High-risk bin (lots of defaulters here)
- IV = 0.670 → Extremely strong predictor (but flag for leakage check; DTI this powerful suggests it may contain origination-time model information)

After WOE transformation, the feature `DTI_WOE` will contain the WOE value corresponding to each borrower's DTI bin. The logistic regression will then operate on these WOE values.

---

## 7. Logistic Regression — The Industry Standard

### Why Not Use Neural Networks or Gradient Boosting?

This is the first question every data scientist asks. The answer has multiple dimensions:

**1. Regulatory Interpretability:** Basel II/III IRB requirements and the Federal Reserve's SR 11-7 guidance both demand that models be transparent — meaning a risk officer can explain *why* a specific prediction was made. Logistic regression coefficients directly map to risk contributions. "A DTI above 35% increases log-odds of default by 0.85" is a statement regulators accept. "The 347th weight in the neural network's third hidden layer contributes positively" is not.

**2. Scorecard Additivity:** WOE-based logistic regression produces a **credit scorecard** — a table showing exactly how many points each bin of each variable contributes. This is auditable, explainable to customers (for adverse action notices under ECOA/FCRA), and stable over time.

**3. Stability and Deployment Horizon:** Credit scorecards are deployed for 3–5 years without retraining. Logistic regression is far less prone to overfitting than complex models, degrading more gracefully as the population drifts over time.

**4. Calibration:** The logistic function naturally outputs a number between 0 and 1 that is directly interpretable as a probability — no post-hoc calibration required (unlike most tree-based models).

### The Logistic Function

Logistic regression models the binary outcome (1 = default, 0 = non-default) as a function of predictors through the **sigmoid (logistic) function**:

```
P(Default) = 1 / (1 + e^(−z))

where z = β₀ + β₁X₁ + β₂X₂ + ... + βₙXₙ
```

Here, `z` is called the **log-odds** or **logit**. Let us understand why.

If `p` is the probability of default, then:

```
odds = p / (1 − p)             [e.g., if p = 0.25, odds = 0.25/0.75 = 1/3 = "1 in 3"]

logit(p) = ln(odds) = ln(p / (1−p)) = β₀ + β₁X₁ + ... + βₙXₙ
```

The logit is the **log-odds** of default, and logistic regression models it as a linear function of the predictors. This is the critical bridge: WOE values are themselves transformations of log-odds, making WOE + logistic regression a natural and mathematically coherent pairing.

### Interpreting Coefficients in Credit Context

For a coefficient `βⱼ` associated with the WOE-transformed feature `Xⱼ`:

- **βⱼ > 0:** Higher WOE (lower-risk bin) reduces predicted default → coefficient should be **negative** for WOE-encoded features (counterintuitive but correct — WOE is defined from the non-defaulter perspective)
- **|βⱼ|:** The magnitude of the coefficient determines how strongly this feature influences the prediction
- **Odds Ratio = exp(βⱼ):** A one-unit increase in `Xⱼ` multiplies the default odds by `exp(βⱼ)`

**Example:**
If the coefficient for DTI_WOE is β = −0.72, then:
- Odds Ratio = exp(−0.72) = 0.487
- Moving from a high-risk DTI bin (WOE = −1.058) to a low-risk bin (WOE = +1.386) changes the log-odds by: −0.72 × (1.386 − (−1.058)) = −0.72 × 2.444 = −1.76
- This corresponds to multiplying the default odds by exp(−1.76) ≈ 0.172 — the borrower is roughly 83% less likely to default if they are in the lowest DTI bin vs. the highest.

### Scorecard Scaling

Banks transform the raw model output into a **credit score** (e.g., 300–850 for FICO-like scales, or 0–1000 for internal scorecards). The scaling makes the model more interpretable to business stakeholders and borrowers.

Three parameters define the scale:

| Parameter | Definition | Typical Value |
|---|---|---|
| Base Score (B) | Score at the base odds | 600 |
| Base Odds (θ₀) | Good-to-bad odds at the base score | 50:1 |
| PDO | Points to Double the Odds | 20 |

From these, derive:

```
Factor = PDO / ln(2) = 20 / 0.6931 = 28.854

Offset = B − Factor × ln(θ₀) = 600 − 28.854 × ln(50) = 600 − 28.854 × 3.912 = 487.12

Score = Offset + Factor × ln(good:bad odds)
      = Offset + Factor × (−logit(PD))
      = 487.12 + 28.854 × (−logit(PD))
```

Since a higher score = lower risk, the score *decreases* as the predicted PD increases.

**Score Examples at Different PDO:**

| PD | Odds (Good:Bad) | Score |
|---|---|---|
| 1.96% | 50:1 (base) | 600 |
| 0.99% | 100:1 | 600 + 28.854 × ln(2) = **620** |
| 3.76% | 25:1 | 600 − 28.854 × ln(2) = **580** |
| 7.41% | 12.5:1 | 600 − 2 × 28.854 × ln(2) = **560** |

---

## 8. The Full Python Implementation

Now we build the complete credit risk model from scratch. The code is structured as a pipeline that can be executed end-to-end.

### Prerequisites

```bash
pip install pandas numpy scikit-learn matplotlib seaborn optbinning imbalanced-learn shap scorecardpy
```

### The Complete Code

```python
# ============================================================
# CREDIT RISK MODEL — PROBABILITY OF DEFAULT ESTIMATION
# Full Pipeline: EDA → WOE/IV → Logistic Regression → Metrics
# ============================================================

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec
import seaborn as sns
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import (roc_auc_score, roc_curve,
                             classification_report, confusion_matrix,
                             accuracy_score)
from sklearn.preprocessing import StandardScaler
from sklearn.utils.class_weight import compute_class_weight
import warnings
import shap

warnings.filterwarnings('ignore')
np.random.seed(42)

# ── STYLE ────────────────────────────────────────────────────
plt.rcParams['figure.figsize'] = (12, 6)
plt.rcParams['font.size'] = 12
sns.set_style('whitegrid')


# ============================================================
# SECTION 1: DATA LOADING AND PREPARATION
# ============================================================

def load_and_prepare(path: str) -> pd.DataFrame:
    """
    Load Lending Club CSV, filter to completed loans,
    create binary target variable, and select features.
    """
    print("Loading data...")
    df = pd.read_csv(path, low_memory=False)

    # Keep only fully resolved loans
    df = df[df['loan_status'].isin(['Fully Paid', 'Charged Off'])].copy()

    # Binary target: 1 = default, 0 = repaid
    df['default'] = (df['loan_status'] == 'Charged Off').astype(int)

    # Select modeling features
    features = [
        'loan_amnt', 'int_rate', 'annual_inc', 'dti',
        'fico_range_low', 'revol_util', 'revol_bal',
        'open_acc', 'delinq_2yrs', 'inq_last_6mths',
        'pub_rec', 'emp_length', 'term', 'home_ownership',
        'purpose', 'default'
    ]
    df = df[features].copy()

    # ── Clean numeric columns ────────────────────────────────
    # emp_length: "10+ years" → 10, "< 1 year" → 0
    emp_map = {
        '< 1 year': 0, '1 year': 1, '2 years': 2, '3 years': 3,
        '4 years': 4, '5 years': 5, '6 years': 6, '7 years': 7,
        '8 years': 8, '9 years': 9, '10+ years': 10
    }
    df['emp_length'] = df['emp_length'].map(emp_map)

    # term: "36 months" → 36
    df['term'] = df['term'].str.strip().str.replace(' months', '').astype(float)

    # int_rate: "12.5%" → 12.5
    if df['int_rate'].dtype == object:
        df['int_rate'] = df['int_rate'].str.replace('%', '').astype(float)

    # Clip extreme outliers (99.9th percentile cap)
    for col in ['annual_inc', 'revol_bal', 'loan_amnt']:
        cap = df[col].quantile(0.999)
        df[col] = df[col].clip(upper=cap)

    # Fill missing values with median (a reasonable default for scorecard models)
    num_cols = df.select_dtypes(include=[np.number]).columns
    df[num_cols] = df[num_cols].fillna(df[num_cols].median())

    # Categorical missing values → mode
    cat_cols = df.select_dtypes(include='object').columns
    for col in cat_cols:
        df[col] = df[col].fillna(df[col].mode()[0])

    print(f"  Records loaded: {len(df):,}")
    print(f"  Default rate:   {df['default'].mean():.2%}")
    return df


# ============================================================
# SECTION 2: WOE / IV FEATURE ENGINEERING
# ============================================================

def compute_woe_iv(df: pd.DataFrame, feature: str,
                   target: str = 'default',
                   bins: int = 10,
                   cat: bool = False) -> tuple:
    """
    Compute Weight of Evidence (WOE) and Information Value (IV)
    for a single feature.

    Returns:
        woe_map  : dict mapping bin → WOE value
        iv       : float, Information Value
        woe_df   : DataFrame with full binning statistics
    """
    total_events     = df[target].sum()
    total_non_events = len(df) - total_events

    if cat:
        # Categorical: use each unique value as a bin
        grouped = df.groupby(feature)[target].agg(['sum', 'count'])
    else:
        # Continuous: equal-frequency binning
        df['__bin__'] = pd.qcut(df[feature], q=bins, duplicates='drop')
        grouped = df.groupby('__bin__', observed=False)[target].agg(['sum', 'count'])

    grouped.columns = ['events', 'total']
    grouped['non_events'] = grouped['total'] - grouped['events']

    # Avoid log(0) by adding a small smoothing constant
    eps = 0.5
    grouped['events']     = grouped['events'] + eps
    grouped['non_events'] = grouped['non_events'] + eps

    grouped['dist_events']     = grouped['events'] / total_events
    grouped['dist_non_events'] = grouped['non_events'] / total_non_events

    grouped['woe'] = np.log(grouped['dist_non_events'] / grouped['dist_events'])
    grouped['iv']  = (grouped['dist_non_events'] - grouped['dist_events']) * grouped['woe']

    iv = grouped['iv'].sum()

    # Build mapping dict
    if cat:
        woe_map = grouped['woe'].to_dict()
    else:
        # Map each cut interval to WOE
        woe_map = grouped['woe'].to_dict()
        df.drop(columns='__bin__', inplace=True)

    return woe_map, iv, grouped


def screen_features_by_iv(df: pd.DataFrame,
                           cont_features: list,
                           cat_features: list,
                           target: str = 'default',
                           min_iv: float = 0.02) -> pd.DataFrame:
    """
    Compute IV for all features and return a ranked DataFrame.
    Features with IV < min_iv are flagged for removal.
    """
    results = []
    for feat in cont_features:
        try:
            _, iv, _ = compute_woe_iv(df, feat, target, bins=10, cat=False)
            results.append({'feature': feat, 'iv': iv, 'type': 'continuous'})
        except Exception:
            pass

    for feat in cat_features:
        try:
            _, iv, _ = compute_woe_iv(df, feat, target, bins=10, cat=True)
            results.append({'feature': feat, 'iv': iv, 'type': 'categorical'})
        except Exception:
            pass

    iv_df = pd.DataFrame(results).sort_values('iv', ascending=False)

    def categorize_iv(iv):
        if iv < 0.02:   return 'Useless'
        elif iv < 0.10: return 'Weak'
        elif iv < 0.30: return 'Medium'
        elif iv < 0.50: return 'Strong'
        else:           return 'Very Strong (check leakage)'

    iv_df['strength']  = iv_df['iv'].apply(categorize_iv)
    iv_df['selected']  = iv_df['iv'] >= min_iv
    return iv_df


def apply_woe_transform(df: pd.DataFrame,
                         cont_features: list,
                         cat_features: list,
                         target: str = 'default') -> tuple:
    """
    Apply WOE transformation to all features.
    Returns the transformed DataFrame and a dict of WOE maps.
    """
    df_woe  = df[[target]].copy()
    woe_maps = {}

    for feat in cont_features:
        try:
            woe_map, _, _ = compute_woe_iv(df, feat, target, bins=10, cat=False)
            # Assign WOE using bin membership
            bins_series = pd.qcut(df[feat], q=10, duplicates='drop', retbins=False)
            df_woe[feat + '_WOE'] = bins_series.map(woe_map)
            woe_maps[feat] = woe_map
        except Exception:
            pass

    for feat in cat_features:
        try:
            woe_map, _, _ = compute_woe_iv(df, feat, target, bins=10, cat=True)
            df_woe[feat + '_WOE'] = df[feat].map(woe_map)
            woe_maps[feat] = woe_map
        except Exception:
            pass

    # Drop any columns with all NaN
    df_woe.dropna(axis=1, how='all', inplace=True)
    df_woe.fillna(0, inplace=True)

    return df_woe, woe_maps


# ============================================================
# SECTION 3: MULTICOLLINEARITY CHECK (VIF)
# ============================================================

def compute_vif(X: pd.DataFrame) -> pd.DataFrame:
    """
    Compute Variance Inflation Factor for all features.
    VIF > 10 signals severe multicollinearity.
    """
    from sklearn.linear_model import LinearRegression

    vif_data = []
    cols = X.columns.tolist()

    for i, col in enumerate(cols):
        others = [c for c in cols if c != col]
        model  = LinearRegression().fit(X[others], X[col])
        r2     = model.score(X[others], X[col])
        vif    = 1 / (1 - r2) if r2 < 1.0 else np.inf
        vif_data.append({'feature': col, 'VIF': round(vif, 2)})

    return pd.DataFrame(vif_data).sort_values('VIF', ascending=False)


# ============================================================
# SECTION 4: MODEL TRAINING
# ============================================================

def train_logistic_model(X_train, y_train):
    """
    Train a logistic regression model with balanced class weights
    to handle the minority class (defaulters).
    """
    # class_weight='balanced' makes the model pay more attention to
    # the minority class (defaulters) by scaling loss contributions
    model = LogisticRegression(
        penalty='l2',           # L2 regularization (Ridge) to prevent overfitting
        C=1.0,                  # Inverse of regularization strength
        class_weight='balanced',# Upweight defaulters
        max_iter=1000,
        solver='lbfgs',
        random_state=42
    )
    model.fit(X_train, y_train)
    return model


# ============================================================
# SECTION 5: PERFORMANCE METRICS
# ============================================================

def compute_ks_statistic(y_true: np.ndarray, y_scores: np.ndarray) -> tuple:
    """
    Compute the KS (Kolmogorov-Smirnov) statistic.

    KS = maximum difference between the CDF of defaulters and
         the CDF of non-defaulters, across all score thresholds.

    Returns:
        ks_stat   : float — the KS value (0 to 1)
        ks_thresh : float — score threshold at which KS is achieved
    """
    # Sort by score descending
    order     = np.argsort(-y_scores)
    y_true_s  = y_true[order]
    y_scores_s = y_scores[order]

    # Cumulative proportions
    n_events     = y_true.sum()
    n_non_events = len(y_true) - n_events

    cum_events     = np.cumsum(y_true_s) / n_events
    cum_non_events = np.cumsum(1 - y_true_s) / n_non_events

    ks_values = np.abs(cum_events - cum_non_events)
    idx       = np.argmax(ks_values)

    return ks_values[idx], y_scores_s[idx]


def compute_gini(auc: float) -> float:
    """Gini coefficient: Gini = 2 × AUC − 1"""
    return 2 * auc - 1


def evaluate_model(model, X_test, y_test, dataset_name='Test'):
    """
    Comprehensive model evaluation: AUC, KS, Gini, Accuracy,
    Classification Report, and Confusion Matrix.
    """
    y_pred_prob = model.predict_proba(X_test)[:, 1]
    y_pred      = model.predict(X_test)

    auc       = roc_auc_score(y_test, y_pred_prob)
    ks, ks_t  = compute_ks_statistic(y_test.values, y_pred_prob)
    gini      = compute_gini(auc)
    accuracy  = accuracy_score(y_test, y_pred)

    print(f"\n{'='*50}")
    print(f"  MODEL PERFORMANCE — {dataset_name.upper()} SET")
    print(f"{'='*50}")
    print(f"  AUC (ROC):          {auc:.4f}  {'✓ Good' if auc >= 0.75 else '✗ Needs work'}")
    print(f"  KS Statistic:       {ks:.4f}  {'✓ Good' if ks >= 0.25 else '✗ Needs work'}")
    print(f"  Gini Coefficient:   {gini:.4f}  {'✓ Good' if gini >= 0.40 else '✗ Needs work'}")
    print(f"  Accuracy:           {accuracy:.4f}")
    print(f"  KS Threshold:       {ks_t:.4f} (optimal cutoff score)")
    print(f"\n  Classification Report (default threshold = 0.5):")
    print(classification_report(y_test, y_pred,
                                target_names=['Non-Default', 'Default']))

    return {
        'auc': auc, 'ks': ks, 'gini': gini,
        'accuracy': accuracy, 'y_pred_prob': y_pred_prob
    }


def plot_roc_curve(y_test, y_pred_prob, auc, title='ROC Curve'):
    """Plot the ROC curve with AUC annotation."""
    fpr, tpr, _ = roc_curve(y_test, y_pred_prob)

    fig, ax = plt.subplots(figsize=(8, 7))
    ax.plot(fpr, tpr, color='steelblue', lw=2,
            label=f'Model (AUC = {auc:.4f})')
    ax.plot([0, 1], [0, 1], 'k--', lw=1.5, label='Random Classifier (AUC = 0.50)')
    ax.fill_between(fpr, tpr, alpha=0.15, color='steelblue')
    ax.set_xlabel('False Positive Rate (1 − Specificity)')
    ax.set_ylabel('True Positive Rate (Sensitivity / Recall)')
    ax.set_title(title)
    ax.legend(loc='lower right')
    ax.annotate(f'AUC = {auc:.4f}', xy=(0.6, 0.3),
                fontsize=13, color='steelblue',
                bbox=dict(boxstyle='round', facecolor='white', alpha=0.8))
    plt.tight_layout()
    plt.savefig('roc_curve.png', dpi=150)
    plt.show()
    print("  [Saved: roc_curve.png]")


def plot_ks_chart(y_test, y_pred_prob, title='KS Chart'):
    """
    Plot the KS chart: cumulative distributions of defaulters
    and non-defaulters across score thresholds.
    """
    order          = np.argsort(-y_pred_prob)
    y_true_s       = y_test.values[order]
    n              = len(y_true_s)
    n_events       = y_true_s.sum()
    n_non_events   = n - n_events

    cum_events     = np.cumsum(y_true_s) / n_events
    cum_non_events = np.cumsum(1 - y_true_s) / n_non_events
    ks_values      = np.abs(cum_events - cum_non_events)
    ks_idx         = np.argmax(ks_values)
    pct_pop        = np.arange(1, n + 1) / n * 100

    fig, ax = plt.subplots(figsize=(10, 6))
    ax.plot(pct_pop, cum_events * 100,     color='crimson',    lw=2, label='Cumulative Defaulters (%)')
    ax.plot(pct_pop, cum_non_events * 100, color='steelblue',  lw=2, label='Cumulative Non-Defaulters (%)')
    ax.axvline(x=pct_pop[ks_idx], color='gray', linestyle='--', lw=1.5)

    ks_val = ks_values[ks_idx]
    ax.annotate(
        f'KS = {ks_val:.3f} at\n{pct_pop[ks_idx]:.1f}% of population',
        xy=(pct_pop[ks_idx], (cum_events[ks_idx] + cum_non_events[ks_idx]) / 2 * 100),
        xytext=(pct_pop[ks_idx] + 5, 40),
        fontsize=11, color='black',
        arrowprops=dict(arrowstyle='->', color='black'),
        bbox=dict(boxstyle='round', facecolor='lightyellow', alpha=0.9)
    )

    ax.set_xlabel('% of Population (sorted by descending risk score)')
    ax.set_ylabel('Cumulative %')
    ax.set_title(title)
    ax.legend()
    plt.tight_layout()
    plt.savefig('ks_chart.png', dpi=150)
    plt.show()
    print("  [Saved: ks_chart.png]")


# ============================================================
# SECTION 6: SCORECARD SCALING
# ============================================================

def build_scorecard(model, woe_feature_names: list,
                    base_score: int = 600,
                    base_odds: int = 50,
                    pdo: int = 20) -> pd.DataFrame:
    """
    Convert logistic regression coefficients + WOE maps
    into a points-based credit scorecard.

    Each feature × bin combination receives a points value.
    A borrower's total score = sum of points across all features.
    """
    factor = pdo / np.log(2)
    offset = base_score - factor * np.log(base_odds)

    n = len(woe_feature_names)  # number of variables
    intercept = model.intercept_[0]
    coefs     = model.coef_[0]

    records = []
    for i, feat in enumerate(woe_feature_names):
        beta   = coefs[i]
        # Points contribution of each variable is spread equally from intercept
        points = -(beta * 0 + intercept / n) * factor + offset / n
        records.append({
            'Feature':     feat,
            'Coefficient': round(beta, 4),
            'Points at WOE=0': round(points, 1)
        })

    return pd.DataFrame(records)


def predict_score(log_odds: float,
                  base_score: int = 600,
                  base_odds: int = 50,
                  pdo: int = 20) -> float:
    """Convert log-odds to a scaled credit score."""
    factor = pdo / np.log(2)
    offset = base_score - factor * np.log(base_odds)
    return offset + factor * (-log_odds)


# ============================================================
# SECTION 7: SHAP FEATURE IMPORTANCE
# ============================================================

def plot_shap_importance(model, X_train, X_test,
                          feature_names: list):
    """
    Compute SHAP values for the logistic regression model
    and plot global feature importance + a beeswarm plot.
    """
    print("\nComputing SHAP values (this may take a moment)...")
    explainer  = shap.LinearExplainer(model, X_train)
    shap_values = explainer.shap_values(X_test)

    # Summary bar plot (global importance)
    plt.figure(figsize=(10, 7))
    shap.summary_plot(shap_values, X_test,
                      feature_names=feature_names,
                      plot_type='bar', show=False)
    plt.title('SHAP Feature Importance (Mean |SHAP Value|)')
    plt.tight_layout()
    plt.savefig('shap_importance.png', dpi=150)
    plt.show()
    print("  [Saved: shap_importance.png]")

    # Beeswarm / dot plot (shows direction of impact)
    plt.figure(figsize=(10, 8))
    shap.summary_plot(shap_values, X_test,
                      feature_names=feature_names,
                      plot_type='dot', show=False)
    plt.title('SHAP Beeswarm — Feature Impact Direction')
    plt.tight_layout()
    plt.savefig('shap_beeswarm.png', dpi=150)
    plt.show()
    print("  [Saved: shap_beeswarm.png]")

    return shap_values


# ============================================================
# SECTION 8: PSI (POPULATION STABILITY INDEX)
# ============================================================

def compute_psi(expected: np.ndarray,
                actual: np.ndarray,
                bins: int = 10) -> float:
    """
    Compute PSI to detect distribution shift between
    development (expected) and current (actual) score distributions.

    PSI < 0.10  → Stable
    PSI 0.10–0.25 → Monitor
    PSI > 0.25  → Significant drift; consider model rebuild
    """
    # Define bins on the expected distribution
    _, bin_edges = np.histogram(expected, bins=bins)
    bin_edges[0]  = -np.inf
    bin_edges[-1] =  np.inf

    exp_counts, _ = np.histogram(expected, bins=bin_edges)
    act_counts, _ = np.histogram(actual,   bins=bin_edges)

    exp_pct = exp_counts / len(expected)
    act_pct = act_counts / len(actual)

    # Smoothing to avoid log(0)
    eps = 1e-4
    exp_pct = np.where(exp_pct == 0, eps, exp_pct)
    act_pct = np.where(act_pct == 0, eps, act_pct)

    psi = np.sum((act_pct - exp_pct) * np.log(act_pct / exp_pct))
    return psi


# ============================================================
# SECTION 9: EXPECTED LOSS CALCULATION
# ============================================================

def compute_expected_loss(pd_values: np.ndarray,
                           lgd: float,
                           ead_values: np.ndarray) -> pd.DataFrame:
    """
    Compute Expected Loss for each loan in the portfolio.
    EL = PD × LGD × EAD
    """
    el  = pd_values * lgd * ead_values
    df  = pd.DataFrame({
        'PD':  pd_values,
        'LGD': lgd,
        'EAD': ead_values,
        'EL':  el
    })
    print(f"\n  Portfolio Expected Loss Summary:")
    print(f"  Total EAD:     ${df['EAD'].sum():>15,.0f}")
    print(f"  Total EL:      ${df['EL'].sum():>15,.0f}")
    print(f"  EL / EAD:      {df['EL'].sum() / df['EAD'].sum():.3%}  (annualized loss rate)")
    return df


# ============================================================
# SECTION 10: MAIN EXECUTION PIPELINE
# ============================================================

def main():
    # ── 1. Load data ─────────────────────────────────────────
    df = load_and_prepare('lending_club.csv')

    # ── 2. Define feature lists ──────────────────────────────
    continuous_features = [
        'loan_amnt', 'int_rate', 'annual_inc', 'dti',
        'fico_range_low', 'revol_util', 'revol_bal',
        'open_acc', 'delinq_2yrs', 'inq_last_6mths',
        'pub_rec', 'emp_length', 'term'
    ]
    categorical_features = ['home_ownership', 'purpose']

    # ── 3. IV Screening ──────────────────────────────────────
    print("\n── IV Feature Screening ──")
    iv_df = screen_features_by_iv(df, continuous_features,
                                   categorical_features)
    print(iv_df.to_string(index=False))

    # Keep features with IV >= 0.02
    selected_cont = iv_df[
        (iv_df['type'] == 'continuous') & (iv_df['selected'])
    ]['feature'].tolist()
    selected_cat  = iv_df[
        (iv_df['type'] == 'categorical') & (iv_df['selected'])
    ]['feature'].tolist()

    # ── 4. WOE Transformation ────────────────────────────────
    print("\n── WOE Transformation ──")
    df_woe, woe_maps = apply_woe_transform(df, selected_cont,
                                            selected_cat)
    print(f"  WOE features created: {df_woe.shape[1] - 1}")

    # ── 5. Train / Test Split (temporal-style via random here) ──
    X = df_woe.drop(columns='default')
    y = df_woe['default']

    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.20, random_state=42, stratify=y
    )
    print(f"\n  Train size: {len(X_train):,} | Test size: {len(X_test):,}")
    print(f"  Train default rate: {y_train.mean():.2%}")
    print(f"  Test default rate:  {y_test.mean():.2%}")

    # ── 6. VIF Check ─────────────────────────────────────────
    print("\n── VIF Multicollinearity Check ──")
    vif_df = compute_vif(X_train)
    print(vif_df.to_string(index=False))
    high_vif = vif_df[vif_df['VIF'] > 10]['feature'].tolist()
    if high_vif:
        print(f"  Warning: High VIF features: {high_vif}")
        X_train.drop(columns=high_vif, inplace=True)
        X_test.drop(columns=high_vif, inplace=True)

    # ── 7. Train Model ───────────────────────────────────────
    print("\n── Training Logistic Regression ──")
    model = train_logistic_model(X_train, y_train)
    print("  Model trained successfully.")

    # ── 8. Evaluate ──────────────────────────────────────────
    results = evaluate_model(model, X_test, y_test, 'Test')

    # ── 9. Plots ─────────────────────────────────────────────
    print("\n── Generating Plots ──")
    plot_roc_curve(y_test, results['y_pred_prob'],
                   results['auc'], 'ROC Curve — Credit Risk Model')
    plot_ks_chart(y_test, results['y_pred_prob'],
                  'KS Chart — Credit Risk Model')

    # ── 10. SHAP Importance ──────────────────────────────────
    shap_vals = plot_shap_importance(model, X_train, X_test,
                                      X_train.columns.tolist())

    # ── 11. Scorecard ────────────────────────────────────────
    print("\n── Credit Scorecard ──")
    scorecard = build_scorecard(model, X_train.columns.tolist())
    print(scorecard.to_string(index=False))

    # ── 12. Expected Loss ────────────────────────────────────
    pd_values  = model.predict_proba(X_test)[:, 1]
    ead_values = df.loc[y_test.index, 'loan_amnt'].values
    el_df = compute_expected_loss(pd_values, lgd=0.75, ead_values=ead_values)

    # ── 13. PSI ──────────────────────────────────────────────
    # In practice: compare train score distribution to a new time window
    # Here, compare train to test as a demonstration
    psi_val = compute_psi(
        model.predict_proba(X_train)[:, 1],
        model.predict_proba(X_test)[:, 1]
    )
    print(f"\n── Population Stability Index (PSI) ──")
    print(f"  PSI (Train vs. Test): {psi_val:.4f} → ", end='')
    if   psi_val < 0.10:  print("Stable (no action needed)")
    elif psi_val < 0.25:  print("Moderate shift (monitor closely)")
    else:                  print("Significant shift (investigate)")

    print("\n── Pipeline Complete ──")
    return model, iv_df, results, el_df


if __name__ == '__main__':
    model, iv_df, results, el_df = main()
```

---

## 9. Line-by-Line Code Explanation

This section walks through the most important — and potentially confusing — parts of the code above.

### `load_and_prepare()` — The Data Cleaning Function

**Why do we only keep "Fully Paid" and "Charged Off" loans?**
Lending Club data also contains loans with statuses like "Current," "Late (31-120 days)," and "In Grace Period." These loans have not yet reached a final outcome, so we cannot know whether they will default. Including them would introduce noise — the label would be ambiguous. We need **resolved** loans with a definitive outcome.

**Why clip at the 99.9th percentile?**
Features like `annual_inc` have extreme outliers (some records show annual incomes of $10 million+). These outliers can distort the logistic regression coefficients. Clipping at 99.9% removes extreme values while retaining 99.9% of the distribution intact.

**Why fill missing values with the median?**
Logistic regression cannot handle `NaN` values. The median is more robust than the mean for skewed financial data. More sophisticated approaches include WOE bins for "missing" as a separate category — this is standard in credit scorecard development.

---

### `compute_woe_iv()` — The Heart of Feature Engineering

The +0.5 smoothing constant (`eps = 0.5`) deserves attention. Without it, a bin with zero events would produce `ln(0)` which is undefined. Adding 0.5 (a common choice, also called "Laplace smoothing") avoids this while having minimal impact on bins with many observations.

**The formula expanded:**

```python
grouped['dist_events']     = grouped['events'] / total_events
# This is: (# defaults in bin) / (total defaults in dataset)

grouped['dist_non_events'] = grouped['non_events'] / total_non_events
# This is: (# non-defaults in bin) / (total non-defaults in dataset)

grouped['woe'] = np.log(grouped['dist_non_events'] / grouped['dist_events'])
# WOE = ln( (non-defaults in bin / all non-defaults) / (defaults in bin / all defaults) )
```

The ratio `dist_non_events / dist_events` is the **relative representation** of non-defaulters vs. defaulters in each bin. A high ratio (many non-defaulters, few defaulters) → positive WOE → low-risk bin.

---

### `compute_ks_statistic()` — The KS Metric

```python
order = np.argsort(-y_scores)
# Sort all observations from highest predicted PD to lowest.
# We are asking: "if we rank every borrower from riskiest to safest,
#  how well does this ranking separate actual defaulters from non-defaulters?"

cum_events     = np.cumsum(y_true_s) / n_events
# At each threshold, what % of all actual defaulters have we "captured"?
# (This is the CDF of defaulters over the ranked population)

cum_non_events = np.cumsum(1 - y_true_s) / n_non_events
# At each threshold, what % of all non-defaulters have we "captured"?

ks_values = np.abs(cum_events - cum_non_events)
# The separation between the two CDFs at each threshold
# KS = maximum separation across all thresholds
```

The KS statistic identifies the **optimal cutoff threshold** — the score at which the model maximally separates the two populations.

---

### `train_logistic_model()` — Handling Class Imbalance

```python
model = LogisticRegression(class_weight='balanced', ...)
```

With `class_weight='balanced'`, scikit-learn automatically sets the weight for each class to be inversely proportional to its frequency:

```
weight[0] = n_samples / (2 × n_class_0)
weight[1] = n_samples / (2 × n_class_1)
```

If there are 1,000 non-defaults and 100 defaults:
```
weight[0] = 1100 / (2 × 1000) = 0.55
weight[1] = 1100 / (2 × 100)  = 5.50
```

Every misclassified defaulter now costs 10× more in the loss function than a misclassified non-defaulter. This forces the model to "try harder" to correctly identify the minority class.

---

## 10. Feature Analysis — Identifying Key Risk Drivers

### Understanding the IV Rankings

After running `screen_features_by_iv()`, you will typically see output like this:

```
feature            iv       type         strength    selected
fico_range_low     0.412    continuous   Strong         True
int_rate           0.389    continuous   Strong         True
dti                0.218    continuous   Medium         True
revol_util         0.197    continuous   Medium         True
delinq_2yrs        0.182    continuous   Medium         True
purpose            0.148    categorical  Medium         True
inq_last_6mths     0.127    continuous   Medium         True
home_ownership     0.098    categorical  Weak           True
emp_length         0.071    continuous   Weak           True
open_acc           0.043    continuous   Weak           True
annual_inc         0.038    continuous   Weak           True
pub_rec            0.031    continuous   Weak           True
revol_bal          0.021    continuous   Weak           True
loan_amnt          0.009    continuous   Useless       False
```

**Reading this table:**

1. **FICO Score (IV = 0.412):** The strongest predictor. This is exactly what we expect — FICO is specifically designed to predict credit default. A FICO score captures payment history, utilization, credit age, and other signals.

2. **Interest Rate (IV = 0.389):** Very strong, but as discussed, this may partially reflect circularity — the rate was set based on the lender's prior risk assessment of the borrower.

3. **DTI (IV = 0.218):** A medium-strength predictor. Higher debt relative to income is a robust risk driver.

4. **Revolving Utilization (IV = 0.197):** Credit card utilization above 30% is a well-documented signal of financial stress.

5. **Delinquencies (IV = 0.182):** Past payment behavior is one of the most predictive signals — borrowers who have been delinquent before are more likely to be delinquent again.

6. **Loan Amount (IV = 0.009):** Essentially useless on its own — the loan amount without context (e.g., relative to income) has little discriminating power.

### Visualizing WOE Patterns

For each selected feature, the WOE chart reveals the shape of the risk relationship:

**FICO Score WOE Pattern (Expected):**

```
WOE
+2.5 |         ____/
+2.0 |       _/
+1.5 |      /
+1.0 |    _/
+0.5 |   /
 0   |  /
-0.5 | /
-1.0 |/
-1.5 +────────────────────────→ FICO Score
     550  600  650  700  750  800
```

This **monotonically increasing** WOE pattern means: higher FICO → higher WOE → more non-defaulters proportionally → lower risk. Exactly as expected.

**Delinquency WOE Pattern (Expected):**

```
WOE
+1.2 |●
     |  ●
 0   |    ●
     |      ●
-0.8 |        ●
     +──────────────→ # Delinquencies
     0   1   2   3+
```

**Monotonically decreasing** — more delinquencies → lower WOE → higher risk.

### Feature Importance from Model Coefficients

After training, you can quantify each feature's contribution to the model:

```python
coef_df = pd.DataFrame({
    'feature':     X_train.columns,
    'coefficient': model.coef_[0]
}).sort_values('coefficient', key=abs, ascending=False)

print(coef_df)
```

Expected output:

```
feature               coefficient
fico_range_low_WOE       -0.912
int_rate_WOE             -0.887
dti_WOE                  -0.734
revol_util_WOE           -0.698
delinq_2yrs_WOE          -0.621
inq_last_6mths_WOE       -0.509
purpose_WOE              -0.312
home_ownership_WOE       -0.214
emp_length_WOE           -0.198
open_acc_WOE             -0.141
annual_inc_WOE           -0.127
pub_rec_WOE              -0.089
revol_bal_WOE            -0.055
```

**Why are all coefficients negative?**
Because WOE is defined from the non-defaulter perspective — higher WOE means lower risk. The logistic regression correctly assigns negative coefficients: higher WOE → lower log-odds of default.

The **absolute magnitude** tells us the relative importance: FICO and interest rate are the dominant drivers; revolving balance contributes least.

---

## 11. Model Performance — ROC, KS, and Gini

### The ROC Curve in Depth

The **ROC (Receiver Operating Characteristic) curve** was originally developed in radar signal detection theory in the 1940s — hence the name. It asks: *at every possible decision threshold, how does the model trade off catching true signals (defaulters) against false alarms (good borrowers incorrectly flagged)?*

**Construction:** Sweep the threshold `t` from 1.0 (predict everyone as non-default) to 0.0 (predict everyone as default). At each `t`:

```
TPR(t) = True Positive Rate = TP(t) / (TP(t) + FN(t))
       = "Of all actual defaulters, what % did we catch?"

FPR(t) = False Positive Rate = FP(t) / (FP(t) + TN(t))
       = "Of all actual non-defaulters, what % did we falsely flag?"
```

Plot each `(FPR(t), TPR(t))` pair. At `t = 1.0`: (0, 0) — nobody predicted as default. At `t = 0.0`: (1, 1) — everybody predicted as default.

**AUC — Area Under the Curve:**

```
AUC = ∫₀¹ TPR(FPR) dFPR
    = P(score(defaulter) > score(non-defaulter))
```

The second interpretation is powerful: AUC is the probability that a randomly chosen defaulter gets a **higher risk score** than a randomly chosen non-defaulter. A perfect model scores every defaulter above every non-defaulter → AUC = 1.0. A random guesser: AUC = 0.5.

**Industry Benchmarks for Credit Models:**

| AUC | Assessment | Interpretation |
|---|---|---|
| < 0.60 | Poor | Barely better than random; reject |
| 0.60–0.70 | Acceptable | Some discriminating power |
| 0.70–0.80 | Good | Solid performance |
| 0.75–0.85 | Very Good | Typical range for well-built scorecard |
| > 0.85 | Excellent | Strong; verify no data leakage |
| > 0.95 | Suspicious | Almost certainly data leakage present |

---

### The KS Statistic in Depth

The **KS (Kolmogorov-Smirnov)** statistic is the credit risk industry's most-cited single-number performance metric. It measures the maximum gap between the cumulative distribution of defaulter scores and non-defaulter scores.

**Visual interpretation (the KS Chart):**

Imagine two armies: defaulters on one side, non-defaulters on the other. You sort all borrowers from highest risk score to lowest, then march through them one by one. You track two cumulative counts:
- How many defaulters you have passed so far (as a %)
- How many non-defaulters you have passed so far (as a %)

If your model is good, you will accumulate defaulters rapidly at the high-score end (the model concentrated defaulters at the top). The gap between the two curves is widest at some point — that is the KS threshold, and the gap itself is the KS statistic.

**Industry Benchmarks:**

| KS | Assessment |
|---|---|
| < 20% | Poor |
| 20–30% | Acceptable |
| 30–40% | Good |
| 40–50% | Very Good |
| > 50% | Excellent |

A KS of 40% means: at the optimal threshold, the model captures 40% more of the defaults per unit of population than a random model. Concretely, if you screen the top 20% of borrowers by risk score, you might capture 55% of actual defaults (35 percentage points above the 20% random baseline).

---

### The Gini Coefficient

The **Gini coefficient** in credit risk (also called the Accuracy Ratio) is derived from the **Cumulative Accuracy Profile (CAP)** curve, and has a direct mathematical relationship to AUC:

```
Gini = 2 × AUC − 1
```

This means if AUC = 0.76, then Gini = 0.52. The Gini ranges from 0 (random) to 1 (perfect).

**Why two metrics that are just linear transforms of each other?**
Historical convention. European banks tend to report Gini; US banks tend to report AUC. They convey identical information — just shifted and scaled.

**Industry Benchmarks for Gini:**

| Gini | Assessment |
|---|---|
| < 0.20 | Poor |
| 0.20–0.40 | Acceptable |
| 0.40–0.60 | Good |
| > 0.60 | Very Good |

---

### The Confusion Matrix and Classification Report

At a chosen threshold (not necessarily 0.5 — we discuss threshold selection in Section 13), the model produces a confusion matrix:

```
                    Predicted Non-Default  Predicted Default
Actual Non-Default        TN                    FP
Actual Default            FN                    TP
```

**In credit risk language:**

| Cell | Label | Business Meaning | Cost |
|---|---|---|---|
| TN | True Negative | Good borrower correctly approved | Revenue earned |
| TP | True Positive | Defaulter correctly declined | Loss avoided |
| FP | False Positive | Good borrower incorrectly declined | Opportunity cost; lost margin |
| FN | False Negative | Defaulter incorrectly approved | **Credit loss — most costly** |

> **Critical asymmetry:** FN (approving a future defaulter) is typically 5–20× more costly than FP (declining a future good borrower). This drives threshold selection toward lower values to reduce FN at the expense of more FP.

---

## 12. Numerical Examples — Worked by Hand

### Example 1: Computing WOE and IV by Hand

**Setup:** 500 loans, 50 defaults. Feature: `delinq_2yrs` (number of delinquencies in the past 2 years).

| Delinquencies | Defaults | Non-Defaults | Events% | Non-Events% | WOE | (NE%-E%)×WOE |
|---|---|---|---|---|---|---|
| 0 | 20 | 400 | 20/50 = **40%** | 400/450 = **88.9%** | ln(88.9/40.0) = **0.799** | (0.889−0.400)×0.799 = **0.391** |
| 1 | 15 | 35 | 15/50 = **30%** | 35/450 = **7.8%** | ln(7.8/30.0) = **−1.346** | (0.078−0.300)×(−1.346) = **0.299** |
| 2 | 10 | 10 | 10/50 = **20%** | 10/450 = **2.2%** | ln(2.2/20.0) = **−2.207** | (0.022−0.200)×(−2.207) = **0.393** |
| 3+ | 5 | 5 | 5/50 = **10%** | 5/450 = **1.1%** | ln(1.1/10.0) = **−2.207** | (0.011−0.100)×(−2.207) = **0.197** |
| **Total** | **50** | **450** | **100%** | **100%** | | **IV = 1.280** |

**Interpretation:**
- Zero delinquencies: WOE = +0.80 → Strongly low-risk (88.9% of non-defaulters here vs. only 40% of defaulters)
- 3+ delinquencies: WOE = −2.21 → Severely high-risk
- IV = 1.28 → Extraordinary predictive power (verify for data leakage; in practice, recent delinquency history is a legitimate, powerful predictor)

---

### Example 2: Computing PD from Logistic Regression

**Setup:** A loan application has the following WOE-transformed values:

| Feature | WOE Value |
|---|---|
| `fico_range_low_WOE` | 1.20 |
| `int_rate_WOE` | −0.85 |
| `dti_WOE` | −0.60 |
| `revol_util_WOE` | −0.40 |
| `delinq_2yrs_WOE` | 0.80 |

**Model coefficients:**

| Feature | Coefficient (β) |
|---|---|
| Intercept (β₀) | −2.10 |
| fico_range_low_WOE (β₁) | −0.91 |
| int_rate_WOE (β₂) | −0.89 |
| dti_WOE (β₃) | −0.73 |
| revol_util_WOE (β₄) | −0.70 |
| delinq_2yrs_WOE (β₅) | −0.62 |

**Step 1: Compute z (log-odds):**

```
z = −2.10
  + (−0.91 × 1.20)    ← FICO contribution     = −1.092
  + (−0.89 × −0.85)   ← int_rate contribution = +0.757
  + (−0.73 × −0.60)   ← DTI contribution      = +0.438
  + (−0.70 × −0.40)   ← revol_util contribution = +0.280
  + (−0.62 × 0.80)    ← delinq contribution   = −0.496

z = −2.10 − 1.092 + 0.757 + 0.438 + 0.280 − 0.496
z = −2.213
```

**Step 2: Convert z to probability:**

```
PD = 1 / (1 + e^(−z)) = 1 / (1 + e^(2.213)) = 1 / (1 + 9.141) = 1 / 10.141 = 0.0986
```

**PD = 9.86%** — this borrower has approximately a 1-in-10 chance of defaulting in the next year. A prime lender would likely decline this application; a near-prime lender might approve with elevated pricing.

---

### Example 3: Converting PD to a Credit Score

Using the scorecard parameters:
- Base Score = 600, Base Odds = 50:1, PDO = 20

```
Factor = 20 / ln(2) = 20 / 0.6931 = 28.854
Offset = 600 − 28.854 × ln(50) = 600 − 28.854 × 3.912 = 600 − 112.85 = 487.15

Odds = (1 − PD) / PD = (1 − 0.0986) / 0.0986 = 0.9014 / 0.0986 = 9.143 (good-to-bad)

Score = Offset + Factor × ln(Odds)
      = 487.15 + 28.854 × ln(9.143)
      = 487.15 + 28.854 × 2.213
      = 487.15 + 63.87
      = 551 points
```

A score of 551 corresponds to the "near-prime" / "subprime" band. A lender with a cutoff of 620 would decline this application.

---

### Example 4: Computing KS by Hand (Small Example)

**Setup:** 10 borrowers sorted by descending predicted PD:

| Rank | Predicted PD | Actual Default? | Cum. % Defaults | Cum. % Non-Defaults | Gap |
|---|---|---|---|---|---|
| 1 | 0.85 | 1 | 1/4 = 25% | 0/6 = 0% | **25%** |
| 2 | 0.72 | 1 | 2/4 = 50% | 0/6 = 0% | **50%** |
| 3 | 0.61 | 0 | 2/4 = 50% | 1/6 = 17% | **33%** |
| 4 | 0.55 | 1 | 3/4 = 75% | 1/6 = 17% | **58%** ← **KS = 58%** |
| 5 | 0.48 | 0 | 3/4 = 75% | 2/6 = 33% | 42% |
| 6 | 0.35 | 0 | 3/4 = 75% | 3/6 = 50% | 25% |
| 7 | 0.28 | 0 | 3/4 = 75% | 4/6 = 67% | 8% |
| 8 | 0.19 | 1 | 4/4 = 100% | 4/6 = 67% | 33% |
| 9 | 0.12 | 0 | 4/4 = 100% | 5/6 = 83% | 17% |
| 10 | 0.05 | 0 | 4/4 = 100% | 6/6 = 100% | 0% |

**KS = 58%** at rank 4 (threshold = 0.55). This means: if we screen only the top 40% of riskiest borrowers (ranks 1–4), we capture 75% of all actual defaults while only "capturing" 17% of non-defaults. The separation of 58 percentage points is excellent (though with only 10 samples, this is illustrative only).

---

## 13. Business and Risk Interpretation

### How Lenders Use the PD Model

The model output flows through the entire lending decision process:

**1. Application Scoring:**
Every new loan application is scored through the model. The score (or raw PD) determines the application's fate:

```
Score > 680 → Approve at standard rate
600 < Score ≤ 680 → Approve at risk-based premium rate
540 < Score ≤ 600 → Refer to manual underwriting
Score ≤ 540 → Decline
```

These cutoffs are calibrated to the lender's risk appetite, regulatory requirements, and profitability targets. Different lenders have very different cutoffs.

**2. Risk-Based Pricing:**

```
Loan Interest Rate = Base Rate + Credit Spread

Credit Spread = f(PD, LGD, Operating Costs, Target Return on Capital)
              ≈ EL / (1 − EL) + Capital Cost
```

A borrower with PD = 2% might receive a rate of 8.5%, while a borrower with PD = 15% might receive 22%. The higher rate is not punitive — it is the price that compensates the lender for the additional expected loss.

**3. Portfolio Monitoring:**
The PD model is run monthly on the entire active loan portfolio. Rising average PD signals deteriorating portfolio quality and triggers early warning processes.

### Expected Loss in Practice

Returning to our worked example (PD = 9.86%, loan amount = $20,000, LGD = 75%):

```
EL = PD × LGD × EAD
   = 0.0986 × 0.75 × $20,000
   = $1,479 per year
```

To break even on this loan, the interest income must cover at minimum:
- EL = $1,479/year
- Funding cost (cost of capital used to fund the loan)
- Operating expenses (origination, servicing, collections)

For a $20,000 loan over 36 months, these costs must be recovered through the interest rate charged.

### Portfolio-Level Expected Loss

For a portfolio of N loans:

```
Portfolio EL = Σᵢ (PDᵢ × LGDᵢ × EADᵢ)
```

Under IFRS 9, this sum (with appropriate forward-looking adjustments) must be recognized as a **provision** on the bank's balance sheet. Rising provisions reduce reported profits — a primary mechanism by which credit losses affect bank earnings.

### Setting the Optimal Cutoff

The classic decision-theory framework for cutoff selection:

Let:
- `L` = Loss if default is approved (FN cost) ≈ LGD × EAD
- `R` = Revenue if good borrower is approved (TN benefit)
- `c` = Opportunity cost if good borrower is declined (FP cost) ≈ R

Approve if the expected value of approving is positive:

```
(1 − PD) × R − PD × L > 0
(1 − PD) × R > PD × L
R / L > PD / (1 − PD)
```

This gives the **break-even PD** (cutoff):

```
PD_cutoff = L / (R + L)
```

**Numerical example:**
- A $10,000 personal loan generates $2,000 in net interest income over its life (R = $2,000)
- If defaulted with LGD = 75%: Loss = 0.75 × $10,000 = $7,500 (L = $7,500)

```
PD_cutoff = 7,500 / (2,000 + 7,500) = 7,500 / 9,500 = 78.9%
```

This extreme result (79% cutoff) suggests the lender could approve almost everyone from a pure expected-value standpoint. But real-world cutoffs are far lower because this model ignores:
- Capital costs (banks must hold capital against RWA)
- Regulatory requirements
- Concentration risk
- Operational constraints
- Risk appetite policies

In practice, prime lenders apply cutoffs corresponding to PD 1–3%, accepting that they decline some economically viable loans in exchange for capital efficiency and risk management discipline.

### Regulatory Cutoffs and Fair Lending

An important constraint: cutoff scores must be tested for **disparate impact** under the Equal Credit Opportunity Act (ECOA) and Fair Housing Act. If a neutral-sounding variable like ZIP code functions as a proxy for race, using it may violate fair lending laws regardless of its predictive power.

Modern credit models must:
1. Exclude protected class variables (race, gender, religion, national origin, age, marital status, family status)
2. Test for disparate impact of the model as a whole
3. Provide adverse action reasons to declined applicants (e.g., "too many delinquencies," "high revolving utilization")

---

## 14. Model Validation — PSI, OOT, and Vintage Analysis

### Why Validation Is Non-Negotiable

Building a model that performs well on the training data is the easy part. The hard part is ensuring it performs well on **future data** — borrowers who apply months or years after the model was built, in a potentially different economic environment.

Model validation answers the question: *is this model still fit for purpose?*

The Federal Reserve's **SR 11-7** guidance (2011) requires banks to perform independent, comprehensive model validation including conceptual soundness review, ongoing monitoring, and outcomes analysis. Failure to validate models adequately was identified as a contributing factor to the 2008 financial crisis.

---

### Population Stability Index (PSI)

PSI detects **input drift** — whether the distribution of borrowers applying today looks different from the borrowers on whom the model was trained.

**Why this matters:** A model trained on 2018–2019 data may have been calibrated on a specific economic environment. If the 2022 applicant pool has significantly higher average DTI and lower FICO scores (perhaps due to inflation), the model's predictions may no longer be accurate.

**Computing PSI Step by Step:**

1. Take the score distribution from the development (training) period. Divide it into 10 equal-frequency bins.
2. Observe the score distribution on a new, current population.
3. For each bin `i`:
   ```
   PSI_i = (Actual_i% − Expected_i%) × ln(Actual_i% / Expected_i%)
   ```
4. Sum across all bins: `PSI = Σ PSI_i`

**Manual Example (10 bins, simplified to 5 for illustration):**

| Bin | Expected% | Actual% | Actual% − Expected% | ln(Actual/Expected) | PSI contribution |
|---|---|---|---|---|---|
| Bin 1 (score 300–400) | 10% | 15% | +5% | ln(1.50) = +0.405 | 0.05 × 0.405 = **0.020** |
| Bin 2 (score 400–500) | 20% | 25% | +5% | ln(1.25) = +0.223 | 0.05 × 0.223 = **0.011** |
| Bin 3 (score 500–600) | 40% | 35% | −5% | ln(0.875) = −0.134 | −0.05 × −0.134 = **0.007** |
| Bin 4 (score 600–700) | 20% | 18% | −2% | ln(0.90) = −0.105 | −0.02 × −0.105 = **0.002** |
| Bin 5 (score 700–850) | 10% | 7% | −3% | ln(0.70) = −0.357 | −0.03 × −0.357 = **0.011** |
| **Total** | | | | | **PSI = 0.051** |

PSI = 0.051 → **Stable** (< 0.10). The population has not shifted significantly.

**PSI Thresholds (Industry Standard):**

| PSI | Signal | Action |
|---|---|---|
| < 0.10 | Green: Stable | Continue using the model |
| 0.10–0.25 | Amber: Moderate shift | Investigate cause; monitor more frequently |
| > 0.25 | Red: Significant shift | Consider model recalibration or full rebuild |

---

### Out-of-Time (OOT) Validation

OOT validation tests model performance on borrowers from a **time period not used in model development.** It is the gold standard for credit model validation.

**Why OOT and not just a holdout set?**

A random holdout (say, 20% of the training data) is drawn from the same time period as the training set. Borrowers in 2019 have similar economic conditions whether they are in the training set or holdout set. A model that fits the 2017–2019 population well will also score well on a random holdout from that same period.

OOT validation (e.g., using 2020–2021 originations as the test set) exposes the model to genuinely unseen conditions — different economic environment, different mix of borrowers, potentially different lender underwriting policies.

**Standard OOT Protocol:**

```
Development window: Originations Jan 2017 – Dec 2018
                    Performance observation: 12 months from origination
                    → Labeling complete by Dec 2019

OOT window:         Originations Jan 2019 – Dec 2019
                    Performance observation: 12 months from origination
                    → Labeling complete by Dec 2020

The OOT window must be fully matured (all loans have had
their full 12-month performance observation period completed)
before OOT AUC/KS can be computed.
```

**Acceptable Performance Degradation from Development to OOT:**

| Metric | Acceptable Degradation |
|---|---|
| AUC | < 3–5 percentage points |
| KS | < 5 percentage points |
| Gini | < 5–10 percentage points |

If degradation exceeds these thresholds, the model likely overfit to the development period's conditions and requires recalibration (adjusting coefficients to the new population) or full rebuild.

---

### Vintage Analysis

A **vintage** is a cohort of loans originated in the same calendar period (typically a month or quarter). Vintage analysis tracks how each cohort's cumulative default rate evolves over time (as the loans age), providing:

1. **A validation tool:** Are new vintages defaulting at rates consistent with model predictions?
2. **An underwriting quality monitor:** Are we maintaining consistent credit standards across time?
3. **An early warning system:** Rapid early deterioration in a new vintage signals a potential problem.

**Reading a Vintage Chart:**

```
Cumulative Default Rate (%)

12% |                       2020 Q4 (peak COVID vintage)
    |                   ___/
 9% |    2019 Q3   ___-/
    |           __/ 2021 Q1
 6% |        __/ 2022 Q2
    |     __/
 3% | ___/ 2023 Q1 (recent vintage; still maturing)
    |/
 0% +────────────────────────────────────→ Loan Age (months)
    0    6   12   18   24   30   36
```

**Reading this chart:**
- The 2020 Q4 vintage is performing worst (highest cumulative defaults at every age) — COVID-19 shock originated in this period
- The 2023 Q1 vintage is newest and still maturing; early data looks favorable
- All vintages show the typical **S-curve shape**: slow initial default accumulation, rapid middle-period defaults (peak default period 12–24 months for personal loans), then plateauing

**Computing vintage curves in Python:**

```python
def compute_vintage_curves(df: pd.DataFrame,
                            orig_date_col: str,
                            default_col: str,
                            loan_id_col: str) -> pd.DataFrame:
    """
    Compute cumulative default rates by vintage cohort.
    df must contain: origination date, a default flag, and a loan ID.
    """
    df[orig_date_col] = pd.to_datetime(df[orig_date_col])
    df['vintage']     = df[orig_date_col].dt.to_period('Q')  # Quarterly vintages

    # Count originations per vintage
    orig_counts = df.groupby('vintage')[loan_id_col].count()

    # Compute cumulative default rate at each age (months)
    results = []
    for vintage, group in df.groupby('vintage'):
        n_orig = len(group)
        cum_def_rate = group[default_col].cumsum() / n_orig
        results.append({
            'vintage': str(vintage),
            'n_loans': n_orig,
            'cum_def_rate_12m': group[default_col].sum() / n_orig
        })

    return pd.DataFrame(results)
```

---

## 15. Exercises and Further Exploration

### Beginner Exercises

**Exercise 1 — WOE Calculation:**
The `annual_inc` variable is split into four bins. Bin 1 (income < $30,000): 60 defaults, 90 non-defaults. Bin 2 ($30,000–$55,000): 50 defaults, 200 non-defaults. Bin 3 ($55,000–$90,000): 30 defaults, 350 non-defaults. Bin 4 (> $90,000): 10 defaults, 260 non-defaults.

Total: 150 defaults, 900 non-defaults.

Calculate WOE and IV for `annual_inc`. Is this a strong predictor?

**Exercise 2 — PD from Logit:**
A logistic regression model has intercept = −1.5 and one predictor (FICO_WOE) with coefficient = −0.80. A borrower falls in a bin with WOE = −1.20 (high-risk FICO bin).

Compute the predicted PD for this borrower.

**Exercise 3 — Expected Loss:**
A bank has three loans in its portfolio:

| Loan | PD | LGD | EAD |
|---|---|---|---|
| A | 2% | 45% | $500,000 |
| B | 8% | 70% | $50,000 |
| C | 25% | 80% | $20,000 |

Compute the EL for each loan and the total portfolio EL. Which loan contributes the most EL?

---

### Intermediate Exercises

**Exercise 4 — Confusion Matrix Costs:**
Your model, at a threshold of 0.30, produces: TP = 1,200, TN = 8,500, FP = 1,500, FN = 300. Each FN costs $3,500 (average credit loss). Each FP costs $200 (opportunity cost of declined good loan). What is the total model cost? What happens to this cost if you move the threshold to 0.20?

**Exercise 5 — PSI Calculation:**
Development score distribution (10 bins, 10% each). Current distribution: bins 1–3 each have 15%, bins 4–7 each have 10%, bins 8–10 each have 5%.
Compute PSI. Should this model be rebuilt?

**Exercise 6 — Optimal Threshold:**
A credit card product generates $500 net revenue per account over its life if the borrower repays. The average loss if the borrower defaults is $2,000 (after recoveries). Using the break-even PD formula, compute the theoretical optimal cutoff. Why would a real bank apply a lower cutoff than this?

---

### Advanced Exercises

**Exercise 7 — Build an Improved Model:**
Download the Home Credit Default Risk dataset from Kaggle (link in references). The dataset includes multiple auxiliary tables (bureau data, previous applications, credit card balances). Engineer at least 10 features from the auxiliary tables (e.g., number of late payments in bureau history, average revolving utilization over past 12 months). Apply the full pipeline from this guide. Target: achieve AUC > 0.75 and KS > 0.35.

**Exercise 8 — SMOTE vs. Class Weighting:**
Using the Lending Club data, compare two approaches to handling class imbalance:
(a) `class_weight='balanced'` in logistic regression (no resampling)
(b) SMOTE oversampling of the training set (from `imbalanced-learn`)

Evaluate both approaches on an out-of-sample test set. Compare AUC, KS, and Gini. Which approach better preserves model calibration (i.e., does the average predicted PD match the observed default rate)?

**Exercise 9 — Vintage Analysis:**
Filter the Lending Club dataset to include only loans with complete 36-month performance histories. Create quarterly vintage cohorts (loans originated in the same quarter). Plot cumulative 12-month and 24-month default rates for each vintage. Which vintage performed worst? Can you correlate this to macroeconomic conditions in that period?

---

## 16. References and Further Reading

### Primary Academic and Regulatory Sources

1. Basel Committee on Banking Supervision. (2004). *International Convergence of Capital Measurement and Capital Standards: A Revised Framework* (Basel II). Bank for International Settlements. [https://www.bis.org/publ/bcbs128.htm](https://www.bis.org/publ/bcbs128.htm)

2. Basel Committee on Banking Supervision. (2005). *An Explanatory Note on the Basel II IRB Risk Weight Functions*. Bank for International Settlements. [https://www.bis.org/bcbs/irbriskweight.pdf](https://www.bis.org/bcbs/irbriskweight.pdf)

3. Board of Governors of the Federal Reserve System. (2011). *SR 11-7: Supervisory Guidance on Model Risk Management*. [https://www.federalreserve.gov/supervisionreg/srletters/sr1107.htm](https://www.federalreserve.gov/supervisionreg/srletters/sr1107.htm)

4. Lundberg, S. M., & Lee, S.-I. (2017). *A Unified Approach to Interpreting Model Predictions*. NeurIPS 2017. [https://proceedings.neurips.cc/paper/2017/file/8a20a8621978632d76c43dfd28b67767-Paper.pdf](https://proceedings.neurips.cc/paper/2017/file/8a20a8621978632d76c43dfd28b67767-Paper.pdf)

5. Vasicek, O. (2002). *Loan Portfolio Value*. Risk Magazine. The foundational paper for the Basel ASRF model.

6. Thomas, L. C., Edelman, D. B., & Crook, J. N. (2002). *Credit Scoring and Its Applications*. SIAM. The definitive textbook on credit scoring.

7. Siddiqi, N. (2006). *Credit Risk Scorecards: Developing and Implementing Intelligent Credit Scoring*. Wiley. The standard practitioner reference.

### Educational Resources

8. Anderson, R. (2007). *The Credit Scoring Toolkit*. Oxford University Press.

9. Mays, E. (ed.) (2004). *Credit Scoring for Risk Managers*. Thomson South-Western.

10. ListenData: Weight of Evidence and Information Value Explained. [https://www.listendata.com/2015/03/weight-of-evidence-woe-and-information.html](https://www.listendata.com/2015/03/weight-of-evidence-woe-and-information.html)

11. ListenData: Population Stability Index. [https://www.listendata.com/2015/05/population-stability-index.html](https://www.listendata.com/2015/05/population-stability-index.html)

12. ListenData: Gini, CAP, and AUC in Credit Scoring. [https://www.listendata.com/2019/09/gini-cumulative-accuracy-profile-auc.html](https://www.listendata.com/2019/09/gini-cumulative-accuracy-profile-auc.html)

13. ListenData: Vintage Analysis. [https://www.listendata.com/2019/09/credit-risk-vintage-analysis.html](https://www.listendata.com/2019/09/credit-risk-vintage-analysis.html)

14. ListenData: Datasets for Credit Risk Modeling. [https://www.listendata.com/2019/08/datasets-for-credit-risk-modeling.html](https://www.listendata.com/2019/08/datasets-for-credit-risk-modeling.html)

### Datasets

15. Lending Club Loan Data (2007–2020). Kaggle. [https://www.kaggle.com/datasets/wordsforthewise/lending-club](https://www.kaggle.com/datasets/wordsforthewise/lending-club)

16. German Credit Risk Dataset (Statlog). UCI Machine Learning Repository / Kaggle. [https://www.kaggle.com/datasets/uciml/german-credit](https://www.kaggle.com/datasets/uciml/german-credit)

17. Home Credit Default Risk. Kaggle Competition. [https://www.kaggle.com/c/home-credit-default-risk](https://www.kaggle.com/c/home-credit-default-risk)

18. Give Me Some Credit. Kaggle Competition. [https://www.kaggle.com/c/GiveMeSomeCredit](https://www.kaggle.com/c/GiveMeSomeCredit)

### Python Libraries

19. `scorecardpy` — Credit scorecard development in Python (WOE, IV, scorecard scaling). [https://github.com/ShichenXie/scorecardpy](https://github.com/ShichenXie/scorecardpy)

20. `optbinning` — Optimal binning with monotonicity constraints. [https://github.com/guillermo-navas-palencia/optbinning](https://github.com/guillermo-navas-palencia/optbinning)

21. `imbalanced-learn` — SMOTE and other resampling strategies. [https://imbalanced-learn.org](https://imbalanced-learn.org)

22. `shap` — SHAP values for model explainability. [https://github.com/slundberg/shap](https://github.com/slundberg/shap)

### Regulatory Documents (Advanced)

23. European Banking Authority. (2017). *Guidelines on PD estimation, LGD estimation and treatment of defaulted exposures*. EBA/GL/2017/16.

24. International Accounting Standards Board. (2014). *IFRS 9 Financial Instruments*. Full standard available at [https://www.ifrs.org](https://www.ifrs.org)

25. OSFI. (2026). *Capital Adequacy Requirements (CAR) — Chapter 5: Credit Risk — Internal Ratings-Based Approach*. [https://www.osfi-bsif.gc.ca/en/guidance/guidance-library/capital-adequacy-requirements-car-2026-chapter-5](https://www.osfi-bsif.gc.ca/en/guidance/guidance-library/capital-adequacy-requirements-car-2026-chapter-5)

26. Corporate Finance Institute: Exposure at Default (EAD). [https://corporatefinanceinstitute.com/resources/commercial-lending/exposure-at-default-ead/](https://corporatefinanceinstitute.com/resources/commercial-lending/exposure-at-default-ead/)

---

*End of Guide — Credit Risk Modeling: Estimating Probability of Default*

> **A final thought from your instructor:** Credit risk modeling sits at the intersection of statistics, economics, law, and ethics. The model you build determines who gets access to credit and at what price — decisions that profoundly affect people's lives. Build carefully, validate rigorously, and never stop asking whether your model is fair as well as accurate.
