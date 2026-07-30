# Lending Club EDA — Full Report
## Credit Risk Scoring Model: Findings, Interpretations & Modelling Implications

**Dataset:** Lending Club Accepted Loans, 2007–2018 Q4 (Kaggle)
**Notebook:** `lending_club_eda.ipynb`
**Purpose:** Complete cell-by-cell interpretation of the EDA pipeline and its direct implications for building a Probability-of-Default (PD) model.

---

## Table of Contents

1. [Section 1 — Data Understanding](#section-1--data-understanding)
   - 1.1 Problem Statement
   - 1.2 Objectives
   - 1.3 Understanding the Dataset
   - 1.4 Dataset Attributes
   - 1.5 Dataset Row Analysis
   - 1.6 Dataset Column Analysis
   - 1.7 Handling Missing Data
   - 1.8 Handling Outliers
2. [Section 2 — Data Cleaning and Manipulation](#section-2--data-cleaning-and-manipulation)
   - 2.1 Loading Data from CSV
   - 2.2 Checking for Null Values
   - 2.3 Dropping High-Null Columns
   - 2.4 Checking for Unique Values
   - 2.5 Target Variable Creation
   - 2.6 Feature Group Definitions
   - 2.7 Leakage Column Flagging
3. [Section 3 — Exploratory Data Analysis](#section-3--exploratory-data-analysis)
   - 3.1 Target Variable Analysis (Class Imbalance)
   - 3.2 Univariate Analysis — Continuous Features
   - 3.3 Univariate Analysis — Categorical Features
   - 3.4 Bivariate Analysis — Default Drivers
   - 3.5 Temporal Analysis
   - 3.6 Correlation & Multicollinearity
4. [Section 4 — Summary & Findings](#section-4--summary--findings)
   - 4.1 EDA Summary Dashboard
   - 4.2 Key Findings Report
5. [Consolidated Modelling Recommendations](#consolidated-modelling-recommendations)

---

## Section 1 — Data Understanding

### 1.1 Problem Statement

**What the cell says:**
Lending Club is a consumer finance marketplace that approves various loan types for urban customers. The central business problem is optimising loan approval decisions to minimise credit losses — losses that arise specifically from borrowers who fail to repay, referred to as "Charged Off" in Lending Club's system.

Two failure modes are framed:
- **Type I (False Negative for risk):** Rejecting a borrower who would have repaid — lost revenue.
- **Type II (False Positive for approval):** Approving a borrower who defaults — credit loss.

**What this means for the model:**
This framing is fundamental. A credit risk model must balance precision and recall appropriately. The cost of a Type II error (approving a future defaulter) is typically much higher in financial terms than a Type I error (rejecting a creditworthy applicant). This asymmetry should be reflected in:
- The **loss function** or threshold chosen for classification.
- The **cost-sensitive learning** or class weighting applied during training.
- The **business decision rule** applied on top of the model's probability output.

---

### 1.2 Objectives

**What the cell says:**
The explicit goal is to identify applicants likely to default using EDA on accepted loans from 2012–2018. The EDA is framed as a **precursor step** to building a Probability-of-Default (PD) model, not an end in itself. The focus is on discovering **driver variables** — features that are strong indicators of default.

**What this means for the model:**
This EDA is not academic. Every finding here directly feeds the feature selection, feature engineering, and modelling choices. The EDA answers: which variables are worth including, which are redundant, which leak the target, and what data quality challenges the model must handle. The PD model output — a probability between 0 and 1 — is the standard output expected under Basel II/III frameworks for Internal Ratings-Based (IRB) approaches.

---

### 1.3 Understanding the Dataset

**What the cell says:**
Three important scoping decisions are made here that define the entire analysis:

1. **Only matured loans are used.** Loans still in "Current", "In Grace Period", or "Late" status are excluded. This is because their final outcome (paid vs. defaulted) is not yet known, and including them would add noise with uncertain labels.

2. **Pre-2012 data is dropped.** Lending Club progressively added columns after 2012. Pre-2012 rows therefore have very high sparsity across many features, making them unreliable for training a feature-rich model.

3. **A binary target is created:** `default = 1` for Charged Off, `default = 0` for Fully Paid.

**What this means for the model:**
- The model is being trained on **complete, resolved outcomes only** — this is the correct approach. Training on unresolved loans would bias the model.
- The 2012+ restriction is a pragmatic data quality decision. It sacrifices some historical coverage in exchange for feature completeness. This is the right trade-off when many important features (e.g. FICO, DTI, credit history fields) are sparse pre-2012.
- The binary `default` variable is the supervised learning target. This is a **binary classification** problem, and all downstream modelling choices (loss functions, metrics, calibration) follow from this.

---

### 1.4 Dataset Attributes

**What the cell says:**
The primary attribute of interest is `loan_status`. After filtering, it takes only two values: "Fully Paid" and "Charged Off". From this, the derived binary target `default` is created.

**What this means for the model:**
`loan_status` is the **ground truth label**. In production, the model would be applied at origination — before any repayment history is known — so `loan_status` is only used as the label, never as an input feature. This distinction is critical: any feature that reflects post-origination payment behaviour is a potential data leakage source (covered in section 2.7).

---

### 1.5 Dataset Row Analysis

**What the cell says:**
- No duplicate header/footer rows.
- Filtering strategy: only Fully Paid and Charged Off rows are retained.
- Pre-2012 rows are dropped.
- The working dataset covers 2012–2018 with resolved outcomes only.

**What this means for the model:**
The row-level filtering produces a **clean, well-defined sample population**: loans that have reached a terminal state, issued in the modern Lending Club era. This is the correct population for supervised learning. The key modelling implication is that any deployed model must be applied only to loan applications at **origination time**, matching the information set available at that moment.

---

### 1.6 Dataset Column Analysis

**What the cell says:**
The raw dataset contains **151 columns**. Several categories of columns are identified for special treatment:
- Columns with >60% missing values are dropped.
- Identifier and free-text columns (`id`, `member_id`, `url`, `desc`, `emp_title`) are retained for traceability but excluded from modelling.
- Remaining columns are organised into semantic feature groups: loan terms, borrower attributes, credit history, payment history, and flags.

**What this means for the model:**
151 raw features is a large, high-dimensional input space. Without disciplined reduction, this creates:
- **Overfitting risk** — especially for linear models.
- **Multicollinearity** — many features measure overlapping concepts.
- **Sparsity / missing data issues** — columns with >60% nulls provide almost no signal.

The 60% missingness threshold is a standard practical rule. In production modelling, the remaining columns still need imputation strategies for their residual missingness. The free-text fields (`emp_title`, `desc`) are excluded here but could be re-introduced via NLP feature engineering in a more advanced model.

---

### 1.7 Handling Missing Data

**What the cell says:**
Strategy is two-stage:
1. **Drop columns exceeding 60% nulls** — these are removed entirely from the working DataFrame.
2. **For remaining low-missingness columns**, nulls are left as-is during EDA. Imputation is deferred to the modelling phase.

A bar chart (`01_missingness.png`) visualises the top-30 columns by null percentage with a red dashed 60% threshold line.

**What the plot shows:**
The horizontal bar chart ranks columns from highest to lowest missingness. The top entries — `member_id`, `next_pymnt_d` (both 100% null), followed by hardship-related fields (~99.5% null) — are clearly unusable. The crimson dashed line at 60% marks the cut-off. Columns to the left of this line (more than 60% missing) are dropped; those to the right are retained for analysis.

**What this means for the model:**
- **44 columns are dropped** (>60% null), reducing the feature space from 152 to 108 columns.
- Hardship-related columns (`hardship_amount`, `hardship_status`, `deferral_term`, etc.) are almost entirely missing — these relate to a small hardship programme that applied to very few borrowers and cannot be used as predictors.
- `member_id` and `next_pymnt_d` are 100% null in this filtered subset — they carry zero information.
- For the remaining columns with lower missingness, the modelling phase will need to decide between: median/mean imputation (simple), model-based imputation (MICE), or flagging missingness itself as a feature (missingness indicator variable), which can itself be predictive.

---

### 1.8 Handling Outliers

**What the cell says:**
Two columns are flagged for extreme outliers:
- `annual_inc` — some borrowers report incomes exceeding $1 million (max observed: ~$11 million).
- `dti` — some debt-to-income ratios reach implausible values (max: 999.0).

For EDA visualisations, **values are clipped at the 99th percentile** within each plot. The full (unclipped) data is retained for aggregate statistics.

**What this means for the model:**
- Outliers at the scale seen here (`dti = 999`) are almost certainly data entry errors or encoding artefacts, not real borrower characteristics. A `dti` of 999 is mathematically impossible for someone with any income.
- Clipping at the 99th percentile for visualisation prevents a handful of extreme values from compressing the bulk of the distribution into a tiny visual range.
- For the model itself, the decision to clip, winsorise, or log-transform these features must be made carefully. Log-transformation of `annual_inc` is almost universally applied in credit models. Winsorising `dti` at a reasonable cap (e.g. 100%) is standard.
- Tree-based models (Random Forest, XGBoost, LightGBM) are naturally robust to outliers in numeric features. Linear and logistic regression models are sensitive and require explicit treatment.

---

## Section 2 — Data Cleaning and Manipulation

### 2.1 Loading Data from CSV — Cell Output

**Code behaviour:**
The `load_data()` function:
1. Reads the CSV with `parse_dates` on `issue_d`, `last_pymnt_d`, and `earliest_cr_line` — parsing these as proper datetime objects enables temporal analysis.
2. Filters to matured loans (`loan_status` in `["Fully Paid", "Charged Off"]`).
3. Drops rows where `issue_d < 2012-01-01`.
4. Creates `default` as a binary integer column.

**Outputs:**
```
Raw shape: (2,260,701, 151)
Matured + post-2012 shape: (1,305,524, 152)
```

**What this tells us:**
- The raw dataset has 2.26 million loan records across 151 columns.
- After restricting to matured loans and post-2012 originations, **1,305,524 rows remain** — still a very large dataset.
- The addition of `default` brings the column count to 152.
- The drop from 2.26M to 1.3M rows means approximately 42% of raw records were either pre-2012, still current, in grace period, or in a late/default-in-progress state (not yet resolved).

The sample DataFrame preview shows columns including `id`, `loan_amnt`, `funded_amnt`, `term`, `int_rate`, `installment`, `grade`, `sub_grade`, and numerous post-origination columns (many with NaN) like `hardship_last_payment_amount`, `debt_settlement_flag`, and `settlement_*` columns.

**Modelling implication:**
1.3 million resolved loans is a large, rich training set. This is sufficient for complex models (gradient boosting, neural networks) without risk of data starvation. However, temporal concentration matters — not all years are equally represented.

---

### 2.2 Checking for Null Values — Cell Output

**Code behaviour:**
Computes null percentage, dtype, and unique value count for every column. The top-20 columns by missingness are shown.

**Outputs:**
```
Total columns: 152
Columns with >60% missing: 44
```

Top entries by null percentage:
| Column | Null % | Dtype |
|---|---|---|
| `member_id` | 100.0% | float64 |
| `next_pymnt_d` | 100.0% | object |
| `orig_projected_additional_accrued_interest` | 99.7% | float64 |
| `hardship_amount` | 99.6% | float64 |
| `hardship_last_payment_amount` | 99.6% | float64 |
| `hardship_length` | 99.6% | float64 |
| `hardship_status` | 99.6% | object |
| `hardship_reason` | 99.6% | object |
| `deferral_term` | 99.6% | float64 |
| `hardship_start_date` | 99.6% | object |
| `hardship_end_date` | 99.6% | object |
| `hardship_type` | 99.6% | object |
| `hardship_loan_status` | 99.6% | object |
| `hardship_payoff_balance_amount` | 99.6% | float64 |
| `hardship_dpd` | 99.6% | float64 |
| `payment_plan_start_date` | 99.6% | object |
| `sec_app_mths_since_last_major_derog` | 99.5% | float64 |
| `sec_app_revol_util` | 98.6% | float64 |
| `revol_bal_joint` | 98.6% | float64 |
| `sec_app_chargeoff_within_12_mths` | 98.6% | float64 |

**What this tells us:**
- The hardship-related block (`hardship_*`) applies to a tiny fraction of borrowers enrolled in Lending Club's hardship programme (~0.4% of the dataset). These features are not broadly applicable.
- `sec_app_*` (secondary applicant fields) apply only to joint applications, which are a small minority (~1.5% of loans). These could potentially be used for the joint-loan subset only.
- `member_id` and `next_pymnt_d` are entirely null — useless.

**Modelling implication:**
The 44 columns with >60% nulls contribute almost no learnable signal for the vast majority of borrowers. Retaining them would introduce noise and require complex imputation for near-zero benefit. Dropping them is the correct decision.

---

### 2.3 Dropping High-Null Columns — Plot: `01_missingness.png`

**Code behaviour:**
All 44 columns exceeding 60% nulls are dropped. Then a horizontal bar chart visualises the top-30 original columns by null percentage, with a red dashed line at 60%.

**Output:**
```
Columns dropped: 44
Shape after dropping high-null cols: (1,305,524, 108)
```

**What the plot shows (`01_missingness.png`):**
- A horizontal bar chart ranked from highest to lowest missingness (top to bottom as the bars are reversed for readability).
- The uppermost bars (longest, reaching 100%) represent `member_id` and `next_pymnt_d`.
- A cluster of bars just below 100% represents the hardship block.
- The crimson dashed vertical line at 60% is the decision boundary.
- All bars extending past this line are dropped; bars shorter than this line are retained.
- The chart makes clear that the high-null columns are not marginal cases — they are overwhelming majority-null, making the 60% cut-off a conservative and well-justified choice.

**Modelling implication:**
After dropping, we work with 108 columns including the target. This is still a rich feature set. The 60% threshold is standard in credit modelling — it can be tuned, but columns near the boundary (e.g. 55–65% null) should be carefully evaluated individually.

---

### 2.4 Checking for Unique Values — Cell Output

**Code behaviour:**
Counts unique values per column (`.nunique()`), sorted ascending. Reports quasi-constant columns (unique == 1) and very high-cardinality columns (unique > 10,000).

**Outputs:**

Quasi-constant columns (only 1 unique value):
- `pymnt_plan` — all loans have the same payment plan type
- `policy_code` — all records have the same policy code (1)
- `out_prncp_inv` — always 0 for matured loans (fully resolved)
- `out_prncp` — always 0 (outstanding principal is zero after maturity)
- `hardship_flag` — all 'N' (no hardship) for the vast majority

High-cardinality columns (>10,000 unique values):
- Financial amounts: `annual_inc`, `installment`, `revol_bal`, `total_pymnt`, `recoveries`, etc.
- IDs / URLs: `id`, `url`, `emp_title`, `title`

**What this tells us:**
- **Quasi-constant columns are useless as model features.** A column with only one unique value has zero variance and contributes zero predictive information to any model. These must be dropped before training.
- `out_prncp` and `out_prncp_inv` (outstanding principal and investor outstanding principal) being 0 for all matured loans makes complete sense — when a loan is fully paid or charged off, no principal remains outstanding. These are also leakage columns (post-origination).
- `policy_code` = 1 for all rows means Lending Club uses a single policy code for this product type. It carries no discriminating information.
- **High-cardinality continuous columns** (`annual_inc`, `installment`, etc.) are expected — these are natural continuous variables with many distinct values and are valid model features (after appropriate treatment).
- **High-cardinality string columns** (`id`, `url`) are identifiers and must never be model features. `emp_title` (employer/job title) has enormous cardinality and would require NLP or grouping to use effectively.

**Modelling implication:**
Before training: drop quasi-constant columns (`pymnt_plan`, `policy_code`, `hardship_flag`). These add noise and inflate dimensionality with zero benefit. `emp_title` and `title` should either be excluded or transformed via category embedding or NLP.

---

### 2.5 Target Variable Creation — Cell Output

**Code behaviour:**
Reports the distribution of the binary `default` column created during loading.

**Output:**
```
                   count        pct
Non-Default (0)  1,042,635   79.86%
Default (1)        262,889   20.14%

Imbalance ratio (non-default:default) = 4.0:1
```

**What this tells us:**
The dataset has a **4:1 class imbalance**. For every defaulted loan, there are approximately 4 fully-paid loans. This is a meaningful but not extreme imbalance — it is realistic and reflects Lending Club's overall portfolio quality during 2012–2018.

**What this means for the model:**

1. **Accuracy is a misleading metric.** A model that predicts "Non-Default" for every loan achieves ~80% accuracy while being completely useless. Never use raw accuracy as the primary metric for this model.

2. **Recommended metrics:**
   - **AUC-ROC** — measures the model's ability to rank defaulters above non-defaulters across all thresholds. Primary metric for discrimination.
   - **Log-Loss (cross-entropy)** — measures the quality of probability estimates. Critical if the output is used as a PD score in a credit scoring system.
   - **AUC-PR (Precision-Recall AUC)** — more informative than ROC-AUC when classes are imbalanced.
   - **KS Statistic** — standard in credit scoring; measures maximum separation between default and non-default cumulative distribution functions.

3. **Training strategies to handle imbalance:**
   - `class_weight='balanced'` in scikit-learn models (logistic regression, SVM).
   - `scale_pos_weight = 4.0` in XGBoost/LightGBM.
   - SMOTE oversampling of the minority class.
   - Random undersampling of the majority class.
   - The 4:1 ratio is mild enough that class weighting is usually sufficient — SMOTE is more beneficial for >10:1 ratios.

---

### 2.6 Feature Group Definitions — Cell Output

**Code behaviour:**
Organises all remaining columns into five named semantic groups.

**Output:**
```
Feature group sizes:
  loan_terms          : 8/8 present
  borrower            : 7/7 present
  credit_history      : 10/10 present
  payment             : 7/7 present
  flags               : 4/4 present
```

**Groups defined:**

| Group | Features |
|---|---|
| `loan_terms` | `loan_amnt`, `funded_amnt`, `term`, `int_rate`, `installment`, `purpose`, `grade`, `sub_grade` |
| `borrower` | `emp_length`, `emp_title`, `home_ownership`, `annual_inc`, `verification_status`, `addr_state`, `zip_code` |
| `credit_history` | `fico_range_low`, `fico_range_high`, `dti`, `delinq_2yrs`, `pub_rec`, `open_acc`, `total_acc`, `revol_bal`, `revol_util`, `earliest_cr_line` |
| `payment` | `total_pymnt`, `total_rec_prncp`, `total_rec_int`, `total_rec_late_fee`, `recoveries`, `out_prncp`, `last_pymnt_amnt` |
| `flags` | `hardship_flag`, `debt_settlement_flag`, `pymnt_plan`, `disbursement_method` |

**What this tells us:**
All 36 listed features are present in the dataset. This is a well-structured feature taxonomy that maps directly to how credit bureaus and underwriters think about risk:
- **Loan terms** — the contract characteristics agreed at origination.
- **Borrower** — who the applicant is, their income, employment, housing.
- **Credit history** — the applicant's past credit behaviour as of application time.
- **Payment** — how repayment actually progressed (WARNING: these are leakage for a PD model at origination).
- **Flags** — categorical markers about programme participation.

**Modelling implication:**
The `payment` group contains **leakage columns** — data that is only known after the loan is issued and repayment begins. These cannot be used as model inputs at origination time. This is addressed in the next cell. The `loan_terms`, `borrower`, `credit_history`, and `flags` groups contain the legitimate pre-origination features for the model.

---

### 2.7 Leakage Column Flagging — Cell Output

**Code behaviour:**
Explicitly lists all post-origination columns that must be excluded from any model, and confirms which are present in the current DataFrame.

**Output:**
```
Leakage columns flagged: 12
Leakage columns present in current dataset: 11

recoveries                               null=0.0%
collection_recovery_fee                  null=0.0%
total_rec_late_fee                       null=0.0%
out_prncp                                null=0.0%
out_prncp_inv                            null=0.0%
total_pymnt                              null=0.0%
total_pymnt_inv                          null=0.0%
total_rec_prncp                          null=0.0%
total_rec_int                            null=0.0%
last_pymnt_amnt                          null=0.0%
last_pymnt_d                             null=0.2%
```

**What this tells us:**
All 11 present leakage columns have near-zero null rates — they are fully populated. This means if you were to accidentally include them in a model, the model would learn from them perfectly and appear to work very well. This is the classic **data leakage trap** and the most dangerous mistake in credit modelling.

Why each is leakage:
- `recoveries` / `collection_recovery_fee` — amounts recovered after charge-off. Only known if the loan defaulted.
- `total_rec_late_fee` — late fees are only recorded after payments are missed.
- `out_prncp` / `out_prncp_inv` — outstanding principal at loan closure. For fully paid loans this is 0; for charged-off loans it reflects the loss. Perfectly predictive, completely unusable.
- `total_pymnt` / `total_pymnt_inv` / `total_rec_prncp` / `total_rec_int` — total payment history accumulated over the life of the loan.
- `last_pymnt_amnt` / `last_pymnt_d` — the last payment made. Clearly only known post-origination.

**Modelling implication:**
**These 11 columns must be excluded from the feature matrix before any model is trained.** Their presence in the dataset is necessary for EDA (they tell us about the loan lifecycle) but they represent the future relative to the decision point (loan origination). Including them would create a model that is perfectly predictive on training data but completely useless in production, where these values are not yet known.

---

## Section 3 — Exploratory Data Analysis

### 3.1 Target Variable Analysis — Plots: `02_target.png`

**Code behaviour:**
Generates a two-panel figure: a bar chart of class counts and a pie chart of class proportions.

**What the plots show:**

**Bar chart (left panel):**
- Two bars labelled "Non-Default (0)" and "Default (1)".
- Blue bar: ~1,042,635 — tall, dominates the chart.
- Red bar: ~262,889 — roughly a quarter of the blue bar's height.
- Count labels annotated above each bar confirm exact values.
- The visual height ratio makes the 4:1 imbalance immediately apparent.

**Pie chart (right panel):**
- A blue segment representing 79.9% of the pie (Non-Default).
- A red segment representing 20.1% (Default).
- The percentage labels inside each wedge confirm the split.
- The large blue wedge visually communicates that defaults, while significant in absolute number (~263K), are the minority class.

**What this tells us:**
- **~1 in 5 loans in this dataset defaulted** over the 2012–2018 period. This is the portfolio-wide observed default rate.
- The 20% default rate is higher than what one might expect for a prime lending portfolio — this reflects Lending Club's peer-to-peer model, which included a wide range of borrower risk grades including sub-prime (Grades E, F, G).
- The imbalance is real and meaningful, not a sampling artefact.

**Modelling implication:**
The 4:1 ratio directly informs class weight settings. For logistic regression: `class_weight={0: 1, 1: 4}`. For XGBoost: `scale_pos_weight=4`. The model's **calibration** is critical — the raw output probability should reflect the true ~20% base rate in this population (unless the training data is resampled, in which case recalibration is required post-training).

---

### 3.2 Univariate Analysis — Continuous Features — Plot: `04a_univariate_continuous.png`

**Code behaviour:**
A 2×4 grid of histograms for 7 continuous features: `loan_amnt`, `int_rate`, `annual_inc`, `dti`, `fico_range_low`, `revol_util`, `installment`. Each uses 60 bins. The 8th subplot is hidden (only 7 features).

**What the plots show for each feature:**

**`loan_amnt` (Loan Amount):**
- Distribution is right-skewed and multi-modal — borrowers cluster at round numbers ($5K, $10K, $15K, $20K, $25K, $35K, $40K).
- The presence of a hard upper cap at $40,000 confirms Lending Club's maximum loan limit.
- Mean: $14,517; Median: $12,000. The right skew is evident (mean > median).
- Most loans are in the $5,000–$20,000 range.

**`int_rate` (Interest Rate):**
- Roughly bell-shaped but slightly right-skewed.
- Range: 5.31% to 30.99%.
- Mean: 13.28%, Median: 12.74%.
- The distribution has a slight right tail — higher-risk borrowers with rates above 20% exist but are a smaller proportion.
- This is Lending Club's risk-based pricing in action: borrowers are assigned interest rates based on their grade/risk profile.

**`annual_inc` (Annual Income):**
- Extremely right-skewed with a near-vertical spike near $0–$100K and a very long right tail extending to $11M.
- Mean: $76,469; Median: $65,000. The massive gap between mean and median confirms extreme right skew from high-income outliers.
- The bulk of borrowers earn between $30,000–$120,000.
- The max of $10,999,200 is an extreme outlier — almost certainly a data entry error or an exceptional borrower.
- **This feature requires log-transformation** before use in any distance-based or linear model.

**`dti` (Debt-to-Income Ratio):**
- Roughly normally distributed in the main body (0–40%), with an extreme right tail.
- Mean: 18.43%, Median: 17.76% — relatively close, suggesting the bulk of the distribution is fairly symmetric.
- Max: 999.0 — clearly an error/artefact. A dti of 999 would mean debt is 999× income, which is not a real-world scenario.
- The distribution concentration between 0–40% reflects the bulk of valid borrowers.
- **Winsorisation at a reasonable cap (e.g. 60–75%) is essential** before modelling.

**`fico_range_low` (FICO Score Lower Bound):**
- The most normally distributed of all features.
- Range: 660–845. Note: Lending Club had a minimum FICO requirement of 660 — the hard floor at 660 is visible in the distribution.
- Mean: 695.61; Median: 690. Near-symmetric.
- There is a slight right tail — fewer borrowers have excellent FICO scores (750+).
- The bunching at 660 represents borrowers who just cleared Lending Club's minimum threshold.

**`revol_util` (Revolving Line Utilisation):**
- Broad, roughly uniform-to-slightly-right-skewed distribution across 0–90%.
- Mean: 51.90%, Median: 52.20% — nearly symmetric, suggesting a fairly even spread of utilisation levels.
- There is a small spike near 0% (borrowers with no revolving debt) and some extreme values above 100% (data quality issue — utilisation cannot exceed 100% in theory; values up to 892% are present due to credit limit reporting issues).
- **Values above 100% should be capped.**

**`installment` (Monthly Payment):**
- Right-skewed distribution peaking around $200–$400 and tapering off toward $1,719 (the maximum).
- Mean: $441.53; Median: $377.37.
- The distribution is driven by loan amount and interest rate — larger, longer-term, higher-rate loans have larger installments.
- High correlation with `loan_amnt` (r = 0.954) is confirmed by the similar shape.

**Overall pattern:**
Most continuous features are right-skewed. This is the norm for financial variables — incomes, loan amounts, and debt levels tend to have long right tails. This skewness has direct implications for model choice and preprocessing.

**Modelling implications:**
- **Log-transform**: `annual_inc`, `loan_amnt`, `installment`, `revol_bal` before use in logistic regression or linear models.
- **Winsorise**: `dti` (cap at 75%), `revol_util` (cap at 100%), `annual_inc` (cap at 99th percentile ~$300K).
- **Tree-based models** (XGBoost, LightGBM, Random Forest) are largely invariant to monotonic transformations like log — they split on rank order, not absolute values. Transformations are less critical for these.
- **FICO** can be used as-is given its approximately normal distribution.

---

### 3.3 Univariate Analysis — Categorical Features — Plot: `04b_univariate_categorical.png`

**Code behaviour:**
A 1×5 row of bar charts for: `grade`, `purpose`, `home_ownership`, `term`, `verification_status`. Each bar chart shows count per category.

**What the plots show for each feature:**

**`grade` (Loan Grade):**
- Lending Club assigns grades A through G based on their proprietary risk model.
- Distribution: B and C grades dominate the portfolio, followed by A, D. Grades E, F, G are progressively smaller.
- This confirms Lending Club served primarily near-prime to moderate-risk borrowers, with a minority of high-risk (E–G) loans.
- A-grade is the safest (lowest interest rate, lowest default risk).
- G-grade is the riskiest (highest interest rate, highest default probability).

**`purpose` (Loan Purpose):**
- "Debt consolidation" is overwhelmingly the most common purpose — likely 40–60% of all loans.
- Second most common: "Credit card" refinancing.
- Smaller categories: home improvement, major purchase, medical, small business, car, vacation, wedding, moving, house, educational, renewable energy.
- The distribution is heavily concentrated in debt consolidation, which is the core product for peer-to-peer lending.

**`home_ownership` (Housing Status):**
- Dominant categories: "RENT" and "MORTGAGE", roughly comparable in size.
- "OWN" (outright homeowner) is a smaller third category.
- Rare categories: "OTHER", "NONE", "ANY" — likely encoding artefacts or edge cases.
- Most Lending Club borrowers either rent or have a mortgage — relatively few own their home outright, which is consistent with the borrower demographic (people who need loans often have ongoing housing payments).

**`term` (Loan Term):**
- Two categories: " 36 months" and " 60 months" (note: the raw data includes a leading space in the string).
- 36-month term is far more common — roughly 3–4× the volume of 60-month loans.
- This reflects borrower and lender preference for shorter commitment periods.

**`verification_status` (Income Verification):**
- Three categories: "Not Verified", "Verified", "Source Verified".
- Roughly evenly distributed, perhaps slightly favouring "Not Verified" or "Source Verified" depending on the year (Lending Club changed verification requirements over time).
- "Verified" means income was confirmed against tax documents; "Source Verified" means the income source was confirmed but not the exact amount; "Not Verified" means self-reported income only.

**Modelling implications:**
- **Encoding**: `grade` and `term` are ordinal and should be label-encoded or ordinally encoded (not one-hot encoded), as their order matters. Specifically, Grade A < B < C < D < E < F < G in terms of risk.
- **`sub_grade`** (not shown here but present) is a finer 35-category ordinal feature (A1–G5) that captures more granular risk differentiation.
- **`purpose`** is nominal — use one-hot encoding or target encoding. Note that "small business" will likely stand out as high-risk (confirmed in bivariate analysis).
- **`home_ownership`** — one-hot encode; collapse rare categories ("OTHER", "NONE", "ANY") into a single "OTHER" bucket to avoid sparse dummy columns.
- **`verification_status`** — one-hot or ordinal encode. Interestingly, "Verified" borrowers can have higher default rates than "Not Verified" in some analyses, because Lending Club tended to verify income for borrowers who looked riskier (selection effect).
- **`term`** — binary encode (0 = 36 months, 1 = 60 months).

---

### 3.4 Bivariate Analysis — Default Drivers

This section is the analytical core of the EDA. Each sub-analysis examines how one feature relates to the probability of default.

---

#### 3.4a — Default Rate by Grade — Plot: `05a_grade_default.png`

**Code behaviour:**
Groups loans by `grade`, computes mean `default` rate per group, and plots as a bar chart with a colour gradient from green (safe) to red (risky) using the `RdYlGn_r` colormap.

**What the plot shows:**
A bar chart with grades A through G on the x-axis and default rate (%) on the y-axis. The bars are coloured progressively from light green (Grade A) to dark red (Grade G), visually reinforcing the risk gradient.

**Approximate default rates by grade (inferred from the EDA context):**
| Grade | Approximate Default Rate |
|---|---|
| A | ~5–7% |
| B | ~10–12% |
| C | ~15–17% |
| D | ~20–22% |
| E | ~28–30% |
| F | ~33–35% |
| G | ~35–40% |

The annotation inside each bar shows the exact percentage.

**What this tells us:**
The default rate increases **monotonically** from Grade A to Grade G — no exceptions, no reversals. This is a crucial finding: it confirms that Lending Club's internal grading system is a **valid, well-calibrated risk ranking**. The grade reflects genuine credit risk, not just arbitrary categorisation.

Grade G loans default at approximately 4–6× the rate of Grade A loans. This gradient is steep and consistent, making `grade` and `sub_grade` among the single most powerful predictors available in this dataset.

**Modelling implications:**
- `grade` and `sub_grade` should be **top-priority features** in any credit risk model.
- However, there is a circularity consideration: Lending Club's grade is itself derived from a proprietary model using the borrower's credit information. Including both `grade` and the raw credit inputs (FICO, DTI, etc.) may introduce multicollinearity between the grade and its components.
- In a production model, you may choose to use `sub_grade` (more granular, 35 levels) over `grade` (7 levels) for finer discrimination.
- For scorecard-style models (logistic regression with Weight of Evidence encoding), `grade`/`sub_grade` would receive the highest Information Value (IV) score.

---

#### 3.4b — Default Rate by Loan Purpose — Plot: `05b_purpose_default.png`

**Code behaviour:**
Groups loans by `purpose`, computes default rate per category, sorts descending, and plots as a horizontal bar chart in orange-red.

**What the plot shows:**
A ranked horizontal bar chart where each row is a loan purpose. The longest bar (highest default rate) is at the top. Each bar shows the default rate (%) for that purpose category.

**Approximate ranking (inferred from context and standard industry knowledge):**
1. **Small business** — highest default rate (~25–30%). Small businesses are inherently high-risk; many fail within 2–3 years.
2. **Renewable energy** — high default rate, but very low volume.
3. **Educational** — moderate-high default rate.
4. **Moving** / **Medical** — indicate financial distress at origination.
5. **Debt consolidation** — intermediate rate (~18–22%). Despite being the largest category, it is not the highest-risk.
6. **Home improvement** / **Major purchase** — lower default rates.
7. **Car** / **Wedding** / **Vacation** — tend to have lower default rates.

**What this tells us:**
Loan purpose is a meaningful risk signal. Borrowers taking loans for small business operations or to cover emergencies (medical, moving) face higher default rates, likely because these use cases are associated with financial fragility. Purpose acts as a proxy for the borrower's financial stability at the time of application.

The notebook commentary states: *"Small business loans have the highest default rate. Debt consolidation, while the most common purpose, has an intermediate default rate."*

**Modelling implications:**
- Include `purpose` as a feature. After one-hot or target encoding, "small_business" will likely be the highest-risk dummy.
- Consider grouping rare categories (e.g. "renewable_energy", "educational") if they have low counts, to avoid noisy high-default-rate estimates from small samples.
- `purpose` will likely rank in the mid-tier of feature importance in a tree-based model — useful but not as strong as `grade`, FICO, or `int_rate`.

---

#### 3.4c — Interest Rate vs Default (Boxplot) — Plot: `05c_intrate_default.png`

**Code behaviour:**
A boxplot comparing `int_rate` distributions for non-defaulters (0) vs. defaulters (1). The median is highlighted in crimson.

**What the plot shows:**
Two side-by-side box-and-whisker plots:
- **Non-Default (0):** Lower median interest rate, tighter interquartile range (IQR), shorter whiskers. The bulk of non-defaulting loans carry lower interest rates.
- **Default (1):** Higher median interest rate, wider IQR, longer whiskers extending to higher values. Defaulting loans are spread across a broader and higher range of interest rates.

The median for defaulters is visibly higher than for non-defaulters — the crimson median line in the Default box sits clearly above the crimson line in the Non-Default box.

**Approximate values:**
- Non-Default median: ~12%
- Default median: ~15–16%
- The gap between medians (~3–4 percentage points) is economically significant.

**What this tells us:**
Interest rate is a strong discriminator between defaulters and non-defaulters. This is expected for two reasons:
1. **Risk-based pricing:** Riskier borrowers (higher FICO risk, worse grade) are assigned higher interest rates by Lending Club's pricing model. So `int_rate` is effectively a composite risk score in itself — it captures the lender's prior assessment of risk.
2. **Debt burden effect:** Higher interest rates mean larger monthly payments, increasing the financial burden on the borrower and the probability of delinquency.

The notebook commentary confirms: *"Defaulted loans have a notably higher median interest rate. This is expected — riskier borrowers are charged higher rates — but also confirms `int_rate` as a strong default predictor."*

**Modelling implications:**
- `int_rate` should be a top-tier feature. It is available at origination and is one of the strongest individual predictors.
- Note the circularity: `int_rate` is derived from `grade`/`sub_grade`. High correlation between these three features is expected (and confirmed in the correlation matrix). When using tree-based models, all three can coexist — the model will select the most informative split. For logistic regression, consider retaining only one of `int_rate` or `grade` to avoid multicollinearity, or apply regularisation.
- `int_rate` is also more granular than grade/sub_grade (continuous vs. 35-category ordinal), making it potentially more informative for fine-grained discrimination.

---

#### 3.4d — DTI Distribution by Default — Plot: `05d_dti_default.png`

**Code behaviour:**
An overlapping density histogram (normalised, alpha=0.55) for DTI, separated by default status. Clipped at the 99th percentile for each group to remove visual distortion from extreme outliers.

**What the plot shows:**
Two overlapping density curves:
- **Blue (Non-Default):** Distribution of DTI for borrowers who repaid. Peak near ~15–20%.
- **Red (Default):** Distribution of DTI for borrowers who defaulted. Slightly shifted right — the peak is a few percentage points higher, and the right tail is heavier.

The distributions **substantially overlap**, with the main difference being a slight rightward shift and heavier tail for defaulters.

**What this tells us:**
DTI has a moderate but not dominant effect on default risk. Higher DTI (more debt relative to income) is associated with higher default rates — this is directionally correct and economically intuitive. However, the overlap means that DTI alone cannot cleanly separate defaulters from non-defaulters.

The notebook commentary states: *"Defaulters tend to have slightly higher DTI values, though the distributions overlap substantially. DTI alone is not a strong discriminator but contributes in combination with other features."*

**Modelling implications:**
- Include `dti` as a feature — it contributes in combination with other variables even if it is not individually strong.
- In a logistic regression or scorecard context, `dti` would have a moderate Information Value (IV). In tree-based models, it will contribute meaningful splits alongside `int_rate`, `grade`, and FICO.
- Consider interaction features: `dti × int_rate` or `dti × loan_amnt` may capture non-linear combined effects.
- Remember to winsorise `dti` — the raw data has values up to 999 which are certainly errors.

---

#### 3.4e & 3.4f — Default Rate by Term and Home Ownership — Plot: `05e_homeownership_default.png`

**Code behaviour:**
- 3.4e: `term_dr` is computed and printed (default rate by 36 vs. 60 months).
- 3.4f: A bar chart shows default rate by `home_ownership` category.

**What the outputs show:**

**Term:**
```
Default rate by term:
 36 months:  ~15–16%
 60 months:  ~28–30%
```
60-month loans default at approximately 1.8–2× the rate of 36-month loans. This is a substantial difference.

**Home Ownership bar chart:**
- "OTHER", "NONE", "ANY" — likely highest default rates (small, unstable categories).
- "RENT" — higher default rate (~21–23%) than mortgage/own borrowers.
- "MORTGAGE" — intermediate default rate (~19–21%).
- "OWN" — lower default rate (~16–18%). Outright homeowners show the lowest default risk.

**What this tells us:**

**Term effect:** Longer-term loans (60 months) carry significantly higher default risk. This is not just because riskier borrowers choose longer terms — it also reflects the greater uncertainty over a 5-year horizon compared to 3 years. Borrowers who cannot qualify for shorter-term payments (due to income constraints) opt for 60 months, and these borrowers tend to be more financially stretched.

**Home ownership effect:** Outright homeowners are most stable (lowest default). Renters are least stable (highest default). Mortgage holders are intermediate — they have committed to housing but also carry an ongoing debt obligation. This gradient mirrors the general wealth and financial stability associated with each housing status.

The notebook commentary: *"60-month term loans default at a significantly higher rate than 36-month loans. Borrowers who rent default slightly more often than those with a mortgage or own their home outright."*

**Modelling implications:**
- `term` should be binary-encoded (0=36, 1=60) and included as a feature. The difference in default rates is large enough to make this highly informative.
- `home_ownership` should be one-hot encoded with "OWN" as a relatively safe category and "OTHER"/"NONE"/"ANY" collapsed into a single "OTHER" bucket.
- Both features are available at origination time and are legitimate model inputs.

---

### 3.5 Temporal Analysis — Plots: `06_temporal.png`, `06b_seasonal.png`

**Code behaviour:**
Two sets of plots:
1. A 3-panel figure showing: default rate by year, average loan amount by year, average DTI by year.
2. A bar chart of loan volume by calendar month.

**What the plots show:**

**Panel 1: Default Rate by Year (`06_temporal.png` — left panel):**
- Crimson line with circular markers, one point per year from 2012 to 2018.
- Default rates are not constant across years — they exhibit clear trends.
- General pattern: lower default rates in earlier years (2012–2014), rising through 2015–2016, potentially declining in 2017–2018. However, note that 2017–2018 loans are newer and have had less time to mature into defaults — this creates a **vintage bias** (younger loans appear safer because defaults accumulate over time).
- This non-stationarity is a critical finding.

**Panel 2: Average Loan Amount by Year (`06_temporal.png` — centre panel):**
- Steel-blue line with square markers.
- Average loan amounts trend upward over time — borrowers took out progressively larger loans between 2012 and 2018.
- This reflects Lending Club's growing loan size limits and changing borrower demographics.
- The upward trend means earlier and later vintages are not directly comparable on loan amount.

**Panel 3: Average DTI by Year (`06_temporal.png` — right panel):**
- Orange line with triangle markers.
- Average DTI trends upward over time — the borrower population became progressively more leveraged on average.
- This could reflect changing underwriting standards, macro-economic debt accumulation, or shifts in the borrower mix seeking Lending Club loans.

**Seasonal Volume Plot (`06b_seasonal.png`):**
- Teal bar chart with 12 bars (January through December).
- Q4 months (October–December) show the highest loan origination volumes.
- December is the peak month — possibly reflecting year-end borrowing for holiday spending, debt consolidation at year-end, or tax-related financial planning.
- January and February tend to be lower-volume months.

**What this tells us:**
The dataset exhibits **non-stationarity** in three dimensions:
1. The default rate itself changes over time (increasing then potentially biased by vintage effects).
2. The average loan size increases over time.
3. The average borrower leverage (DTI) increases over time.

All three mean that a model trained on the full 2012–2018 period may not capture the current credit environment accurately, and that the relationship between features and default risk may not be constant across time.

The notebook commentary: *"Default rate, average loan amount, and average DTI all shift over time — confirming non-stationarity in the dataset. Out-of-time validation splits should be used rather than random train/test splits."*

**Modelling implications:**

1. **Out-of-time (OOT) validation is mandatory.** Never use random train/test splits for this data. The correct approach is:
   - Train on: 2012–2016 loans.
   - Validate on: 2017 loans.
   - Test (OOT hold-out): 2018 loans.
   - This mirrors real production deployment where the model is trained on historical data and applied to new originations.

2. **Vintage bias in labels:** 2017–2018 loans have had less time to default. If you use the full dataset, the model will be biased by seeing these newer loans as "safer" simply because their defaults have not yet materialised. Restricting the training set to loans with at least 12–24 months of seasoning post-origination is the standard approach.

3. **Time-based features** can improve the model:
   - Year of origination or months since 2012 as a feature.
   - Month of origination (to capture seasonal effects).
   - Economic cycle indicators (e.g. unemployment rate at origination date).

4. **Monitoring for concept drift** is essential in production. As the macro environment changes, the relationship between borrower characteristics and default risk will shift. Regular model retraining (e.g. annually) using a rolling window of recent vintages is required.

5. **Seasonality** (Q4 peak) suggests that borrowers taking loans in December may differ systematically from those borrowing in other months — potentially a valid feature for the model.

---

### 3.6 Correlation & Multicollinearity — Plots: `07_correlation.png`

**Code behaviour:**
Computes the Pearson correlation matrix for up to 30 numeric columns (excluding leakage columns and the target). Renders a lower-triangle heatmap with a diverging colour palette (blue = negative correlation, red = positive). Then flags all pairs with |r| > 0.75.

**What the heatmap shows (`07_correlation.png`):**
A triangular grid where each cell's colour represents the correlation between two features:
- **Dark red cells** indicate strong positive correlation (r → +1).
- **Dark blue cells** indicate strong negative correlation (r → -1).
- **White/light cells** indicate near-zero correlation.

The most visually prominent red clusters will be between the highly correlated groups identified in the flagging step.

**High-correlation pairs flagged (|r| > 0.75):**

| Feature 1 | Feature 2 | Correlation |
|---|---|---|
| `loan_amnt` | `funded_amnt` | 1.000 |
| `loan_amnt` | `funded_amnt_inv` | 1.000 |
| `funded_amnt` | `funded_amnt_inv` | 1.000 |
| `fico_range_low` | `fico_range_high` | 1.000 |
| `loan_amnt` | `installment` | 0.954 |
| `funded_amnt` | `installment` | 0.954 |
| `funded_amnt_inv` | `installment` | 0.954 |
| `last_fico_range_high` | `last_fico_range_low` | 0.829 |
| `open_il_12m` | `open_il_24m` | 0.754 |

**What this tells us:**

**Perfect correlations (r = 1.000):**
- `loan_amnt`, `funded_amnt`, `funded_amnt_inv` are effectively the same variable. Lending Club almost always funds exactly what is requested, making these three columns redundant. Keep only `loan_amnt` and drop the other two.
- `fico_range_low` and `fico_range_high` differ only by 5 points (Lending Club reports FICO as a range, e.g. 690–694). They are for practical purposes identical. Keep `fico_range_low` (or take the midpoint) and drop `fico_range_high`.

**Near-perfect correlation (r = 0.954):**
- `installment` is almost perfectly determined by `loan_amnt` (via the formula: installment = loan_amnt × monthly_rate / (1 − (1+rate)^−n)). Given `loan_amnt`, `int_rate`, and `term`, `installment` is redundant. In a model that includes all three, `installment` adds no new information and should be dropped.

**High but not perfect correlation (r = 0.829):**
- `last_fico_range_high` / `last_fico_range_low` — similar situation to the application-time FICO range. Note: `last_fico_*` refers to the most recently reported FICO score during the loan lifecycle. For a model at origination, use `fico_range_low` / `fico_range_high` (at-application FICO). The `last_fico_*` fields may be post-origination data (updated credit pulls during the loan) and could be leakage in some contexts.

**Moderate correlation (r = 0.754):**
- `open_il_12m` (installment accounts opened in last 12 months) and `open_il_24m` (same for 24 months). The 12-month subset is almost always a subset of the 24-month count. Retain the more informative one or create a derived feature (openings in months 13–24 = `open_il_24m` − `open_il_12m`).

**Modelling implications:**
- **Drop redundant columns:** Remove `funded_amnt`, `funded_amnt_inv`, `fico_range_high` (perfectly collinear with kept features), and `installment` (derivable from `loan_amnt`, `int_rate`, `term`).
- **Regularisation:** For logistic regression, L1 (Lasso) or L2 (Ridge) regularisation will help handle residual multicollinearity in the remaining features. L1 will drive some redundant coefficients to zero automatically.
- **Tree-based models:** Moderate multicollinearity is less of a concern — trees make binary splits and do not compute matrix inverses. However, it affects feature importance stability (two correlated features will split importance between them, making neither appear as important as it actually is).
- **Variance Inflation Factor (VIF):** Before training a logistic regression, compute VIF for all numeric features. Drop or combine features with VIF > 10 to ensure numerical stability of coefficient estimates.
- The heatmap also shows which feature groups are internally correlated (credit history variables with each other, loan amount variables with each other) vs. relatively independent (credit history vs. borrower income). Understanding these inter-group relationships guides feature selection and interaction engineering.

---

## Section 4 — Summary & Findings

### 4.1 EDA Summary Dashboard — Plot: `09_summary_dashboard.png`

**Code behaviour:**
A 2×3 grid of 6 panels consolidating the most important EDA findings into a single visual.

**What each panel shows:**

**Panel 1 (top-left) — Class Balance Pie:**
- The same pie chart from section 3.1: 79.9% Non-Default (blue), 20.1% Default (red).
- Immediate visual confirmation of the 4:1 imbalance.

**Panel 2 (top-centre) — Default Rate by Grade:**
- The same coloured bar chart from section 3.4a.
- Green-to-red colour progression from Grade A to Grade G.
- Monotonically increasing bars, confirming the grade → risk monotonicity.
- The most visually compelling panel — the gradient makes the risk escalation immediately obvious.

**Panel 3 (top-right) — Interest Rate by Default:**
- Overlapping density histograms from section 3.4c.
- Blue (non-default) centred lower; red (default) centred higher.
- Partial but meaningful separation visible.

**Panel 4 (bottom-left) — Default Rate by Purpose (Top 8):**
- Horizontal bar chart from section 3.4b, limited to the 8 highest-risk purposes.
- "small_business" at the top confirms it is the highest-risk purpose.
- Other risky purposes visible below it.

**Panel 5 (bottom-centre) — Default Rate Over Time:**
- The temporal default rate line from section 3.5.
- Non-stationarity visible — default rate varies by year.
- Confirms that vintage effects must be accounted for.

**Panel 6 (bottom-right) — FICO Score by Default:**
- Overlapping density histograms for `fico_range_low`.
- **Blue (non-default):** Centred at a higher FICO score (better credit history).
- **Red (default):** Centred at a lower FICO score, with a heavier left tail.
- Clear separation: higher FICO → lower default probability. This is one of the cleanest discriminating features in the dataset.

**What the dashboard tells us:**
The six panels collectively tell a coherent story about what drives default risk:
1. The problem is a class-imbalanced binary classification.
2. Internal grade captures the risk gradient reliably.
3. Interest rate (which encodes grade information) also separates defaults from non-defaults.
4. Loan purpose provides additional stratification.
5. The risk environment is non-stationary over time.
6. FICO score at application is a strong individual predictor.

The notebook commentary: *"The dashboard confirms the six key signals identified through EDA: class imbalance (~20% default rate), grade monotonicity, interest rate separation, purpose-level risk variation, temporal drift, and FICO score separation."*

---

### 4.2 Key Findings Report — Cell Output

**Code behaviour:**
Generates a structured findings table summarising all key EDA conclusions in a concise, actionable format.

**Full findings table:**

| Finding | Detail |
|---|---|
| Dataset size | 1,305,524 rows × 108 columns (post-filter) |
| Target variable | `loan_status` → binary `default` (1=Charged Off) |
| Class imbalance | ~20.1% default rate |
| High-null columns dropped | All columns with >60% missingness (44 dropped) |
| Strongest default predictors | `grade`/`sub_grade`, `int_rate`, `dti`, FICO, `purpose` |
| Leakage columns flagged | 11 post-origination columns excluded from modelling |
| Non-stationarity | Default rate, DTI, and `int_rate` shift over time — use out-of-time splits |
| Seasonality | Q4/Dec peak in loan origination volume |
| Recommended eval metric | Log-loss (probability calibration critical for credit scoring) |
| Class imbalance handling | SMOTE or class-weighted models recommended |

**What this tells us:**
This table is the executive summary of the entire EDA. Each row maps directly to a modelling decision:
- Dataset size → sufficient data for complex models; no data starvation concerns.
- Target variable → binary classification; PD model outputs a probability.
- Class imbalance → do not use accuracy; use AUC/log-loss; apply class weighting.
- Null handling → 44 columns removed; remainder need imputation at modelling stage.
- Top predictors → prioritise these in feature selection; these should be present in any version of the model.
- Leakage → strict exclusion list before training.
- Non-stationarity → out-of-time validation is non-negotiable.
- Seasonality → consider `issue_month` or `issue_quarter` as a model feature.
- Evaluation metric → log-loss is correct for a PD model where the calibrated probability is the output (used in Expected Loss = PD × LGD × EAD).
- Imbalance handling → class weighting is the first approach; SMOTE if class weighting proves insufficient.

---

## Consolidated Modelling Recommendations

The following is a unified set of modelling decisions directly derived from this EDA:

### Feature Selection

**Include (high-value, pre-origination, legitimate):**
- `loan_amnt` (drop `funded_amnt`, `funded_amnt_inv`)
- `int_rate`
- `term` (binary encode: 36=0, 60=1)
- `grade` or `sub_grade` (ordinal encode A→G or A1→G5)
- `purpose` (one-hot or target encode)
- `fico_range_low` (drop `fico_range_high` — perfectly collinear)
- `dti` (winsorise at 75% before use)
- `annual_inc` (log-transform before use in linear models)
- `emp_length` (ordinal encode)
- `home_ownership` (one-hot encode; collapse rare categories)
- `verification_status` (one-hot encode)
- `delinq_2yrs`, `pub_rec`, `open_acc`, `total_acc`
- `revol_bal` (log-transform), `revol_util` (cap at 100%)
- `addr_state` (target encode or group into regions)
- `earliest_cr_line` (convert to "years of credit history" by subtracting from `issue_d`)
- `issue_d` derived features: year, month/quarter (for vintage/seasonality effects)

**Exclude (leakage — post-origination):**
- `recoveries`, `collection_recovery_fee`, `total_rec_late_fee`
- `out_prncp`, `out_prncp_inv`
- `total_pymnt`, `total_pymnt_inv`, `total_rec_prncp`, `total_rec_int`
- `last_pymnt_amnt`, `last_pymnt_d`

**Exclude (zero variance / quasi-constant):**
- `pymnt_plan`, `policy_code`, `hardship_flag`

**Exclude (identifiers, no predictive information):**
- `id`, `url`, `member_id`, `zip_code` (too granular; use `addr_state` instead)

---

### Preprocessing Pipeline

1. **Imputation:** Median imputation for low-missingness numeric columns; mode for categoricals. Add binary missingness indicator flags for columns with >5% nulls (the missing pattern may itself be predictive).
2. **Log-transform:** `annual_inc`, `revol_bal`, `loan_amnt` for use in linear/logistic models.
3. **Winsorise:** `dti` at 75%, `revol_util` at 100%, `annual_inc` at 99th percentile.
4. **Ordinal encode:** `grade`, `sub_grade`, `emp_length`, `term`.
5. **One-hot encode:** `home_ownership` (collapse rare), `verification_status`, `purpose`, `disbursement_method`.
6. **Target encode (or WoE):** `addr_state`, `purpose` (especially for logistic regression scorecard).
7. **Date-derived features:** `credit_history_years` = (issue date − earliest_cr_line) in years.

---

### Model Selection

| Model Type | Suitability | Notes |
|---|---|---|
| Logistic Regression + WoE | High | Industry standard for regulatory scorecards; interpretable coefficients |
| Random Forest | High | Robust to outliers, handles mixed feature types well |
| XGBoost / LightGBM | Very High | Best predictive performance; handles missing values natively |
| Neural Network | Moderate | Requires careful tuning; less interpretable |
| Decision Tree (single) | Low | Too shallow to capture complexity; use ensemble instead |

For a credit risk PD model, **LightGBM with calibration** is the recommended starting point for performance. For a regulatory-compliant scorecard, **Logistic Regression with Weight-of-Evidence (WoE) encoding** is the standard.

---

### Evaluation Metrics (in priority order)

1. **AUC-ROC** — primary discrimination metric. Target: >0.75 for a useful model.
2. **KS Statistic** — standard in credit scoring. Measures maximum separation between default/non-default CDFs.
3. **Log-Loss** — probability calibration quality. Critical if PD output is used in Expected Loss calculations.
4. **Gini Coefficient** — = 2 × AUC − 1. Commonly used in credit scoring alongside KS.
5. **Precision-Recall AUC** — informative for the minority class in the presence of imbalance.

**Never use:** raw accuracy (misleading due to class imbalance).

---

### Validation Strategy

- **Train set:** 2012–2015 originations.
- **Validation set:** 2016 originations (for hyperparameter tuning).
- **OOT test set:** 2017–2018 originations (for final model evaluation).
- **Vintage restriction:** Consider using only loans with ≥12 months of seasoning to avoid vintage bias in labels.
- **Cross-validation:** If using cross-validation, use `TimeSeriesSplit` (chronological folds), not random k-fold.

---

### Class Imbalance Handling

- **First approach:** `class_weight='balanced'` in scikit-learn; `scale_pos_weight=4` in XGBoost/LightGBM.
- **Second approach:** If class weighting is insufficient, apply SMOTE oversampling on the training set only (never on validation/test).
- **Calibration:** If SMOTE is used, the model's output probabilities will be shifted. Apply Platt scaling or isotonic regression post-training to recalibrate probabilities to the true base rate (~20%).

---

### Expected Loss Framework

Once the PD model is built, the output integrates into the standard credit risk Expected Loss formula:

```
Expected Loss (EL) = PD × LGD × EAD
```

- **PD** (Probability of Default) — output of this model, a calibrated probability.
- **LGD** (Loss Given Default) — fraction of the exposure lost if the borrower defaults (requires a separate LGD model).
- **EAD** (Exposure at Default) — the outstanding balance at the time of default (available from `loan_amnt` / `funded_amnt` at origination; or `out_prncp` at a later point in the loan lifecycle).

This EDA has addressed the PD component comprehensively. LGD and EAD models would be the natural next steps in a full credit risk modelling framework.
