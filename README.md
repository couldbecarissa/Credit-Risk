# Credit Risk Scoring: Probability of Default + Benford's Law Screen

A deployable credit risk model: a WOE/logistic-regression scorecard for
estimating Probability of Default (PD) on Lending Club loan applications,
paired with a Benford's Law data-integrity screen on reported financial
fields. Includes a FastAPI scoring service, a CLI training pipeline, and
an automated test suite.

The full theory (WOE/IV, Basel framework, scorecard scaling, ROC/KS/Gini,
PSI/OOT validation) is written up in [`credit_risk_model_guide.md`](credit_risk_model_guide.md);
the underlying dataset exploration is in [`EDA_Report.md`](EDA_Report.md)
and [`lending_club_eda.ipynb`](lending_club_eda.ipynb). This README covers
the runnable pipeline built from that theory.

## Quick Start

```bash
pip install -r requirements.txt

# Full end-to-end demo: trains on the bundled 50k-row sample, runs the
# Benford's Law screen, evaluates the model, saves plots + a run report.
python run_demo.py

# Serve predictions (loads the artifact train.py just produced; does not
# retrain per request):
uvicorn service:app --reload
```

```bash
curl -X POST http://127.0.0.1:8000/score -H "Content-Type: application/json" -d '{
    "loan_amnt": 15000, "int_rate": 13.5, "annual_inc": 62000, "dti": 22.4,
    "fico_range_low": 690, "revol_util": 48.0, "revol_bal": 9000,
    "open_acc": 8, "delinq_2yrs": 0, "inq_last_6mths": 1, "pub_rec": 0,
    "emp_length": 5, "term": 36, "home_ownership": "MORTGAGE",
    "purpose": "debt_consolidation"
}'
# -> {"probability_of_default": 0.57, "credit_score": 478.6, "risk_tier": "Very high risk"}
```

### Docker

```bash
docker build -t credit-risk .
docker run -p 8000:8000 credit-risk
```

The image trains on the bundled sample at build time, so the container
serves predictions immediately on `docker run`.

### Full dataset (optional)

The bundled `data/sample_loans.csv` is a stratified 50k-row random sample
of the full [Lending Club dataset on Kaggle](https://www.kaggle.com/datasets/wordsforthewise/lending-club)
(preserves the real ~20% default rate). The full CSV is 1.67GB and is not
committed to this repo. To train on it instead:

```bash
python train.py --data path/to/accepted_2007_to_2018Q4.csv
```

## Architecture

```
credit_risk/        # library: data cleaning, WOE/IV, model, evaluation,
                     # Benford's Law screen, expected loss, scoring
train.py             # CLI: trains the model, saves models/model.pkl
run_demo.py           # single-command demo on the bundled sample
service.py             # FastAPI app: POST /score, GET /health
tests/                 # pytest suite, see "Testing" below
```

Training and serving are separated: `train.py` produces a persisted
artifact (`models/model.pkl`, gitignored) containing the fitted model and
WOE maps; `service.py` loads that artifact at startup and never retrains
per request.

## What the pipeline actually does

1. **Load & clean** (`credit_risk/data.py`): filters to resolved loans
   (Fully Paid / Charged Off), cleans `emp_length`/`term`/`int_rate`
   formatting, caps extreme outliers at the 99.9th percentile, imputes
   missing values.
2. **Benford's Law screen** (`credit_risk/benford.py`, new): a first-digit
   conformity test (chi-square + Nigrini MAD scoring) on `loan_amnt`,
   `annual_inc`, and `revol_bal`, as a data-integrity check alongside the
   outlier findings already documented in `EDA_Report.md`.
3. **IV feature screening + WOE transform** (`credit_risk/features.py`):
   ranks features by Information Value, keeps IV >= 0.02, bins continuous
   features and encodes everything as Weight of Evidence.
4. **VIF check**: drops any WOE feature with VIF > 10.
5. **Train** (`credit_risk/model.py`): class-weighted logistic regression.
6. **Evaluate** (`credit_risk/evaluate.py`): AUC, KS, Gini, ROC/KS plots,
   Population Stability Index.
7. **Scorecard scaling**: converts log-odds to a 600-base, 20-PDO score.
8. **Expected Loss** (`credit_risk/expected_loss.py`): EL = PD x LGD x EAD.

## Real results (bundled sample, 50,000 loans, 19.9% default rate)

| Metric | Value |
|---|---|
| AUC | 0.737 |
| KS | 0.359 |
| Gini | 0.475 |
| Accuracy | 0.654 |
| PSI (train vs. test) | 0.0006 |

### Benford's Law findings

| Feature | MAD | Conformity |
|---|---|---|
| `loan_amnt` | 0.033 | Nonconformity |
| `annual_inc` | 0.053 | Nonconformity |
| `revol_bal` | 0.011 | Acceptable conformity |

`loan_amnt` and `annual_inc` fail the first-digit test, but this reads as
expected round-number bias rather than fraud: borrowers request loans in
round increments ($5k, $10k, $15k...), a pattern already visible in the
EDA's histogram of `loan_amnt`, and self-reported income tends to be
rounded too. `revol_bal` accumulates organically across many transactions
and conforms far more closely, which is the expected contrast.

## Known limitations (stated honestly, not glossed over)

- **PD is not probability-calibrated.** The model uses
  `class_weight='balanced'` to handle the ~4:1 class imbalance, which
  improves ranking (AUC/KS/Gini) but shifts predicted probabilities away
  from the true ~20% base rate, visible in the Expected-Loss-rate figure
  above being higher than PD x LGD would suggest at the true base rate.
  `credit_risk_model_guide.md` documents the fix (Platt scaling or
  isotonic regression); it is not yet implemented here.
- **PSI is train-vs-test on a random split**, not a real out-of-time
  check, so the near-zero PSI above just confirms the split is balanced,
  not that the model is stable over time. The guide's recommended
  approach (train on 2012-2015, validate 2016, test 2017-2018) needs
  `issue_d`-based splitting, which the demo sample doesn't preserve.
- No hyperparameter tuning beyond the guide's defaults.

## Testing

```bash
pytest
```

14 tests, each tied to something concrete: the WOE/IV worked example from
`credit_risk_model_guide.md` Section 6, the PD-to-score worked table from
Section 7, constructed conforming/non-conforming Benford's Law cases, and
FastAPI `TestClient` integration tests against a real trained artifact
(not mocks).
