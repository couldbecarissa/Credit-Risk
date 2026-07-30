# Build Progress: Credit Risk Deployable Demo

Tracking doc for turning this folder from tutorial material into a
deployable, testable, portfolio-ready project. Written for you to check
against, not for the employer-facing README.

## Status: done, verified end-to-end, not yet pushed to GitHub

## What existed before this session

- `credit_risk_model_guide.md`: a 1,989-line written tutorial with one
  big pipeline implementation embedded as a markdown code block (not a
  runnable file).
- `EDA_Report.md` + `lending_club_eda.ipynb`: real EDA findings (class
  imbalance, leakage columns, correlations, temporal drift).
- `accepted_2007_to_2018Q4.csv`: the real 1.67GB Lending Club dataset.
- No Benford's Law analysis anywhere.
- Git repo in a **detached HEAD** state, remote pointed at
  `github.com/couldbecarissa/Credit-Risk.git`, which does not actually
  exist yet on GitHub (confirmed via `gh`).

## What was built

- [x] Fixed git: checked out the existing local `main` branch (no commits
      lost), added `.gitignore` for the raw CSV, model artifacts, caches.
- [x] Generated `data/sample_loans.csv`: a real, stratified 50k-row sample
      pulled directly from the full CSV (preserves the ~20% default
      rate), small enough to commit.
- [x] Ported the guide's embedded pipeline code into real modules under
      `credit_risk/`: `data.py`, `features.py`, `model.py`, `evaluate.py`,
      `expected_loss.py`. Fixed one real bug surfaced during porting (a
      pandas categorical-dtype `fillna` crash in `apply_woe_transform`
      that the original markdown code would have hit too).
- [x] Built `credit_risk/benford.py` from scratch: first-digit Benford's
      Law test (chi-square + Nigrini MAD conformity), applied to
      `loan_amnt`, `annual_inc`, `revol_bal`. Real findings, not
      placeholders, see README's "Benford's Law findings" table.
- [x] `train.py` (CLI, saves a model artifact), `run_demo.py`
      (single-command demo on the sample), `service.py` (FastAPI,
      `/score` + `/health`, loads the artifact, does not retrain per
      request).
- [x] 14 pytest tests, each tied to a real worked example (the guide's
      own DTI WOE/IV table, its PD-to-score table) or a constructed
      Benford conforming/non-conforming case, plus FastAPI `TestClient`
      integration tests against a real trained artifact. All passing.
- [x] `Dockerfile` (trains on the sample at build time, serves
      immediately on `docker run`), pinned `requirements.txt`.
- [x] `README.md`, employer-facing, includes an honest "Known
      limitations" section (see below, this matters).
- [x] Ran everything end-to-end: `run_demo.py`, `pytest` (14/14 passing),
      the FastAPI service via curl (both a low-risk and a deliberately
      high-risk synthetic application, confirmed the ranking direction is
      correct).
- [ ] **Docker not verified.** Docker isn't installed in this environment,
      so `docker build`/`docker run` could not actually be tested. The
      Dockerfile was written carefully and reviewed by hand (copy order,
      paths, that `data/sample_loans.csv` is present before `RUN python
      train.py`), but you should run the two commands below yourself
      before relying on it for an application:
      ```
      docker build -t credit-risk .
      docker run -p 8000:8000 credit-risk
      curl http://localhost:8000/health
      ```

## Honest limitations, carried into the README, not hidden

1. **PD is not probability-calibrated.** `class_weight='balanced'` was
   used to handle the class imbalance, which is why the two example
   `/score` calls came back at 57% and 88% PD rather than something
   closer to the ~20% true base rate. Ranking (AUC/KS/Gini) is real and
   good; the raw probability number is not calibrated. The guide itself
   names the fix (Platt scaling / isotonic regression) as a next step,
   not yet implemented.
2. **PSI is train-vs-test on a random split**, not a genuine out-of-time
   check. The guide recommends training on 2012-2015 and testing on
   2017-2018 using `issue_d`; that would be the natural next iteration if
   you want to show real temporal validation.
3. No hyperparameter tuning was done beyond the guide's defaults.

If an interviewer asks about any of these, the honest answer is "here's
what I found, here's why, and here's the documented next step", the same
approach used throughout your other application answers this session.

## What's still pending, needs you

- **GitHub repo doesn't exist yet.** `git remote -v` in this folder
  points at `github.com/couldbecarissa/Credit-Risk`, but that repository
  does not exist on your account. Before pushing, I need to create it
  (via `gh repo create`) and confirm visibility, public or private, since
  the whole point of this project is to be shareable with employers, I'd
  default to public unless you say otherwise.
- `.claude/settings.local.json` has an uncommitted local permissions
  change and is now gitignored, not pushed, that's a local tooling file,
  not project content.
- `pairs_halflife.py`/`.png` (the pairs-trading side script) is untouched
  and untracked, left alone since it's unrelated to this project.
