"""FastAPI scoring service. Loads a persisted model artifact at startup
(produced by train.py) and serves predictions; it does not retrain per
request.

Run:
    uvicorn service:app --reload

Then:
    curl -X POST http://127.0.0.1:8000/score -H "Content-Type: application/json" -d '{
        "loan_amnt": 15000, "int_rate": 13.5, "annual_inc": 62000, "dti": 22.4,
        "fico_range_low": 690, "revol_util": 48.0, "revol_bal": 9000,
        "open_acc": 8, "delinq_2yrs": 0, "inq_last_6mths": 1, "pub_rec": 0,
        "emp_length": 5, "term": 36, "home_ownership": "MORTGAGE",
        "purpose": "debt_consolidation"
    }'
"""
import pickle
from contextlib import asynccontextmanager
from pathlib import Path

from fastapi import FastAPI, HTTPException
from pydantic import BaseModel, Field

from credit_risk.predict import score_application

MODEL_PATH = Path("models/model.pkl")

_artifact = None


def _load_artifact():
    global _artifact
    if MODEL_PATH.exists():
        with open(MODEL_PATH, "rb") as f:
            _artifact = pickle.load(f)


@asynccontextmanager
async def lifespan(app: FastAPI):
    _load_artifact()
    yield


app = FastAPI(
    title="Credit Risk Scoring API",
    description="Probability-of-default scoring for loan applications.",
    lifespan=lifespan,
)


class LoanApplication(BaseModel):
    loan_amnt: float = Field(..., gt=0, description="Requested loan amount (USD)")
    int_rate: float = Field(..., description="Interest rate (%)")
    annual_inc: float = Field(..., ge=0, description="Annual income (USD)")
    dti: float = Field(..., description="Debt-to-income ratio (%)")
    fico_range_low: float = Field(..., description="FICO score (lower bound)")
    revol_util: float = Field(..., description="Revolving credit utilisation (%)")
    revol_bal: float = Field(..., ge=0, description="Total revolving balance (USD)")
    open_acc: int = Field(..., ge=0, description="Number of open credit accounts")
    delinq_2yrs: int = Field(..., ge=0, description="30+ DPD delinquencies in past 2 years")
    inq_last_6mths: int = Field(..., ge=0, description="Credit inquiries in last 6 months")
    pub_rec: int = Field(..., ge=0, description="Number of public records (bankruptcies)")
    emp_length: float = Field(..., ge=0, le=10, description="Employment length in years (0-10)")
    term: float = Field(..., description="Loan term in months (36 or 60)")
    home_ownership: str = Field(..., description="RENT / OWN / MORTGAGE / OTHER")
    purpose: str = Field(..., description="Loan purpose, e.g. debt_consolidation")


@app.get("/health")
def health():
    return {
        "status": "healthy" if _artifact is not None else "unhealthy",
        "model_loaded": _artifact is not None,
    }


@app.post("/score")
def score(application: LoanApplication):
    if _artifact is None:
        raise HTTPException(
            status_code=503,
            detail="Model artifact not found. Run `python train.py` first to produce models/model.pkl.",
        )
    return score_application(application.model_dump(), _artifact)
