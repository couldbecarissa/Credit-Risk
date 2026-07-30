"""FastAPI TestClient smoke tests for /health and /score. Trains a real
(fast, sample-based) model artifact via train.py first, so these are true
integration tests against the actual persisted artifact, not mocks.
"""
import pytest
from fastapi.testclient import TestClient

import train

VALID_APPLICATION = {
    "loan_amnt": 15000, "int_rate": 13.5, "annual_inc": 62000, "dti": 22.4,
    "fico_range_low": 690, "revol_util": 48.0, "revol_bal": 9000,
    "open_acc": 8, "delinq_2yrs": 0, "inq_last_6mths": 1, "pub_rec": 0,
    "emp_length": 5, "term": 36, "home_ownership": "MORTGAGE",
    "purpose": "debt_consolidation",
}


@pytest.fixture(scope="module", autouse=True)
def trained_artifact():
    """Ensure models/model.pkl exists before the service starts."""
    train.main(data_path="data/sample_loans.csv")


@pytest.fixture(scope="module")
def client():
    import service
    service._load_artifact()
    return TestClient(service.app)


def test_health_reports_model_loaded(client):
    response = client.get("/health")
    assert response.status_code == 200
    body = response.json()
    assert body["status"] == "healthy"
    assert body["model_loaded"] is True


def test_score_returns_pd_score_and_tier(client):
    response = client.post("/score", json=VALID_APPLICATION)
    assert response.status_code == 200
    body = response.json()
    assert 0.0 <= body["probability_of_default"] <= 1.0
    assert isinstance(body["credit_score"], float)
    assert body["risk_tier"] in ("Low risk", "Medium risk", "High risk", "Very high risk")


def test_score_rejects_invalid_payload(client):
    bad_application = dict(VALID_APPLICATION)
    bad_application["loan_amnt"] = -500  # violates gt=0
    response = client.post("/score", json=bad_application)
    assert response.status_code == 422
