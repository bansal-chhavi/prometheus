import pytest
from fastapi.testclient import TestClient
from server.api import app

client = TestClient(app)

def test_health():
    response = client.get("/health")
    assert response.status_code == 200
    assert response.json()["status"] == "ok"

def test_evaluate_missing_field():
    response = client.post("/evaluate", json={"query": "Q"})
    assert response.status_code == 422
