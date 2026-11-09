import pytest
from fastapi.testclient import TestClient

from api.main import create_app


@pytest.fixture()
def client(tiny_predictor):
    with TestClient(create_app(tiny_predictor)) as c:
        yield c


def test_health(client):
    r = client.get("/health")
    assert r.status_code == 200
    body = r.json()
    assert body["status"] == "ok" and body["model_version"] == "test-fixture"


def test_predict_returns_label_and_probabilities(client):
    r = client.post("/predict", json={"text": "A debt collector keeps calling me about a debt I do not owe."})
    assert r.status_code == 200
    body = r.json()
    assert body["label"] == "debt_collection"
    assert abs(sum(body["probabilities"].values()) - 1) < 1e-3
    assert body["label"] == max(body["probabilities"], key=body["probabilities"].get)
    assert body["model_version"] == "test-fixture"


def test_predict_handles_redacted_text(client):
    r = client.post(
        "/predict",
        json={"text": "XXXX reported a wrong late payment on my credit report to Equifax"},
    )
    assert r.status_code == 200
    assert r.json()["label"] == "credit_reporting"


@pytest.mark.parametrize("payload", [{}, {"text": ""}, {"text": "hi"}, {"text": 5}])
def test_predict_rejects_bad_input(client, payload):
    assert client.post("/predict", json=payload).status_code == 422
