import requests
from src.nli_client import NLIClient


def test_hf_predict_success(monkeypatch):
    client = NLIClient()

    class FakeResp:
        status_code = 200

        def raise_for_status(self):
            return None

        def json(self):
            return {"label": "entailment", "score": 0.95}

    def fake_post(url, headers=None, json=None, timeout=None):
        return FakeResp()

    monkeypatch.setattr("requests.post", fake_post)

    label, score = client._hf_predict("Test claim", "Test document")
    assert label == "ENTAILMENT"
    assert score == 0.95


def test_predict_uses_groq_on_hf_rate_limit(monkeypatch):
    client = NLIClient()

    def fake_hf_raise(claim, document):
        e = requests.HTTPError("rate limited")
        e.response = type("R", (), {"status_code": 429})
        raise e

    def fake_groq(claim, document):
        return ("ENTAILMENT", 0.7)

    monkeypatch.setattr(client, "_hf_predict", fake_hf_raise)
    monkeypatch.setattr(client, "_groq_predict", fake_groq)

    label, score = client.predict("claim", "doc")
    assert label == "ENTAILMENT"
    assert score == 0.7
