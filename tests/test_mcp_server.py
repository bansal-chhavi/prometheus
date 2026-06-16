import json

from src.trustsignal.mcp_server import handle_mcp_request


def test_handle_mcp_request_basic():
    payload = {
        "query": "Who won 2020 election?",
        "rag_answer": "Candidate A won the 2020 election.",
        "sources": [{"id": "s1", "text": "Candidate A won the 2020 election."}],
    }

    res = handle_mcp_request(payload)

    # Basic assertions about the EvalResult shape
    assert isinstance(res, dict)
    assert "score" in res
    assert "flags" in res
    assert "hallu_spans" in res
    assert "explanation" in res

    # score should be a number between 0 and 1
    assert 0.0 <= float(res["score"]) <= 1.0
