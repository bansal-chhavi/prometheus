import pytest
from prometheus.detectors.claim_extractor import extract_claims

@pytest.mark.asyncio
async def test_extract_claims_success(mock_llm):
    mock_llm.complete_json.return_value = {
        "claims": [{"text": "C1", "source_span": "S1"}]
    }
    claims = await extract_claims(mock_llm, "Q", "Answer with S1 in it")
    assert len(claims) == 1
    assert claims[0].text == "C1"
    assert claims[0].start == 12
    assert claims[0].end == 14

@pytest.mark.asyncio
async def test_extract_claims_empty(mock_llm):
    mock_llm.complete_json.return_value = {"claims": []}
    claims = await extract_claims(mock_llm, "Q", "A")
    assert len(claims) == 0

@pytest.mark.asyncio
async def test_extract_claims_malformed(mock_llm):
    mock_llm.complete_json.return_value = {}
    claims = await extract_claims(mock_llm, "Q", "A")
    assert len(claims) == 0
