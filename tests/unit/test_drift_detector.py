import pytest
from prometheus.detectors.drift_detector import detect_drift

@pytest.mark.asyncio
async def test_drift_detector_09(mock_llm):
    mock_llm.complete_json.return_value = {"relevance_score": 0.9}
    drift, _ = await detect_drift(mock_llm, "Q", "A")
    assert drift == 0.1

@pytest.mark.asyncio
async def test_drift_detector_01(mock_llm):
    mock_llm.complete_json.return_value = {"relevance_score": 0.1}
    drift, _ = await detect_drift(mock_llm, "Q", "A")
    assert drift == 0.9

@pytest.mark.asyncio
async def test_drift_detector_exception(mock_llm):
    mock_llm.complete_json.side_effect = Exception("error")
    drift, _ = await detect_drift(mock_llm, "Q", "A")
    assert drift == 0.5
