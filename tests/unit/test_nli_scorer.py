import pytest
from unittest.mock import AsyncMock, patch
from prometheus.detectors.nli_scorer import score_claims, on_nli_notification
from prometheus.models import NLILabel

@pytest.mark.asyncio
async def test_score_claims_hf_entailment(sample_claims, sample_sources, mock_llm):
    mock_hf = AsyncMock()
    mock_hf.classify.return_value = {"label": "entailment", "scores": {"entailment": 0.9}}
    verdicts, backend = await score_claims(sample_claims[:1], sample_sources, mock_llm, mock_hf)
    assert len(verdicts) == 1
    assert verdicts[0].label == NLILabel.ENTAILMENT
    assert verdicts[0].explanation.startswith("DeBERTa NLI:")
    assert backend == "huggingface"

@pytest.mark.asyncio
async def test_score_claims_hf_fallback(sample_claims, sample_sources, mock_llm):
    mock_hf = AsyncMock()
    mock_hf.classify.return_value = None
    mock_llm.complete_json.return_value = {"label": "contradiction", "confidence": 0.8}
    
    notified = []
    def callback(msg):
        notified.append(msg)
    on_nli_notification(callback)
    
    verdicts, backend = await score_claims(sample_claims[:1], sample_sources, mock_llm, mock_hf)
    assert backend == "huggingface+groq"
    assert len(notified) == 1
    assert "rate limit exceeded" in notified[0].lower()

@pytest.mark.asyncio
async def test_score_claims_no_hf(sample_claims, sample_sources, mock_llm):
    mock_llm.complete_json.return_value = {"label": "entailment", "confidence": 0.9}
    verdicts, backend = await score_claims(sample_claims[:1], sample_sources, mock_llm, None)
    assert backend == "groq"
    assert verdicts[0].label == NLILabel.ENTAILMENT
