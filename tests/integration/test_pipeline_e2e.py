import pytest
from unittest.mock import patch, AsyncMock
from prometheus.pipeline import PrometheusPipeline
from prometheus.config import PrometheusConfig
from prometheus.models import Flag, Claim

@pytest.mark.asyncio
async def test_pipeline_no_claims():
    config = PrometheusConfig(llm_api_key="test")
    pipeline = PrometheusPipeline(config)
    
    with patch("prometheus.pipeline.extract_claims", new_callable=AsyncMock) as mock_extract:
        mock_extract.return_value = []
        res = await pipeline.evaluate("Q", "A", ["S"])
        assert res.score == 1.0
        assert Flag.FAITHFUL in res.flags

@pytest.mark.asyncio
async def test_pipeline_faithful():
    config = PrometheusConfig(llm_api_key="test")
    pipeline = PrometheusPipeline(config)
    
    with patch("prometheus.pipeline.extract_claims", new_callable=AsyncMock) as m_extract, \
         patch("prometheus.pipeline.score_claims", new_callable=AsyncMock) as m_score, \
         patch("prometheus.pipeline.detect_drift", new_callable=AsyncMock) as m_drift:
        
        m_extract.return_value = [Claim(text="c", source_span="c", start=0, end=1)]
        m_score.return_value = ([], "mock")
        m_drift.return_value = (0.0, "ok")
        
        res = await pipeline.evaluate("Q", "A", ["S"])
        assert Flag.FAITHFUL in res.flags
