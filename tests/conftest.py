import pytest
from unittest.mock import AsyncMock
from prometheus import PrometheusConfig
from prometheus.llm_providers.base import BaseLLMProvider
from prometheus.models import Claim, NLILabel


@pytest.fixture
def mock_llm():
    """Mock LLM provider that returns configurable JSON responses."""
    provider = AsyncMock(spec=BaseLLMProvider)
    return provider


@pytest.fixture
def sample_claims():
    return [
        Claim(text="BCG launched Deckster", source_span="BCG launched Deckster", start=0, end=20),
        Claim(text="launched in 2021", source_span="in 2021", start=28, end=35),
        Claim(text="reducing time by 50%", source_span="50%", start=53, end=56),
    ]


@pytest.fixture
def sample_sources():
    return ["BCG launched Deckster in 2022, cutting deck creation time by 30% across 200 pilot engagements."]


@pytest.fixture
def config():
    return PrometheusConfig(llm_api_key="test-key")
