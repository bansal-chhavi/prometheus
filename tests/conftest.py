"""
Shared test fixtures and configuration.
"""
import json
import pytest
from pathlib import Path
from httpx import AsyncClient
from src.main import app


@pytest.fixture
def sample_faithful_payload():
    """Load and return the faithful sample evaluation case."""
    inputs_dir = Path(__file__).parent.parent / "Inputs"
    with open(inputs_dir / "prometheus_eval_01.json", "r") as f:
        data = json.load(f)
    
    return {
        "query": data["question"],
        "answer": data["ai_answer"],
        "documents": [data["document_text"]]
    }


@pytest.fixture
def sample_hallucination_payload():
    """Load and return a hallucination sample evaluation case."""
    inputs_dir = Path(__file__).parent.parent / "Inputs"
    try:
        with open(inputs_dir / "prometheus_eval_05.json", "r") as f:
            data = json.load(f)
        
        return {
            "query": data["question"],
            "answer": data["ai_answer"],
            "documents": [data["document_text"]]
        }
    except FileNotFoundError:
        # Fallback sample if file not found
        return {
            "query": "What was the forecasting accuracy improvement?",
            "answer": "The forecasting accuracy improved by 31%.",
            "documents": ["The forecasting accuracy improved by 24% according to the study."]
        }


@pytest.fixture
async def http_client():
    """Provide an async HTTP client for testing the API."""
    async with AsyncClient(app=app, base_url="http://test") as client:
        yield client


@pytest.fixture
def mock_llm_response():
    """Mock responses for LLM calls in testing."""
    return {
        "claims": ["The answer contains basic factual claims"],
        "support_check": '{"is_supported": true, "evidence_excerpt": "matching text"}'
    }
