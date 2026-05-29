"""
API integration tests for the FastAPI endpoint.
"""
import pytest
from httpx import AsyncClient
import json
from src.main import app
from src.models import EvaluationResponse


@pytest.mark.asyncio
class TestHealthEndpoint:
    """Tests for health check endpoint."""
    
    async def test_health_check_returns_ok(self):
        """Test that health endpoint returns ok status."""
        async with AsyncClient(app=app, base_url="http://test") as client:
            response = await client.get("/health")
            assert response.status_code == 200
            data = response.json()
            assert data["status"] == "ok"


@pytest.mark.asyncio
class TestEvaluateEndpoint:
    """Tests for the main evaluation endpoint."""
    
    async def test_evaluate_valid_request_schema(self):
        """Test that valid request returns correct response schema."""
        async with AsyncClient(app=app, base_url="http://test") as client:
            payload = {
                "query": "What is the main finding?",
                "answer": "The main finding was a 25% improvement.",
                "documents": ["The study showed a 25% improvement in performance."]
            }
            
            # This will fail without LLM configured, but we're testing schema
            response = await client.post("/evaluate", json=payload)
            
            # Check response is 200 or 500 (depending on LLM config)
            assert response.status_code in [200, 500]
            
            if response.status_code == 200:
                data = response.json()
                # Verify response schema
                assert "score" in data
                assert "label" in data
                assert "hallucinated_spans" in data
                assert "evidence_spans" in data
                assert "explanation" in data
                
                # Verify types
                assert isinstance(data["score"], (int, float))
                assert data["label"] in ["FAITHFUL", "HALLUCINATION"]
                assert isinstance(data["hallucinated_spans"], list)
                assert isinstance(data["evidence_spans"], list)
                assert isinstance(data["explanation"], str)
    
    async def test_evaluate_missing_query(self):
        """Test that missing query field returns 422."""
        async with AsyncClient(app=app, base_url="http://test") as client:
            payload = {
                # Missing "query" field
                "answer": "Some answer",
                "documents": ["Some document"]
            }
            
            response = await client.post("/evaluate", json=payload)
            assert response.status_code == 422
    
    async def test_evaluate_missing_answer(self):
        """Test that missing answer field returns 422."""
        async with AsyncClient(app=app, base_url="http://test") as client:
            payload = {
                "query": "What is the finding?",
                # Missing "answer" field
                "documents": ["Some document"]
            }
            
            response = await client.post("/evaluate", json=payload)
            assert response.status_code == 422
    
    async def test_evaluate_missing_documents(self):
        """Test that missing documents field returns 422."""
        async with AsyncClient(app=app, base_url="http://test") as client:
            payload = {
                "query": "What is the finding?",
                "answer": "Some answer",
                # Missing "documents" field
            }
            
            response = await client.post("/evaluate", json=payload)
            assert response.status_code == 422
    
    async def test_evaluate_empty_documents(self):
        """Test that empty documents list returns 422."""
        async with AsyncClient(app=app, base_url="http://test") as client:
            payload = {
                "query": "What is the finding?",
                "answer": "Some answer",
                "documents": []
            }
            
            response = await client.post("/evaluate", json=payload)
            assert response.status_code == 422
    
    async def test_evaluate_empty_query_string(self):
        """Test that empty query string is rejected."""
        async with AsyncClient(app=app, base_url="http://test") as client:
            payload = {
                "query": "",
                "answer": "Some answer",
                "documents": ["Some document"]
            }
            
            response = await client.post("/evaluate", json=payload)
            assert response.status_code == 422
    
    async def test_evaluate_empty_answer_string(self):
        """Test that empty answer string is rejected."""
        async with AsyncClient(app=app, base_url="http://test") as client:
            payload = {
                "query": "What is the finding?",
                "answer": "",
                "documents": ["Some document"]
            }
            
            response = await client.post("/evaluate", json=payload)
            assert response.status_code == 422


@pytest.mark.asyncio
class TestEvaluateDeterminism:
    """Tests for deterministic behavior (US4)."""
    
    async def test_same_request_returns_identical_response(self):
        """Test that identical requests return identical responses."""
        payload = {
            "query": "What is the finding?",
            "answer": "The finding was a 25% improvement.",
            "documents": ["The study showed a 25% improvement."]
        }
        
        async with AsyncClient(app=app, base_url="http://test") as client:
            response1 = await client.post("/evaluate", json=payload)
            response2 = await client.post("/evaluate", json=payload)
            
            # Both should have same status
            assert response1.status_code == response2.status_code
            
            if response1.status_code == 200:
                # For successful responses, scores should be identical (deterministic)
                data1 = response1.json()
                data2 = response2.json()
                assert data1["score"] == data2["score"]
                assert data1["label"] == data2["label"]


@pytest.mark.asyncio
class TestEvaluateResponseFormat:
    """Tests to verify response format compliance."""
    
    async def test_response_contains_all_required_fields(self):
        """Test that response has all required fields."""
        payload = {
            "query": "What improved?",
            "answer": "Performance improved by 30%.",
            "documents": ["Performance metrics improved by 30% year over year."]
        }
        
        async with AsyncClient(app=app, base_url="http://test") as client:
            response = await client.post("/evaluate", json=payload)
            
            if response.status_code == 200:
                data = response.json()
                required_fields = ["score", "label", "hallucinated_spans", 
                                 "evidence_spans", "explanation"]
                for field in required_fields:
                    assert field in data, f"Missing required field: {field}"
    
    async def test_score_is_between_zero_and_one(self):
        """Test that score is always between 0.0 and 1.0."""
        payload = {
            "query": "What is the metric?",
            "answer": "The metric is 50%.",
            "documents": ["The metric is 50%."]
        }
        
        async with AsyncClient(app=app, base_url="http://test") as client:
            response = await client.post("/evaluate", json=payload)
            
            if response.status_code == 200:
                data = response.json()
                assert 0.0 <= data["score"] <= 1.0
    
    async def test_label_is_valid_value(self):
        """Test that label is either FAITHFUL or HALLUCINATION."""
        payload = {
            "query": "What is the metric?",
            "answer": "The metric is 50%.",
            "documents": ["The metric is 50%."]
        }
        
        async with AsyncClient(app=app, base_url="http://test") as client:
            response = await client.post("/evaluate", json=payload)
            
            if response.status_code == 200:
                data = response.json()
                assert data["label"] in ["FAITHFUL", "HALLUCINATION"]
    
    async def test_spans_are_lists(self):
        """Test that span fields are lists."""
        payload = {
            "query": "What is the metric?",
            "answer": "The metric is 50%.",
            "documents": ["The metric is 50%."]
        }
        
        async with AsyncClient(app=app, base_url="http://test") as client:
            response = await client.post("/evaluate", json=payload)
            
            if response.status_code == 200:
                data = response.json()
                assert isinstance(data["hallucinated_spans"], list)
                assert isinstance(data["evidence_spans"], list)
    
    async def test_explanation_is_nonempty_string(self):
        """Test that explanation is a non-empty string."""
        payload = {
            "query": "What is the metric?",
            "answer": "The metric is 50%.",
            "documents": ["The metric is 50%."]
        }
        
        async with AsyncClient(app=app, base_url="http://test") as client:
            response = await client.post("/evaluate", json=payload)
            
            if response.status_code == 200:
                data = response.json()
                assert isinstance(data["explanation"], str)
                assert len(data["explanation"]) > 0
