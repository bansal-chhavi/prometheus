"""
Unit tests for the scorer module.
"""
import pytest
from src.scorer import Scorer


class TestScorer:
    """Tests for faithfulness scoring and classification."""
    
    scorer = Scorer()
    
    def test_compute_faithfulness_all_supported(self):
        """Test score when all claims are supported."""
        claims = ["Claim A", "Claim B", "Claim C"]
        support_map = {"Claim A": True, "Claim B": True, "Claim C": True}
        score = self.scorer.compute_faithfulness_score(claims, support_map)
        assert score == 1.0
    
    def test_compute_faithfulness_no_supported(self):
        """Test score when no claims are supported."""
        claims = ["Claim A", "Claim B"]
        support_map = {"Claim A": False, "Claim B": False}
        score = self.scorer.compute_faithfulness_score(claims, support_map)
        assert score == 0.0
    
    def test_compute_faithfulness_partially_supported(self):
        """Test score with partial support."""
        claims = ["Claim A", "Claim B", "Claim C", "Claim D"]
        support_map = {"Claim A": True, "Claim B": True, "Claim C": False, "Claim D": False}
        score = self.scorer.compute_faithfulness_score(claims, support_map)
        assert score == 0.5
    
    def test_compute_faithfulness_empty_claims(self):
        """Test empty claims list."""
        claims = []
        support_map = {}
        score = self.scorer.compute_faithfulness_score(claims, support_map)
        assert score == 1.0  # No false claims, technically faithful
    
    def test_compute_faithfulness_empty_support_map(self):
        """Test with empty support map."""
        claims = ["Claim A", "Claim B"]
        support_map = {}
        score = self.scorer.compute_faithfulness_score(claims, support_map)
        assert score == 0.0  # No support for any claims
    
    def test_compute_label_faithful(self):
        """Test label when score exceeds threshold."""
        label = self.scorer.compute_label(0.9, threshold=0.8)
        assert label == "FAITHFUL"
    
    def test_compute_label_at_threshold(self):
        """Test label exactly at threshold."""
        label = self.scorer.compute_label(0.8, threshold=0.8)
        assert label == "FAITHFUL"
    
    def test_compute_label_hallucination(self):
        """Test label when score below threshold."""
        label = self.scorer.compute_label(0.7, threshold=0.8)
        assert label == "HALLUCINATION"
    
    def test_compute_label_custom_threshold(self):
        """Test with custom threshold."""
        label1 = self.scorer.compute_label(0.6, threshold=0.5)
        assert label1 == "FAITHFUL"
        
        label2 = self.scorer.compute_label(0.4, threshold=0.5)
        assert label2 == "HALLUCINATION"
    
    def test_compute_label_edge_cases(self):
        """Test edge case scores."""
        assert self.scorer.compute_label(0.0) == "HALLUCINATION"
        assert self.scorer.compute_label(1.0) == "FAITHFUL"
        assert self.scorer.compute_label(0.79, threshold=0.8) == "HALLUCINATION"
        assert self.scorer.compute_label(0.80, threshold=0.8) == "FAITHFUL"
