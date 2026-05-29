"""
Scorer module: Compute faithfulness score and classification label.
"""


class Scorer:
    """Handles faithfulness scoring and label classification."""
    
    @staticmethod
    def compute_faithfulness_score(claims: list[str], support_map: dict[str, bool]) -> float:
        """
        Compute faithfulness score as the ratio of supported claims to total claims.
        
        Args:
            claims: List of extracted claims
            support_map: Dict mapping claim -> is_supported (bool)
            
        Returns:
            Float score between 0.0 and 1.0
        """
        if not claims:
            return 1.0  # Empty answer is technically faithful
        
        if not support_map:
            return 0.0  # No evidence to support any claims
        
        # Count supported claims
        supported = sum(1 for claim in claims if support_map.get(claim, False))
        
        # Calculate ratio
        score = supported / len(claims)
        
        return round(score, 2)
    
    @staticmethod
    def compute_label(score: float, threshold: float = 0.8) -> str:
        """
        Determine if answer is FAITHFUL or HALLUCINATION based on score.
        
        Args:
            score: Faithfulness score (0.0-1.0)
            threshold: Score threshold for FAITHFUL classification (default: 0.8)
            
        Returns:
            Label string: "FAITHFUL" or "HALLUCINATION"
        """
        return "FAITHFUL" if score >= threshold else "HALLUCINATION"
