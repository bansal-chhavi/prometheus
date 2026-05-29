"""
Explanation generator: Produce human-readable evaluation reasoning.
"""
from src.config import get_llm_config


class Explainer:
    """Generates natural language explanations for evaluation results."""
    
    def __init__(self):
        """Initialize explainer with LLM configuration."""
        self.llm_config = get_llm_config()
        self.client = self._init_llm_client()
    
    def _init_llm_client(self):
        """Initialize appropriate LLM client based on configuration."""
        provider = self.llm_config.get("provider", "openai").lower()
        
        if provider == "anthropic":
            try:
                import anthropic
                return anthropic.Anthropic(api_key=self.llm_config.get("api_key"))
            except ImportError:
                raise ImportError("Anthropic SDK not installed.")
        else:
            try:
                import openai
                openai.api_key = self.llm_config.get("api_key")
                return openai.OpenAI(api_key=self.llm_config.get("api_key"))
            except ImportError:
                raise ImportError("OpenAI SDK not installed.")
    
    def generate_explanation(self, query: str, answer: str, claims: list[str],
                            support_map: dict[str, bool], 
                            hallucinated_spans: list[str]) -> str:
        """
        Generate a concise natural language explanation of the evaluation result.
        
        Args:
            query: The original user query
            answer: The RAG-generated answer
            claims: List of extracted claims
            support_map: Dict mapping claim -> is_supported
            hallucinated_spans: List of detected hallucinated spans
            
        Returns:
            Natural language explanation string
        """
        if not claims:
            return "Unable to extract claims from answer for evaluation."
        
        supported_count = sum(1 for is_supported in support_map.values() if is_supported)
        unsupported_count = len(claims) - supported_count
        
        if unsupported_count == 0:
            explanation = f"The answer correctly addresses the query. All {len(claims)} key claims are supported by the source documents."
        else:
            if hallucinated_spans:
                spans_text = ", ".join([f'"{span}"' for span in hallucinated_spans[:3]])
                explanation = f"The answer contains {unsupported_count} unsupported claim(s). Specifically, the following span(s) are not supported by the source documents: {spans_text}."
            else:
                explanation = f"The answer contains {unsupported_count} unsupported claim(s) that are not verifiable from the provided source documents."
        
        return explanation
    
    def generate_detailed_explanation(self, query: str, answer: str, claims: list[str],
                                     support_map: dict[str, bool],
                                     hallucinated_spans: list[str]) -> str:
        """
        Generate a detailed explanation using LLM for more nuanced reasoning.
        
        Args:
            query: The original user query
            answer: The RAG-generated answer
            claims: List of extracted claims
            support_map: Dict mapping claim -> is_supported
            hallucinated_spans: List of detected hallucinated spans
            
        Returns:
            Detailed natural language explanation
        """
        supported_claims = [c for c in claims if support_map.get(c, False)]
        unsupported_claims = [c for c in claims if not support_map.get(c, False)]
        
        prompt = f"""Generate a concise 2-3 sentence explanation of this evaluation result.

Query: {query}
Answer: {answer}

Supported claims ({len(supported_claims)}): {', '.join(supported_claims[:3]) if supported_claims else 'None'}
Unsupported claims ({len(unsupported_claims)}): {', '.join(unsupported_claims) if unsupported_claims else 'None'}
Hallucinated spans: {', '.join([f'"{s}"' for s in hallucinated_spans]) if hallucinated_spans else 'None'}

Provide a brief, clear explanation focusing on accuracy and what was hallucinated or incorrect."""
        
        try:
            return self._call_llm(prompt)
        except Exception as e:
            print(f"Error generating detailed explanation: {e}")
            return self.generate_explanation(query, answer, claims, support_map, hallucinated_spans)
    
    def _call_llm(self, prompt: str) -> str:
        """Make a call to the configured LLM."""
        provider = self.llm_config.get("provider", "openai").lower()
        model = self.llm_config.get("model", "gpt-4")
        
        try:
            if provider == "anthropic":
                response = self.client.messages.create(
                    model=model,
                    max_tokens=300,
                    messages=[{"role": "user", "content": prompt}]
                )
                return response.content[0].text
            else:
                response = self.client.chat.completions.create(
                    model=model,
                    messages=[{"role": "user", "content": prompt}],
                    max_tokens=300,
                    temperature=0.3
                )
                return response.choices[0].message.content
        except Exception as e:
            raise RuntimeError(f"LLM call failed: {str(e)}")
