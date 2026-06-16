"""
Core evaluation engine: LLM-based claim extraction, evidence matching, hallucination detection.
"""
import json
import re
from typing import Tuple
from src.config import get_llm_config, get_nli_config
from src.nli_client import NLIClient


class Evaluator:
    """
    Core evaluation engine using LLM for claim extraction and evidence matching.
    """
    
    def __init__(self):
        """Initialize evaluator with optional LLM and NLI configuration."""
        self.llm_config = get_llm_config()
        self.client = None
        self.nli = None
        
        # Try to initialize LLM (optional)
        try:
            api_key = self.llm_config.get("api_key", "")
            if api_key:  # Only initialize if API key is set
                self.client = self._init_llm_client()
        except Exception as e:
            print(f"LLM client initialization skipped: {e}")
        
        # Try to initialize NLI client for claim-document entailment checks
        try:
            self.nli = NLIClient()
        except Exception as e:
            print(f"NLI client initialization warning: {e}")
    
    def _init_llm_client(self):
        """Initialize appropriate LLM client based on configuration."""
        provider = self.llm_config.get("provider", "openai").lower()
        
        if provider == "anthropic":
            try:
                import anthropic
                return anthropic.Anthropic(api_key=self.llm_config.get("api_key"))
            except ImportError:
                raise ImportError("Anthropic SDK not installed. Install with: pip install anthropic")
        else:  # Default to OpenAI
            try:
                import openai
                openai.api_key = self.llm_config.get("api_key")
                return openai.OpenAI(api_key=self.llm_config.get("api_key"))
            except ImportError:
                raise ImportError("OpenAI SDK not installed. Install with: pip install openai")
    
    def extract_claims(self, answer: str) -> list[str]:
        """
        Use LLM to decompose answer into atomic factual claims.
        
        Args:
            answer: The RAG-generated answer text
            
        Returns:
            List of atomic claims extracted from the answer
            
        Raises:
            RuntimeError: If LLM client not configured
        """
        if not self.client:
            raise RuntimeError("LLM client not configured. Cannot extract claims without LLM.")
        
        prompt = f"""Decompose the following answer into atomic factual claims. Each claim should be a single, 
verifiable statement. Return as a JSON array of strings.

Answer: {answer}

Return ONLY valid JSON array format like: ["claim 1", "claim 2", "claim 3"]"""
        
        try:
            response = self._call_llm(prompt)
            # Parse JSON response
            claims = self._parse_json_array(response)
            return claims if claims else [answer]  # Fallback to full answer if parsing fails
        except Exception as e:
            print(f"Error extracting claims: {e}")
            # Fallback: treat the entire answer as a single claim
            return [answer]
    
    def match_claim_to_evidence(self, claim: str, documents: list[str]) -> Tuple[bool, str]:
        """
        Check if a claim is supported by any document chunk using LLM.
        
        Args:
            claim: The factual claim to verify
            documents: List of source document chunks
            
        Returns:
            Tuple of (is_supported: bool, evidence_excerpt: str)
        """
        # Prefer NLI-based approach when available
        if self.nli is not None:
            try:
                # For each document chunk, run NLI (premise=document, hypothesis=claim)
                best_score = 0.0
                best_label = ""
                best_doc = ""
                for doc in documents:
                    label, score = self.nli.predict(claim, doc)
                    if score > best_score:
                        best_score = score
                        best_label = label
                        best_doc = doc

                is_supported = self.nli.is_entailment(best_label, best_score)
                evidence = best_doc if is_supported else ""
                return (is_supported, evidence)
            except Exception as e:
                print(f"NLI check failed, falling back to LLM: {e}")

        # Fallback to LLM-based check (existing behaviour)
        doc_text = "\n\n".join([f"[Doc {i+1}] {doc}" for i, doc in enumerate(documents)])
        
        prompt = f"""Given the following claim and source documents, determine if the claim is 
SUPPORTED (directly stated or clearly implied) by the documents, or NOT SUPPORTED.

Return a JSON object with exactly these fields:
{{"is_supported": true/false, "evidence_excerpt": "quoted text from docs or empty string"}}

Claim: {claim}

Documents:
{doc_text}

Return ONLY valid JSON, no other text."""
        
        try:
            response = self._call_llm(prompt)
            result = self._parse_json_object(response)
            
            is_supported = result.get("is_supported", False)
            evidence = result.get("evidence_excerpt", "")
            
            return (is_supported, evidence)
        except Exception as e:
            print(f"Error matching claim to evidence: {e}")
            return (False, "")
    
    def detect_hallucinations(self, answer: str, claims: list[str], 
                            support_map: dict[str, bool]) -> list[str]:
        """
        Identify spans in the answer corresponding to unsupported claims.
        
        Args:
            answer: The original answer text
            claims: List of extracted claims
            support_map: Dict mapping claim -> is_supported (bool)
            
        Returns:
            List of hallucinated spans from the answer
        """
        hallucinated_spans = []
        
        for claim, is_supported in support_map.items():
            if not is_supported:
                # Try to find the span in the answer using fuzzy matching
                span = self._find_span_in_text(claim, answer)
                if span:
                    hallucinated_spans.append(span)
        
        # Remove duplicates while preserving order
        seen = set()
        unique_spans = []
        for span in hallucinated_spans:
            normalized = span.lower().strip()
            if normalized not in seen:
                seen.add(normalized)
                unique_spans.append(span)
        
        return unique_spans
    
    def _find_span_in_text(self, claim: str, text: str) -> str:
        """
        Find a span in text that corresponds to the claim.
        Tries exact match first, then substring search.
        
        Args:
            claim: The claim text to find
            text: The text to search in
            
        Returns:
            The matched span or empty string
        """
        # Normalize for comparison
        claim_lower = claim.lower()
        text_lower = text.lower()
        
        # Try exact match first
        if claim_lower in text_lower:
            start = text_lower.index(claim_lower)
            return text[start:start + len(claim)]
        
        # Try finding numeric values (like percentages)
        numbers = re.findall(r'\d{1,3}(?:\.\d{1,2})?\s*%|\d+(?:,\d{3})*', claim)
        for number in numbers:
            if number in text:
                # Find the context around this number (word boundary)
                match = re.search(rf'\b{re.escape(number)}\s*(?:percent|%)?(?:\s+\w+)?', text, re.IGNORECASE)
                if match:
                    return match.group(0)
        
        return ""
    
    def _call_llm(self, prompt: str) -> str:
        """
        Make a call to the configured LLM.
        
        Args:
            prompt: The prompt to send to the LLM
            
        Returns:
            The LLM response text
        """
        provider = self.llm_config.get("provider", "openai").lower()
        model = self.llm_config.get("model", "gpt-4")
        
        try:
            if provider == "anthropic":
                response = self.client.messages.create(
                    model=model,
                    max_tokens=500,
                    messages=[{"role": "user", "content": prompt}]
                )
                return response.content[0].text
            else:  # OpenAI
                response = self.client.chat.completions.create(
                    model=model,
                    messages=[{"role": "user", "content": prompt}],
                    max_tokens=500,
                    temperature=0.1  # Low temperature for consistent extraction
                )
                return response.choices[0].message.content
        except Exception as e:
            raise RuntimeError(f"LLM call failed: {str(e)}")
    
    def _parse_json_array(self, text: str) -> list[str]:
        """Parse JSON array from text response."""
        try:
            # Find JSON array in the response
            match = re.search(r'\[.*\]', text, re.DOTALL)
            if match:
                json_str = match.group(0)
                return json.loads(json_str)
        except (json.JSONDecodeError, AttributeError):
            pass
        return []
    
    def _parse_json_object(self, text: str) -> dict:
        """Parse JSON object from text response."""
        try:
            # Find JSON object in the response
            match = re.search(r'\{.*\}', text, re.DOTALL)
            if match:
                json_str = match.group(0)
                return json.loads(json_str)
        except (json.JSONDecodeError, AttributeError):
            pass
        return {}

    def evaluate_segments_nli(self, answer: str, documents: list[str]) -> Tuple[dict, list[str]]:
        """
        Evaluate answer using NLI directly on sentence segments (No LLM required).
        
        Splits answer into sentences, checks each against documents using NLI,
        and identifies hallucinated spans.
        
        Args:
            answer: The RAG-generated answer text
            documents: List of source document chunks
            
        Returns:
            Tuple of (support_map dict, hallucinated_spans list)
        """
        if not self.nli:
            raise RuntimeError("NLI client not configured")
        
        # Split answer into sentences
        segments = self._split_into_sentences(answer)
        if not segments:
            segments = [answer]
        
        support_map = {}
        hallucinated_spans = []
        
        # Check each segment against documents using NLI
        for segment in segments:
            segment = segment.strip()
            if not segment:
                continue
            
            best_label = "NEUTRAL"
            best_score = 0.0
            
            # Check against each document
            for doc in documents:
                try:
                    label, score = self.nli.predict(segment, doc)
                    if score > best_score:
                        best_score = score
                        best_label = label
                except Exception as e:
                    print(f"NLI check failed for segment: {e}")
                    continue
            
            is_supported = self.nli.is_entailment(best_label, best_score)
            support_map[segment] = is_supported
            
            # Track unsupported segments as hallucinations
            if not is_supported:
                hallucinated_spans.append(segment)
        
        return support_map, hallucinated_spans
    
    def _split_into_sentences(self, text: str) -> list[str]:
        """
        Split text into sentences using simple heuristics.
        
        Args:
            text: Text to split
            
        Returns:
            List of sentences
        """
        # Simple sentence splitting on . ! ? followed by space or end of string
        sentences = re.split(r'(?<=[.!?])\s+', text.strip())
        # Filter out empty sentences
        sentences = [s.strip() for s in sentences if s.strip()]
        return sentences
