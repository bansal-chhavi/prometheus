"""
NLI client that prefers Hugging Face DeBERTa NLI inference and falls back to a Groq inference endpoint.

This module implements a small, dependency-light wrapper around the Hugging Face Inference API
and a generic Groq HTTP fallback. It is intentionally conservative and only uses `requests`.
"""
from __future__ import annotations
import os
import requests
import time
from typing import Tuple, Dict, Any
from src.config import get_nli_config


class NLIClient:
    """Client for performing NLI checks (premise, hypothesis) using HF DeBERTa and Groq fallback."""

    def __init__(self):
        self.cfg = get_nli_config()
        self.hf_token = self.cfg.get("hf_api_token")
        self.hf_model = self.cfg.get("hf_model")
        self.groq_key = self.cfg.get("groq_api_key")
        self.groq_url = self.cfg.get("groq_inference_url")
        self.threshold = float(self.cfg.get("entailment_threshold", 0.6))

    def predict(self, claim: str, document: str) -> Tuple[str, float]:
        """
        Predict NLI label for (premise=document, hypothesis=claim).

        Returns (label, score) where label is one of 'ENTAILMENT'|'NEUTRAL'|'CONTRADICTION' or
        a best-effort string and score is a float confidence (0-1).
        """
        # Try Hugging Face first
        try:
            label, score = self._hf_predict(claim, document)
            return label, score
        except requests.HTTPError as e:
            # If rate limited, fall back to Groq
            if e.response is not None and e.response.status_code == 429:
                return self._groq_predict(claim, document)
            raise
        except Exception:
            # Any other HF error -> try Groq fallback (if configured)
            try:
                return self._groq_predict(claim, document)
            except Exception:
                # Final fallback: return NEUTRAL with low confidence
                return "NEUTRAL", 0.0

    def _hf_predict(self, claim: str, document: str) -> Tuple[str, float]:
        """Call Hugging Face Inference API for NLI models.

        Uses the "premise"/"hypothesis" json input convention.
        """
        url = f"https://api-inference.huggingface.co/models/{self.hf_model}"
        headers = {"Accept": "application/json"}
        if self.hf_token:
            headers["Authorization"] = f"Bearer {self.hf_token}"

        payload = {
            "inputs": {
                "premise": document,
                "hypothesis": claim
            }
        }

        resp = requests.post(url, headers=headers, json=payload, timeout=30)
        if resp.status_code == 429:
            resp.raise_for_status()

        resp.raise_for_status()
        data = resp.json()

        # Typical HF NLI response is dict with 'label' and 'score' or a list of dicts
        if isinstance(data, dict) and "label" in data and "score" in data:
            return data["label"].upper(), float(data["score"])

        if isinstance(data, list) and len(data) > 0:
            # Find best label with highest score
            best = max(data, key=lambda d: d.get("score", 0))
            label = best.get("label", "NEUTRAL").upper()
            return label, float(best.get("score", 0.0))

        # Unexpected format
        return "NEUTRAL", 0.0

    def _groq_predict(self, claim: str, document: str) -> Tuple[str, float]:
        """Call Groq's OpenAI-compatible chat completions endpoint with retries.

        Expects `self.groq_url` to point to an OpenAI-compatible chat endpoint
        (for example: https://api.groq.com/openai/v1/chat/completions).
        """
        if not self.groq_url:
            raise RuntimeError("Groq inference URL not configured")

        headers = {"Content-Type": "application/json"}
        if self.groq_key:
            headers["Authorization"] = f"Bearer {self.groq_key}"

        prompt = (
            "Determine whether the claim is SUPPORTED by the document.\n"
            "Answer succinctly: SUPPORTED, NOT_SUPPORTED, or CONTRADICTED.\n\n"
            f"Document:\n{document}\n\nClaim:\n{claim}\n\nAnswer:"
        )

        payload = {
            "model": os.getenv("GROQ_MODEL", "openai/gpt-oss-120b"),
            "messages": [{"role": "user", "content": prompt}],
            "max_tokens": 120,
            "temperature": 0.0,
        }

        # Basic retry/backoff for transient errors (rate limit / 5xx)
        attempts = 3
        backoff = 1.0
        last_exc = None
        for attempt in range(1, attempts + 1):
            try:
                resp = requests.post(self.groq_url, headers=headers, json=payload, timeout=30)
                # Retry on 429 or 5xx
                if resp.status_code == 429 or 500 <= resp.status_code < 600:
                    last_exc = RuntimeError(f"Groq request failed with status {resp.status_code}")
                    time.sleep(backoff)
                    backoff *= 2
                    continue

                resp.raise_for_status()
                data = resp.json()

                # Parse OpenAI-compatible chat response
                choices = data.get("choices") or []
                if choices and isinstance(choices, list):
                    msg = choices[0].get("message") or choices[0].get("delta") or {}
                    content = msg.get("content") if isinstance(msg, dict) else None
                    if not content:
                        # Some endpoints put text at choices[0].get('text')
                        content = choices[0].get("text")
                else:
                    content = None

                if not content:
                    # Fallback to raw text body
                    content = resp.text

                text = str(content).strip().upper()

                if "SUPPORTED" in text and "NOT" not in text:
                    return "ENTAILMENT", 0.8
                if "CONTRADICT" in text or "CONTRADI" in text:
                    return "CONTRADICTION", 0.8
                if "NOT_SUPPORTED" in text or "NOT" in text:
                    return "NEUTRAL", 0.0

                if "ENTAIL" in text or "YES" in text:
                    return "ENTAILMENT", 0.7

                return "NEUTRAL", 0.0
            except requests.RequestException as e:
                last_exc = e
                # retry on network errors
                time.sleep(backoff)
                backoff *= 2
                continue

        # If we exhausted retries, raise the last exception or return neutral
        if last_exc:
            raise last_exc

        return "NEUTRAL", 0.0

    def is_entailment(self, label: str, score: float) -> bool:
        return label.upper() == "ENTAILMENT" and float(score) >= float(self.threshold)
