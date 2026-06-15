"""
Configuration management for the Prometheus evaluator.
"""
import os
from functools import lru_cache
from typing import Dict, Any


@lru_cache(maxsize=1)
def get_llm_config() -> Dict[str, Any]:
    """
    Load LLM configuration from environment variables.
    
    Returns:
        Dictionary with LLM provider details
    """
    provider = os.getenv("LLM_PROVIDER", "openai").lower()
    model = os.getenv("LLM_MODEL", "gpt-4")
    
    if provider == "anthropic":
        api_key = os.getenv("ANTHROPIC_API_KEY")
        if not api_key:
            raise ValueError("ANTHROPIC_API_KEY not set in environment")
    else:
        api_key = os.getenv("OPENAI_API_KEY")
        if not api_key:
            raise ValueError("OPENAI_API_KEY not set in environment")
    
    return {
        "provider": provider,
        "model": model,
        "api_key": api_key
    }


@lru_cache(maxsize=1)
def get_evaluation_config() -> Dict[str, Any]:
    """
    Load evaluation configuration from environment variables.
    
    Returns:
        Dictionary with evaluation settings
    """
    return {
        "faithfulness_threshold": float(os.getenv("FAITHFULNESS_THRESHOLD", "0.8")),
        "max_chunk_size": int(os.getenv("MAX_CHUNK_SIZE", "1500")),
    }


@lru_cache(maxsize=1)
def get_api_config() -> Dict[str, Any]:
    """
    Load API configuration from environment variables.
    
    Returns:
        Dictionary with API settings
    """
    return {
        "host": os.getenv("API_HOST", "0.0.0.0"),
        "port": int(os.getenv("API_PORT", "8000")),
        "debug": os.getenv("DEBUG", "false").lower() == "true"
    }


@lru_cache(maxsize=1)
def get_nli_config() -> Dict[str, Any]:
    """
    Load configuration for NLI inference providers (Hugging Face + Groq fallback).

    Environment variables used:
      - HF_API_TOKEN: Hugging Face inference API token (optional; no token may still work for public models)
      - HF_NLI_MODEL: Hugging Face model id (default: microsoft/deberta-large-mnli)
      - GROQ_API_KEY: Groq API key for fallback (optional)
      - GROQ_INFERENCE_URL: Groq inference endpoint URL for fallback (optional)
      - NLI_ENTAILMENT_THRESHOLD: score threshold to consider ENTAILMENT (default: 0.6)
    """
    return {
        "hf_api_token": os.getenv("HF_API_TOKEN", ""),
        "hf_model": os.getenv("HF_NLI_MODEL", "microsoft/deberta-large-mnli"),
        "groq_api_key": os.getenv("GROQ_API_KEY", ""),
        "groq_inference_url": os.getenv("GROQ_INFERENCE_URL", ""),
        "entailment_threshold": float(os.getenv("NLI_ENTAILMENT_THRESHOLD", "0.6"))
    }
