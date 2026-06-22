"""Configuration for Prometheus."""

from __future__ import annotations

import os
from enum import Enum
from typing import Optional

from pydantic import BaseModel, Field
from dotenv import load_dotenv


# Load .env file at module import time
load_dotenv()

class LLMProvider(str, Enum):
    GROQ = "groq"
    ANTHROPIC = "anthropic"
    OPENAI = "openai"


class PrometheusConfig(BaseModel):
    """Central config — instantiate once, pass everywhere."""

    # LLM provider (Groq / OpenAI / Anthropic)
    llm_provider: LLMProvider = LLMProvider.GROQ
    llm_model: str = "llama-3.3-70b-versatile"
    llm_api_key: Optional[str] = None
    llm_base_url: Optional[str] = None

    # HuggingFace NLI
    hf_api_token: Optional[str] = None
    hf_nli_model: str = "cross-encoder/nli-deberta-v3-base"

    def get_hf_token(self) -> str | None:
        return self.hf_api_token or os.environ.get("HF_API_TOKEN")

    # Thresholds
    hallucination_threshold: float = Field(0.5)
    drift_threshold: float = Field(0.4)
    unsupported_threshold: float = Field(0.3)

    # LLM temperature
    llm_temperature: float = 0.0

    def get_api_key(self) -> str:
        if self.llm_api_key:
            return self.llm_api_key
        env_map = {
            LLMProvider.GROQ: "GROQ_API_KEY",
            LLMProvider.ANTHROPIC: "ANTHROPIC_API_KEY",
            LLMProvider.OPENAI: "OPENAI_API_KEY",
        }
        key = os.environ.get(env_map[self.llm_provider], "")
        if not key:
            raise ValueError(f"No API key. Set {env_map[self.llm_provider]}.")
        return key

    def get_base_url(self) -> str | None:
        if self.llm_base_url:
            return self.llm_base_url
        return {
            LLMProvider.GROQ: "https://api.groq.com/openai/v1",
            LLMProvider.OPENAI: "https://api.openai.com/v1",
            LLMProvider.ANTHROPIC: None,
        }[self.llm_provider]
