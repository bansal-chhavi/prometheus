import os
import pytest
from prometheus.config import PrometheusConfig, LLMProvider

def test_get_api_key_from_config():
    c = PrometheusConfig(llm_api_key="key1")
    assert c.get_api_key() == "key1"

def test_get_api_key_from_env(monkeypatch):
    monkeypatch.setenv("GROQ_API_KEY", "env_key")
    c = PrometheusConfig()
    assert c.get_api_key() == "env_key"

def test_get_api_key_missing():
    c = PrometheusConfig()
    with pytest.raises(ValueError):
        c.get_api_key()

def test_get_base_url():
    c = PrometheusConfig(llm_provider=LLMProvider.GROQ)
    assert "groq.com" in c.get_base_url()

def test_get_hf_token(monkeypatch):
    c = PrometheusConfig()
    monkeypatch.delenv("HF_API_TOKEN", raising=False)
    assert c.get_hf_token() is None
